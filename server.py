#!/usr/bin/env python3

import os
import time
import datetime
from bottle import route, run, template, request, static_file
import json
import sys
import base64
import cv2
import numpy as np
from backend.inpaint import sttn_inpaint
from backend import config

@route('/gettxtremoved', method='POST')
def get_txtremoved():
    origin = request.files.get('origin')
    mask = request.files.get('mask')
    
    name, ext = os.path.splitext(origin.filename)
    mask_name, mask_ext = os.path.splitext(mask.filename)
    if ext.lower() not in ('.png','.jpg','.jpeg'):
        return "File extension not allowed."
    if mask_ext.lower() not in ('.png','.jpg','.jpeg'):
        return "File extension not allowed."        
        
    timestamp=str(int(time.time()*1000))
    origin_savedName=timestamp+ext
    mask_savedName=timestamp+"-mask"+mask_ext
    ouputName=timestamp+"-text-removed.jpg"
    
    save_path = "./uploaded/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    origin_path = "{path}/{file}".format(path=save_path, file=origin_savedName)
    mask_path = "{path}/{file}".format(path=save_path, file=mask_savedName)
    output_path = "{path}/{file}".format(path=save_path, file=ouputName)
    origin.save(origin_path)        
    mask.save(mask_path)
    process_mask(mask_path)
    img = cv2.imread(origin_path)
    images = []
    images.append(img)
    sttn_images_inpaint = sttn_inpaint.STTNImagesInpaint(images, mask_path, clip_gap=config.STTN_MAX_LOAD_NUM)
    frames = sttn_images_inpaint()
    inpainted = frames[0]
    cv2.imwrite(output_path, inpainted)
    os.remove(mask_path)
    os.remove(origin_path)
    return static_file(ouputName, root='uploaded')

    
def process_mask(mask_path):
    mask_img = cv2.imread(mask_path)
    gray=cv2.cvtColor(mask_img,cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)[1]
    cv2.imwrite(mask_path, thresh)


@route('/gettxtremoved_folder', method='POST')
def get_txtremoved_folder():
    folder_path = request.forms.get('folder')
    mask = request.files.get('mask')
    
    mask_name, mask_ext = os.path.splitext(mask.filename)

    if mask_ext.lower() not in ('.png','.jpg','.jpeg'):
        return "File extension not allowed."        
        
    timestamp=str(int(time.time()*1000))
    mask_savedName=timestamp+"-mask"+mask_ext

    
    save_path = "./uploaded/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    mask_path = "{path}/{file}".format(path=save_path, file=mask_savedName)
    mask.save(mask_path)
    
    process_mask(mask_path)
    
    images = []
    images_path = []
    for filename in os.listdir(folder_path):
        name, ext = os.path.splitext(filename)
        if ext.lower() not in ('.png','.jpg','.jpeg'):
            continue
        path = os.path.join(folder_path,filename)
        images_path.append(path)
        images.append(cv2.imread(path))
        
    sttn_images_inpaint = sttn_inpaint.STTNImagesInpaint(images, mask_path, clip_gap=config.STTN_MAX_LOAD_NUM)
    frames = sttn_images_inpaint()
    frame_index = 0
    for frame in frames:
        cv2.imwrite(images_path[frame_index],frame)
        frame_index = frame_index + 1
    os.remove(mask_path)
    return "done"
    
@route('/<filepath:path>')
def server_static(filepath):
    return static_file(filepath, root='www')

if __name__ == '__main__':
    if len(sys.argv)==2:
        service_port=sys.argv[1]
    else:
        service_port=8189
    run(host='127.0.0.1', port=service_port)   
