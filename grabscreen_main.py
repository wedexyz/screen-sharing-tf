
import time
import cv2
import mss
import numpy
#from PIL import ImageGrab
#from grabscreen import grab_screen

import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

title = "uji coba screen graber"
start_time = time.time()
display_time = 2 # displays the frame rate every 2 second
fps = 0
sct = mss.mss()
# Set monitor size to capture
monitor = {"top": 40, "left": 0, "width": 800, "height": 640}
mon = (0, 40, 800, 640)

thres = 0.6
classNames= []
classFile = 'coco.names'
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'
net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

def screen_recordMSS():
    global fps, start_time
    while True:
        #success,img = sct.grab(monitor)
        # Get raw pixels from the screen, save it to a Numpy array
        img = numpy.array(sct.grab(monitor))
        # To get real color we do this:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #img_np_expanded = numpy.expand_dims(img, axis=0)
        # Display the picture
        real = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Display the picture in grayscale

        classIds, confs, bbox = net.detect(real,confThreshold=thres)
        if len(classIds) != 0:
            for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
                cv2.rectangle(real,box,color=(0,255,0),thickness=2)
                cv2.putText(real,classNames[classId-1].upper(),(box[0]+10,box[1]+30), cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,0),1)
                cv2.putText(real,str(round(confidence*100,2))+' %',(box[0]+100,box[1]+30),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,255),1)

        fps+=1
        TIME = time.time() - start_time
        if (TIME) >= display_time :
            print("FPS: ", fps / (TIME))
            fps = 0
            start_time = time.time()
        #cv2.imshow("Output",img)
       # cv2.imshow(title, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        cv2.imshow(title, real)
        # Press "q" to quit
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break



'''

def screen_recordPIL():
    global fps, start_time
    while True:
        # Get raw pixels from the screen, save it to a Numpy array
        img = numpy.asarray(ImageGrab.grab(bbox=mon))
        # Display the picture
        cv2.imshow(title, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # Display the picture in grayscale
        fps+=1
        TIME = time.time() - start_time
        if (TIME) >= display_time :
            print("FPS: ", fps / (TIME))
            fps = 0
            start_time = time.time()
        # Press "q" to quit
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break

def screen_grab():
    global fps, start_time
    while True:
        # Get raw pixels from the screen 
        img = grab_screen(region=mon)
        # Display the picture
        cv2.imshow(title, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # Display the picture in grayscale
        fps+=1
        TIME = time.time() - start_time
        if (TIME) >= display_time :
            print("FPS: ", fps / (TIME))
            fps = 0
            start_time = time.time()
        # Press "q" to quit
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break
'''

screen_recordMSS()
#screen_recordPIL()
#screen_grab()