#!/bin/bash
# Lab entrypoint: Xvfb + Firefox + loopback x11vnc + noVNC.
# Inside the container we listen on 0.0.0.0:6080. docker-compose MUST publish
# 127.0.0.1:6080 on the host. Do not change the compose ports to 0.0.0.0.
set -euo pipefail
DISPLAY_NUM="${DISPLAY_NUM:-98}"
export DISPLAY=":${DISPLAY_NUM}"
Xvfb ":${DISPLAY_NUM}" -screen 0 1280x720x24 -nolisten tcp &
sleep 0.5
firefox-esr --no-remote "https://example.com/" &
x11vnc -display ":${DISPLAY_NUM}" -forever -shared -nopw -quiet \
       -rfbport 5900 -nossl -listen 127.0.0.1 -localhost &
exec websockify --web /usr/share/novnc 0.0.0.0:6080 127.0.0.1:5900
