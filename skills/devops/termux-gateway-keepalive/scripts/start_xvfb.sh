#!/usr/bin/env bash
# start_xvfb.sh — ensure Xvfb virtual framebuffer is running on :99.
if pgrep -x Xvfb >/dev/null 2>&1; then
  echo "Xvfb already running"
else
  nohup Xvfb :99 -screen 0 1024x600x8 > /dev/null 2>&1 &
  for i in 1 2 3 4 5; do
    sleep 1
    if pgrep -x Xvfb >/dev/null 2>&1; then break; fi
  done
fi

if pgrep -x Xvfb >/dev/null 2>&1; then
  echo "Xvfb READY (pid $(pgrep -x Xvfb | head -1))"
else
  echo "Xvfb FAILED to start"
  exit 1
fi
