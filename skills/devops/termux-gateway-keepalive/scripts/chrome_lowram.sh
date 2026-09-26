#!/usr/bin/env bash
# chrome_lowram.sh — RAM-capped single-process Chromium launcher for low-memory Android devices.
PORT=9222
PREFIX="${PREFIX:-/data/data/com.termux/files/usr}"

# Ensure Xvfb is running
if ! pgrep -x Xvfb >/dev/null 2>&1; then
  nohup Xvfb :99 -screen 0 800x600x8 > /dev/null 2>&1 &
  sleep 3
fi

export DISPLAY=:99

exec nice -n 10 "$PREFIX/lib/chromium/chromium-launcher.sh" \
  --headless --no-sandbox --disable-gpu --disable-software-rasterizer \
  --renderer-process-limit=1 --process-per-site \
  --js-flags=--max-old-space-size=192 \
  --remote-debugging-port=$PORT --remote-allow-origins=* \
  --disable-dev-shm-usage --disable-features=TranslateUI \
  --metrics-recording-only --disable-domain-reliability \
  --lang=en-US --user-data-dir="$HOME/.chromium-cdp"
