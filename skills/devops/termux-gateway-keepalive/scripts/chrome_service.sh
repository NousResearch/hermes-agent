#!/usr/bin/env bash
# chrome_service.sh — start or stop headless Chromium service.
PORT=9222
PREFIX="${PREFIX:-/data/data/com.termux/files/usr}"

action="${1:-start}"

if [ "$action" = "stop" ]; then
  if [ -d "$PREFIX/var/service" ]; then
    cd "$PREFIX/var/service" && SVDIR=$PWD sv down chromium-headless 2>/dev/null || true
  fi
  pkill -f "chromium/chrome" 2>/dev/null || true
  pkill -x Xvfb 2>/dev/null || true
  echo "Chromium service stopped."
  exit 0
fi

if ! pgrep -x Xvfb >/dev/null 2>&1; then
  setsid Xvfb :99 -screen 0 800x600x8 > /dev/null 2>&1 < /dev/null &
  sleep 3
fi

if ! curl -s -m 3 "http://127.0.0.1:$PORT/json/version" >/dev/null 2>&1; then
  DISPLAY=:99 \
  setsid nice -n 10 "$PREFIX/lib/chromium/chromium-launcher.sh" \
    --headless --no-sandbox --disable-gpu --disable-software-rasterizer \
    --renderer-process-limit=1 --process-per-site \
    --no-zygote --single-process \
    --remote-debugging-port=$PORT --remote-allow-origins=* \
    --disable-dev-shm-usage --disable-features=TranslateUI --lang=en-US \
    --metrics-recording-only --disable-domain-reliability \
    --user-data-dir="$HOME/.chromium-cdp" \
    > "$HOME/.hermes/logs/chrome_headless.log" 2>&1 < /dev/null &
  sleep 10
fi

curl -s -m 5 "http://127.0.0.1:$PORT/json/version" >/dev/null 2>&1 && echo "CHROME UP" || echo "CHROME FAILED"
