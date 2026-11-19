#!/usr/bin/env bash
# start_browser_stack.sh — bring up runsvdir (if dead), chromium service, and verify CDP.
PREFIX="${PREFIX:-/data/data/com.termux/files/usr}"

if [ -d "$PREFIX/var/service" ]; then
  cd "$PREFIX/var/service"
  if ! pgrep -x runsvdir >/dev/null 2>&1; then
    echo "starting runsvdir..."
    SVDIR=$PWD setsid runsvdir "$PWD" > /dev/null 2>&1 < /dev/null &
    sleep 3
  fi

  if ! curl -s -m 3 "http://127.0.0.1:9222/json/version" >/dev/null 2>&1; then
    echo "starting chromium-headless service..."
    SVDIR=$PWD sv up chromium-headless 2>/dev/null || true
    for i in $(seq 1 8); do
      sleep 2
      if curl -s -m 3 "http://127.0.0.1:9222/json/version" >/dev/null 2>&1; then break; fi
    done
  fi
fi

if curl -s -m 5 "http://127.0.0.1:9222/json/version" >/dev/null 2>&1; then
  VER=$(curl -s -m 5 "http://127.0.0.1:9222/json/version" | python3 -c "import sys,json; print(json.load(sys.stdin).get('Browser','?'))" 2>/dev/null || echo "alive")
  echo "BROWSER UP: $VER"
else
  echo "BROWSER FAILED to start"
fi
