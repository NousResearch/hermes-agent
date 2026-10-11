#!/usr/bin/env bash
# browse_once.sh — start browser on demand, execute task, shut it down immediately.
# Usage: browse_once.sh "https://example.com" [duration_seconds]
set -u

URL="${1:?Usage: browse_once.sh URL [seconds]}"
DURATION="${2:-120}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[browse_once] starting browser stack for ${DURATION}s..."
bash "$SCRIPT_DIR/start_browser_stack.sh"

if ! curl -s -m 5 http://127.0.0.1:9222/json/version >/dev/null 2>&1; then
  echo "[browse_once] FAILED to start browser"
  exit 1
fi

echo "[browse_once] navigating to $URL"
python3 -c "
import urllib.request, json
c = urllib.request.urlopen('http://127.0.0.1:9222/json/new?' + '$URL')
" 2>/dev/null || true

echo "[browse_once] browsing session active — sleeping ${DURATION}s"
sleep "$DURATION"

echo "[browse_once] shutting down browser (freeing ~400MB RAM)"
if [ -d "/data/data/com.termux/files/usr/var/service" ]; then
  cd /data/data/com.termux/files/usr/var/service && SVDIR=$PWD sv down chromium-headless 2>/dev/null || true
fi
bash "$SCRIPT_DIR/chrome_zombie_hunt.sh" >/dev/null 2>&1 || true
pkill -x Xvfb 2>/dev/null || true

RAM=$(free -m 2>/dev/null | awk 'NR==2{print $7}' || echo "?")
echo "[browse_once] done. RAM available: ${RAM}MB"
