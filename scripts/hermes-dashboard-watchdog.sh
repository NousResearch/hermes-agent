#!/bin/zsh
set -u

PROJECT_DIR="/Users/kevin/codex/projects/hermes"
COMPOSE_FILE="$PROJECT_DIR/compose.hermes.local.yml"
LOG_DIR="$PROJECT_DIR/.hermes-docker/logs"
LOCK_DIR="/tmp/hermes-dashboard-watchdog.lock"
DASHBOARD_URL="http://127.0.0.1:9119/chat"

export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"

mkdir -p "$LOG_DIR"

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  exit 0
fi
trap 'rmdir "$LOCK_DIR" 2>/dev/null || true' EXIT

timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

log() {
  print -r -- "$(timestamp) $*" >> "$LOG_DIR/watchdog.log"
}

cd "$PROJECT_DIR" || exit 0

if ! docker info >/dev/null 2>&1; then
  log "docker unavailable; asking macOS to open Docker Desktop"
  open -ga Docker >/dev/null 2>&1 || true
  for _ in {1..45}; do
    docker info >/dev/null 2>&1 && break
    sleep 2
  done
fi

if ! docker info >/dev/null 2>&1; then
  log "docker still unavailable; leaving service untouched"
  exit 0
fi

if ! docker compose -f "$COMPOSE_FILE" up -d >/dev/null 2>&1; then
  log "docker compose up failed"
  exit 0
fi

http_code="$(curl -sS -o /dev/null -w "%{http_code}" --max-time 5 "$DASHBOARD_URL" 2>/dev/null || true)"
if [[ "$http_code" != "200" ]]; then
  log "dashboard HTTP check failed: ${http_code:-none}; restarting hermes"
  docker compose -f "$COMPOSE_FILE" restart hermes >/dev/null 2>&1 || log "docker compose restart failed"
  sleep 8
  http_code_after="$(curl -sS -o /dev/null -w "%{http_code}" --max-time 5 "$DASHBOARD_URL" 2>/dev/null || true)"
  log "post-restart HTTP check: ${http_code_after:-none}"
  exit 0
fi

python3 - <<'PY' >/dev/null 2>&1
import asyncio
import re
import sys
import urllib.request

try:
    import websockets
except Exception:
    sys.exit(0)

try:
    html = urllib.request.urlopen("http://127.0.0.1:9119/chat", timeout=5).read().decode("utf-8", "replace")
    token_match = re.search(r'window\.__HERMES_SESSION_TOKEN__="([^"]+)"', html)
    if not token_match:
        sys.exit(2)

    async def main():
        uri = f"ws://127.0.0.1:9119/api/pty?token={token_match.group(1)}&channel=watchdog"
        async with websockets.connect(uri, open_timeout=5) as ws:
            await asyncio.wait_for(ws.recv(), timeout=8)

    asyncio.run(main())
except Exception:
    sys.exit(3)
PY

ws_status=$?
if [[ "$ws_status" -eq 0 ]]; then
  exit 0
fi

log "dashboard websocket check failed: $ws_status; restarting hermes"
docker compose -f "$COMPOSE_FILE" restart hermes >/dev/null 2>&1 || log "docker compose restart failed"
sleep 8
http_code_after="$(curl -sS -o /dev/null -w "%{http_code}" --max-time 5 "$DASHBOARD_URL" 2>/dev/null || true)"
log "post-websocket-restart HTTP check: ${http_code_after:-none}"
