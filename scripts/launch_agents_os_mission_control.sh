#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${AGENTS_OS_REPO_DIR:-$PWD}"
HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
HOST="${AGENTS_OS_HOST:-127.0.0.1}"
PORT="${AGENTS_OS_PORT:-18790}"
URL="http://${HOST}:${PORT}"
PY="${REPO_DIR}/venv/bin/python"
LOG_DIR="${HERMES_HOME}/agents_os/logs"
LOG_FILE="${LOG_DIR}/mission-control-web.log"
PID_FILE="${HERMES_HOME}/agents_os/mission-control-web.pid"

mkdir -p "$LOG_DIR"
cd "$REPO_DIR"

if [[ ! -x "$PY" ]]; then
  echo "ERROR: venv python not found: $PY" >&2
  exit 2
fi

health_ok() {
  "$PY" - <<PY
import urllib.request, sys
try:
    with urllib.request.urlopen('${URL}/api/status', timeout=2) as r:
        data = r.read(2000).decode('utf-8', 'replace')
    sys.exit(0 if r.status == 200 and 'state_db' in data else 1)
except Exception:
    sys.exit(1)
PY
}

if health_ok; then
  echo "Mission Control is already running: $URL"
else
  echo "Starting Mission Control local-only: $URL"
  HERMES_HOME="$HERMES_HOME" nohup "$PY" -m hermes_cli.agents_os web --host "$HOST" --port "$PORT" >"$LOG_FILE" 2>&1 &
  echo $! > "$PID_FILE"
  for _ in $(seq 1 30); do
    if health_ok; then
      break
    fi
    sleep 1
  done
  if ! health_ok; then
    echo "ERROR: Mission Control did not become healthy. Log: $LOG_FILE" >&2
    exit 1
  fi
fi

echo "Health OK: $URL/api/status"
if command -v wslview >/dev/null 2>&1; then
  wslview "$URL" >/dev/null 2>&1 || true
elif command -v xdg-open >/dev/null 2>&1; then
  xdg-open "$URL" >/dev/null 2>&1 || true
else
  echo "Open manually: $URL"
fi
