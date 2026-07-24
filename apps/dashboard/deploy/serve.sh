#!/usr/bin/env bash
# HERMES//HUB — dead-simple always-on runner. No Docker, no systemd, no root.
# Backgrounds the server, writes a PID file, survives terminal close (nohup).
#
#   ./deploy/serve.sh start    # start in the background
#   ./deploy/serve.sh stop     # stop it
#   ./deploy/serve.sh status   # is it running?
#   ./deploy/serve.sh logs     # follow the log
#
# Config via environment (all optional):
#   PORT=8787  DATA_DIR=~/.hermes-hub  HERMES_HUB_TOKEN=change-me
#   HERMES_HUB_API_KEY=sk-ant-...   (enables the live Claude agent)
set -euo pipefail

cd "$(dirname "$0")/.."                 # apps/dashboard
PORT="${PORT:-8787}"
DATA_DIR="${DATA_DIR:-$HOME/.hermes-hub}"
PID_FILE="$DATA_DIR/hub.pid"
LOG_FILE="$DATA_DIR/hub.log"
mkdir -p "$DATA_DIR"

running() { [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; }

case "${1:-start}" in
  start)
    if running; then echo "already running (pid $(cat "$PID_FILE"))"; exit 0; fi
    nohup python3 server.py --host 0.0.0.0 --port "$PORT" --data-dir "$DATA_DIR" \
      >>"$LOG_FILE" 2>&1 &
    echo $! >"$PID_FILE"
    sleep 1
    running && echo "started on http://localhost:$PORT (pid $(cat "$PID_FILE"))" \
            || { echo "failed to start — see $LOG_FILE"; exit 1; }
    ;;
  stop)
    if running; then kill "$(cat "$PID_FILE")" && echo "stopped"; else echo "not running"; fi
    rm -f "$PID_FILE"
    ;;
  status)
    running && echo "running (pid $(cat "$PID_FILE")) on port $PORT" || echo "not running"
    ;;
  logs)
    tail -f "$LOG_FILE"
    ;;
  *)
    echo "usage: $0 {start|stop|status|logs}"; exit 2
    ;;
esac
