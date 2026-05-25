#!/bin/bash
set -eu

MODE="${1:-gateway}"

mkdir -p "${HERMES_HOME:-/opt/data}"
export HERMES_HOME="${HERMES_HOME:-/opt/data}"

source /opt/hermes/.venv/bin/activate

echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] Starting Hermes $MODE in $HERMES_HOME"

case "$MODE" in
  gateway)
    echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] Gateway mode: listening for messaging platforms"
    exec hermes gateway run
    ;;
  dashboard)
    echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] Dashboard mode: web UI on 0.0.0.0:9119"
    exec hermes dashboard --host 0.0.0.0 --port 9119 --no-open
    ;;
  *)
    echo "Unknown mode: $MODE" >&2
    echo "Usage: $0 {gateway|dashboard}" >&2
    exit 1
    ;;
esac
