#!/bin/bash
set -eu

MODE="${1:-gateway}"

mkdir -p "${HERMES_HOME:-/opt/data}"
export HERMES_HOME="${HERMES_HOME:-/opt/data}"

source /opt/hermes/.venv/bin/activate

# When HERMES_SKIP_SETUP=true, ensure Hermes bypasses its first-run provider
# check. The check passes as long as any recognised provider env var is set;
# if OPENROUTER_API_KEY is already present (e.g. injected by Railway) it will
# be picked up automatically. If it is absent we set a sentinel so the check
# does not abort startup — the real key can be configured via the web UI once
# the service is running.
if [ "${HERMES_SKIP_SETUP:-}" = "true" ]; then
  echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] HERMES_SKIP_SETUP=true: skipping provider setup check"
  if [ -z "${OPENROUTER_API_KEY:-}" ]; then
    export OPENROUTER_API_KEY="placeholder-configure-via-dashboard"
    echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] No OPENROUTER_API_KEY found; set placeholder to bypass setup check"
  fi
fi

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
