#!/bin/bash
# DS4 Restart Helper
# Restarts the DS4 inference engine after dashboard config changes
# Supports both direct and launchd-managed DS4

set -euo pipefail

DS4_BINARY="${DS4_BINARY:-/Users/m4mbp/ds4/ds4-server}"
DS4_LAUNCHD_LABEL="com.ds4.server"

echo "DS4 Restart Helper"
echo "=================="

# Check if launchd-managed
if launchctl list | grep -q "$DS4_LAUNCHD_LABEL"; then
    echo "DS4 is launchd-managed — restarting via launchctl..."
    launchctl kickstart "gui/$(id -u)/$DS4_LAUNCHD_LABEL"
    echo "Done. Verify: curl -s http://127.0.0.1:8001/telem | jq .state"
    exit 0
fi

# Direct restart
PID=$(lsof -ti:8001 2>/dev/null || true)
if [ -n "$PID" ]; then
    echo "Killing DS4 on port 8001 (PID $PID)..."
    kill "$PID" 2>/dev/null || true
    sleep 2
fi

echo "Starting DS4..."
exec "$DS4_BINARY" \
    --model "$DS4_MODEL" \
    --port 8001 \
    --ctx-size "${DS4_CONTEXT_WINDOW:-131072}" \
    --mtp-path "$DS4_MTP" \
    --kv-cache-path "$DS4_KV_CACHE" \
    --cache-capacity "$DS4_KV_CACHE_BUDGET_MIB" \
    "$@"
