#!/bin/bash
set -e

echo "=== Dual-Ring Gate · Outer Ring ==="
echo "[CHECK] Time: $(date '+%Y-%m-%d %H:%M %A')"

GW_STATUS=$(hermes gateway status 2>&1)
if echo "$GW_STATUS" | grep -q "running"; then
    echo "[PASS] Gateway running"
else
    echo "[FAIL] Gateway not running, starting..."
    hermes gateway run --replace
    sleep 2
    hermes gateway status | grep -q "running" || {
        echo "[FATAL] Gateway start failed"
        exit 1
    }
    echo "[PASS] Gateway started"
fi

echo "=== Outer Ring Passed ==="
