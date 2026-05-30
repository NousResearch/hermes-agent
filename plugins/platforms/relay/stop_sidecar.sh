#!/bin/bash
# 停止 cc-connect sidecar
# 用法: bash stop_sidecar.sh

set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
PID_FILE="$DIR/sidecar.pid"

if [ ! -f "$PID_FILE" ]; then
    echo "[sidecar] No PID file, not running"
    exit 0
fi

PID=$(cat "$PID_FILE")
if kill -0 "$PID" 2>/dev/null; then
    echo "[sidecar] Stopping PID=$PID..."
    kill "$PID"
    # 等待退出
    for i in $(seq 1 10); do
        if ! kill -0 "$PID" 2>/dev/null; then
            echo "[sidecar] Stopped"
            rm -f "$PID_FILE"
            exit 0
        fi
        sleep 0.5
    done
    # 强制杀
    echo "[sidecar] Force killing PID=$PID..."
    kill -9 "$PID" 2>/dev/null || true
    rm -f "$PID_FILE"
    echo "[sidecar] Force stopped"
else
    echo "[sidecar] PID=$PID not running, cleaning up"
    rm -f "$PID_FILE"
fi
