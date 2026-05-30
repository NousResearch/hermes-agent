#!/bin/bash
# 启动 cc-connect sidecar
# 用法: bash start_sidecar.sh

set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
PID_FILE="$DIR/sidecar.pid"
LOG_FILE="$DIR/sidecar.log"

# 检查是否已运行
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "[sidecar] Already running (PID=$PID)"
        exit 0
    else
        echo "[sidecar] Stale PID file, removing..."
        rm -f "$PID_FILE"
    fi
fi

# 检查 Python（优先用 Hermes venv 的 Python，有 aiohttp）
HERMES_PYTHON="$HOME/.hermes/hermes-agent/venv/bin/python"
if [ -x "$HERMES_PYTHON" ] && "$HERMES_PYTHON" -c "import aiohttp" 2>/dev/null; then
    PYTHON="$HERMES_PYTHON"
    echo "[sidecar] Using Hermes venv Python: $PYTHON"
else
    PYTHON=$(command -v python3 || echo "")
    if [ -z "$PYTHON" ]; then
        echo "[sidecar] ERROR: python3 not found"
        exit 1
    fi
    echo "[sidecar] WARNING: Using system Python (aiohttp may not be available)"
fi

echo "[sidecar] Starting..."
cd "$DIR"
nohup "$PYTHON" cc_sidecar.py >> "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"
echo "[sidecar] Started (PID=$(cat "$PID_FILE")), log: $LOG_FILE"
