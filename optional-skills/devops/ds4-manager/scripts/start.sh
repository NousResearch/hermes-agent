#!/bin/bash
# DS4 Dwarfstar Dashboard — launchd wrapper
# Starts dashboard.py with the correct venv and working directory

set -euo pipefail

DASHBOARD_DIR="/Users/m4mbp/ds4-dashboard"
VENV="$DASHBOARD_DIR/.venv"
PORT="${DS4_DASHBOARD_PORT:-8765}"

cd "$DASHBOARD_DIR"

if [ ! -d "$VENV" ]; then
    echo "ERROR: Virtual environment not found at $VENV" >&2
    echo "Run: cd $DASHBOARD_DIR && python3.9 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt" >&2
    exit 1
fi

source "$VENV/bin/activate"

exec python dashboard.py \
    --port "$PORT" \
    2>&1
