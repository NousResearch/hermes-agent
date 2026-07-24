#!/usr/bin/env bash
# HERMES//HUB — one-click launcher (macOS/Linux).
#   ./start.sh
# Pulls the latest build, opens the dashboard, and starts the local server.
set -euo pipefail
cd "$(dirname "$0")"

echo "Updating to the latest build..."
git pull origin main || echo "(skipped git pull)"

URL="http://127.0.0.1:8787"
( sleep 2; (command -v open >/dev/null && open "$URL") || (command -v xdg-open >/dev/null && xdg-open "$URL") ) >/dev/null 2>&1 &

echo "Starting HERMES//HUB on $URL  (Ctrl+C to stop)"
exec python3 server.py
