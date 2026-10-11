#!/usr/bin/env bash
# gateway_monitor.sh — backup keeper for the gateway watchdog.
# Runs from cron (every 10 min). If the gateway AND the watchdog are both
# dead, it relaunches the watchdog (which re-acquires wake-lock + gateway).
set -u
HOME_DIR="$HOME"

# Telegram delivery self-check (runs every tick, cheap)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "$SCRIPT_DIR/telegram_selfcheck.sh" 2>/dev/null || true

if pgrep -f "hermes-agent/venv/bin/hermes gateway" >/dev/null 2>&1; then
    if [ ! -f "$HOME_DIR/.hermes/gateway_watchdog.run" ]; then
        setsid bash "$SCRIPT_DIR/gateway_watchdog.sh" start >> "$HOME_DIR/.hermes/logs/gateway_watchdog.log" 2>&1 &
    fi
    exit 0
fi

if [ ! -f "$HOME_DIR/.hermes/gateway_watchdog.run" ]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') monitor: gateway+watchdog down, relaunching watchdog" >> "$HOME_DIR/.hermes/logs/gateway_watchdog.log"
    setsid bash "$SCRIPT_DIR/gateway_watchdog.sh" start >> "$HOME_DIR/.hermes/logs/gateway_watchdog.log" 2>&1 &
fi
