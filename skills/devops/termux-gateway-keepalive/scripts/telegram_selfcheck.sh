#!/usr/bin/env bash
# telegram_selfcheck.sh — verify gateway Telegram delivery health.
# Runs from cron (called by gateway_monitor.sh every 10 min).
# Checks:
#   1. gateway process alive
#   2. gateway_state.json exists and platforms.telegram.state == 'connected'
#   3. state file freshness (updated within 15 min) — stale 'connected' treated as DOWN
#
set -u
HOME_DIR="$HOME"
STATE="$HOME_DIR/.hermes/gateway_state.json"
LOG="$HOME_DIR/.hermes/logs/telegram_selfcheck.log"
mkdir -p "$(dirname "$LOG")"
NOW=$(date +%s)
ok=1
reasons=""

# 1) process alive?
if ! pgrep -f "venv/bin/hermes gateway" >/dev/null 2>&1; then
  ok=0
  reasons="$reasons [gateway process DEAD]"
fi

# 2) state file exists?
if [ ! -f "$STATE" ]; then
  ok=0
  reasons="$reasons [gateway_state.json missing]"
else
  # 3) telegram state connected?
  tg=$(python3 -c "
import json,sys
try:
    d=json.load(open('$STATE'))
    print(d.get('platforms',{}).get('telegram',{}).get('state','unknown'))
except Exception:
    print('unreadable')
" 2>/dev/null)
  if [ "$tg" != "connected" ]; then
    ok=0
    reasons="$reasons [telegram state=$tg]"
  fi
  # 4) state file fresh? (mtime within 900s)
  mt=$(stat -c %Y "$STATE" 2>/dev/null || echo 0)
  age=$(( NOW - mt ))
  if [ "$age" -gt 900 ]; then
    ok=0
    reasons="$reasons [state STALE: ${age}s old]"
  fi
fi

ts=$(date '+%Y-%m-%d %H:%M:%S')
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ "$ok" -eq 1 ]; then
  echo "$ts selfcheck: OK" >> "$LOG"
else
  echo "$ts selfcheck: FAIL -$reasons" >> "$LOG"
  python3 "$SCRIPT_DIR/hermes_presence.py" "⚠ Telegram gateway check FAILED:$reasons. Watchdog recovering." 2>/dev/null || true
  if ! pgrep -f "venv/bin/hermes gateway" >/dev/null 2>&1; then
    bash "$SCRIPT_DIR/gateway_watchdog.sh" start >> "$LOG" 2>&1 || true
  fi
fi
