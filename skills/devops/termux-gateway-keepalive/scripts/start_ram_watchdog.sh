#!/usr/bin/env bash
# start_ram_watchdog.sh — launch RAM watchdog in its own detached session (idempotent)
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ALREADY=$(for p in $(ls /proc 2>/dev/null | grep -E '^[0-9]+$'); do
  c=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null)
  if echo "$c" | grep -q "ram_watchdog.sh"; then
    echo "$p"
    break
  fi
done)

if [ -n "$ALREADY" ]; then
  echo "RAM watchdog already running: $ALREADY"
else
  setsid bash "$SCRIPT_DIR/ram_watchdog.sh" > /dev/null 2>&1 < /dev/null &
  sleep 3
  WD=$(for p in $(ls /proc 2>/dev/null | grep -E '^[0-9]+$'); do
    c=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null)
    if echo "$c" | grep -q "ram_watchdog.sh"; then
      echo "$p"
      break
    fi
  done)
  echo "RAM watchdog started: ${WD:-FAILED}"
fi
tail -1 "$HOME/.hermes/logs/ram_watchdog.log" 2>/dev/null || true
cat "$HOME/.hermes/ram_watchdog.state" 2>/dev/null || true
echo ""
