#!/usr/bin/env bash
# wd_dedupe.sh — ensure exactly one gateway watchdog is running (kill duplicates).
HOME_DIR="$HOME"
WD=""

for p in $(ls /proc 2>/dev/null | grep -E '^[0-9]+$'); do
  c=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null)
  case "$c" in *gateway_watchdog*start*) WD="$WD $p";; esac
done

COUNT=$(echo $WD | wc -w)
echo "Watchdogs found:$WD ($COUNT)"

if [ "$COUNT" -gt 1 ]; then
  KEEP=$(echo $WD | awk '{print $1}')
  for p in $WD; do
    [ "$p" != "$KEEP" ] && kill "$p" 2>/dev/null && echo "Killed duplicate watchdog pid $p"
  done
  echo "Kept primary watchdog: $KEEP"
  echo "$KEEP" > "$HOME_DIR/.hermes/gateway_watchdog.pid"
elif [ "$COUNT" -eq 1 ]; then
  echo "Single watchdog active OK."
else
  echo "No watchdog active — launching supervisor..."
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  setsid bash "$SCRIPT_DIR/gateway_watchdog.sh" start >/dev/null 2>&1 &
  sleep 2
fi
