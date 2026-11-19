#!/usr/bin/env bash
# termux_priority.sh — non-root OOM score & CPU priority shaper for Termux.
#
# What works without root on Android/SELinux:
#   1. termux-wake-lock: prevents deep sleep and aggressive LMK sweeps
#   2. renice: negative nice values give CPU scheduling priority to the gateway
#   3. oom_score_adj: push browser/sacrificial processes to oom_score_adj=500 and nice=19
#
set -u
LOG="$HOME/.hermes/logs/priority.log"
mkdir -p "$(dirname "$LOG")"
log() { echo "$(date '+%m-%d %H:%M') $*" >> "$LOG"; }

# 1. Wake lock
command -v termux-wake-lock >/dev/null 2>&1 && termux-wake-lock 2>/dev/null || true

# 2. Boost priority of critical processes
GATEWAY_PID=$(pgrep -f "venv/bin/hermes gateway" | head -1)
if [ -n "$GATEWAY_PID" ]; then
  renice -n -15 -p "$GATEWAY_PID" 2>/dev/null && log "gateway ($GATEWAY_PID) reniced to -15" || true
fi

WATCHDOG_PID=$(pgrep -f "ram_watchdog.sh" | head -1)
if [ -n "$WATCHDOG_PID" ]; then
  renice -n -10 -p "$WATCHDOG_PID" 2>/dev/null && log "watchdog ($WATCHDOG_PID) reniced to -10" || true
fi

HERMES_PID=$(pgrep -f "hermes-agent/venv/bin/python" | head -1)
if [ -n "$HERMES_PID" ]; then
  renice -n -10 -p "$HERMES_PID" 2>/dev/null && log "hermes python ($HERMES_PID) reniced to -10" || true
fi

# 3. Deprioritize Chrome/Chromium so kernel OOM killer targets it before the gateway
for p in $(ls /proc 2>/dev/null | grep -E '^[0-9]+$'); do
  exe=$(readlink /proc/$p/exe 2>/dev/null)
  case "$exe" in *chromium/chrome*|*chromium/*)
    echo 500 > /proc/$p/oom_score_adj 2>/dev/null || true
    renice -n 19 -p "$p" 2>/dev/null || true
  ;; esac
done 2>/dev/null
log "chrome deprioritized (oom_score=500, nice=19)"

# 4. Report status
echo "=== Termux Priority Status ==="
echo "wake-lock: active"
echo ""
for p in $(ls /proc 2>/dev/null | grep -E '^[0-9]+$'); do
  rss=$(awk '/VmRSS/{print $2}' /proc/$p/status 2>/dev/null)
  [ "${rss:-0}" -gt 30000 ] || continue
  name=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | cut -c1-50)
  oom=$(cat /proc/$p/oom_score_adj 2>/dev/null || echo "?")
  echo "  ${rss:-0}KB | oom_adj=$oom | $name"
done 2>/dev/null | sort -t'|' -k1 -rn | head -8
log "priority pass complete"
