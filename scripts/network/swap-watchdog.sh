#!/usr/bin/env bash
# swap-watchdog.sh — silent when healthy, alerts if swap exceeds threshold
# Used by: swap-watchdog cron (no_agent mode)
set -euo pipefail
# P13: disabled-staging guard — exit early when cron is disabled
if [ "${DRY_RUN:-0}" = "1" ]; then echo "[DRY_RUN] $(basename "$0")"; exit 0; fi


THRESHOLD=75  # alert if swap usage % exceeds this

total=$(awk '/SwapTotal/{print $2}' /proc/meminfo)
free=$(awk '/SwapFree/{print $2}' /proc/meminfo)

if [ -z "$total" ] || [ "$total" -eq 0 ]; then
  # Silent when healthy — no swap means no pressure
  exit 0
fi

used=$(( total - free ))
pct=$(( used * 100 / total ))

if [ "$pct" -gt "$THRESHOLD" ]; then
  ram_used=$(free -h | awk '/^Mem:/{print $3}')
  ram_total=$(free -h | awk '/^Mem:/{print $2}')
  swap_used=$(free -h | awk '/^Swap:/{print $3}')
  swap_total=$(free -h | awk '/^Swap:/{print $2}')
  cat <<ALERT
⚠️  Swap pressure alert: ${pct}% used (${swap_used}/${swap_total})
   RAM: ${ram_used}/${ram_total}
   Threshold: ${THRESHOLD}%

   Trigger: t_aa0d19f7 was archived — monitor was deployed instead.
   Action: check active gateways + MCP orphans, or consider VPS upgrade.
ALERT
  exit 0
fi

# Silent on healthy — no output means no delivery
