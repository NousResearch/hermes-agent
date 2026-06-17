#!/usr/bin/env bash
# One-shot: ping #hermes-lcm once Arm-A passes ~half its 180 trials, then exit.
# Progress proxy = completed-trial count in the live run log (falls back to elapsed time).
set -u
cd ~/.hermes/hermes-agent
ARM_A_PID="${1:-7198}"
LOG="${2:-/tmp/lcm-arm-a-n180.log}"
HB_CHANNEL=1516341560118345738
hb() { /usr/bin/python3 ~/.hermes/scripts/notify.py --send "$1" --channel discord --target "$HB_CHANNEL" >/dev/null 2>&1 || true; }

START=$(date +%s)
HALF_SECS=$((50*60))   # ~halfway of a ~100min run, time-based backstop
while kill -0 "$ARM_A_PID" 2>/dev/null; do
  DONE=$(grep -ciE 'trial .*(pass|fail|recall|done)|^\[trial' "$LOG" 2>/dev/null || echo 0)
  NOW=$(date +%s); ELAPSED=$((NOW-START))
  if [ "${DONE:-0}" -ge 90 ] || [ "$ELAPSED" -ge "$HALF_SECS" ]; then
    hb "💓 **LCM campaign** · Arm A (raw-store N=180) ~halfway — ${DONE:-?} trials logged, ${ELAPSED}s elapsed, still grinding. Final verdict + Arm B to follow."
    exit 0
  fi
  sleep 60
done
# Arm A ended before we pinged halfway (fast finish) — campaign watcher handles the done-ping.
exit 0
