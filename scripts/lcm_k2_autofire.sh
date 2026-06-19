#!/usr/bin/env bash
# PRD-8.3 autofire: wait for the in-flight Arm A re-run (pid passed as $1) to
# exit, then launch the staged K=2 disambiguation campaign DETACHED with its own
# logfile. macOS has no setsid; use nohup + explicit
# redirection to a tracked file, and double-forks so it survives this shell.
set -u
cd ~/.hermes/hermes-agent || exit 0

ARMA_PID="${1:?usage: autofire <arma_pid>}"
HB_CHANNEL=1516341560118345738
LOG=/tmp/lcm-k2-campaign.out
hb() { /usr/bin/python3 ~/.hermes/scripts/notify.py --send "$1" --channel discord --target "$HB_CHANNEL" >/dev/null 2>&1 || true; }

hb "⏳ **K=2 autofire armed** · waiting on Arm A (pid $ARMA_PID) to finish before launching the staged disambiguation campaign. No gateway collision."

# poll until the Arm A process is gone. Sentinel pid 1 (launchd) = "fire now"
# (used by the reboot-survival net when Arm A is already gone).
if [ "$ARMA_PID" != "1" ]; then
  while kill -0 "$ARMA_PID" 2>/dev/null; do
    sleep 60
  done
fi

# Arm A done. Brief drain so the aegis gateway flush settles before reset.
sleep 90

hb "🟢 **K=2 autofire firing** · Arm A (pid $ARMA_PID) exited. Launching staged K=2 campaign (reset → baseline-repro → shakedown → powered → K=1 re-cert). Log: $LOG"

# Clear the terminal done-marker exactly ONCE, here at fire time (NOT inside the
# campaign, which would race the 15m net cron). After this the campaign owns the
# marker: it only appends on exit, so a finished/aborted run stays "done".
rm -f /tmp/lcm-k2-campaign.done

# fully detach the campaign so it outlives this watchdog
nohup bash ~/.hermes/hermes-agent/scripts/lcm_k2_disambig_campaign.sh > "$LOG" 2>&1 < /dev/null &
disown 2>/dev/null || true

echo "[autofire] launched K=2 campaign, log=$LOG"
