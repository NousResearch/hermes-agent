#!/usr/bin/env bash
# PRD-8.3 K=2 campaign — durable reboot-survival safety-net.
# Runs every 15m (no_agent cron). Idempotent guard:
#   - if the K=2 campaign already ran (powered report exists) -> self-disable marker, exit
#   - if the campaign is RUNNING (script proc) -> exit (nothing to do)
#   - if the autofire watchdog is alive -> exit (it will fire)
#   - if Arm A is still running -> re-arm a watchdog ONLY if none exists
#   - if NOTHING is alive and no powered report -> Arm A died; arm a watchdog on
#     a sentinel pid (1) so the campaign fires on next tick (gateway is free).
set -u
cd ~/.hermes/hermes-agent || exit 0

REPORTS=docs/reports/lcm-qa
POWERED="$REPORTS/k2-powered-gate-n600.json"
DONE_MARKER=/tmp/lcm-k2-campaign.done
HB_CHANNEL=1516341560118345738
hb() { /usr/bin/python3 ~/.hermes/scripts/notify.py --send "$1" --channel discord --target "$HB_CHANNEL" >/dev/null 2>&1 || true; }

# already complete?
if [ -f "$POWERED" ] || [ -f "$DONE_MARKER" ]; then
  touch "$DONE_MARKER"
  echo "[k2-net] campaign complete (powered report or done-marker present) — self-disable."
  # remove this cron so it stops ticking
  hermes cron remove lcm-k2-autofire-net >/dev/null 2>&1 || true
  exit 0
fi

# campaign script running?
if pgrep -f "lcm_k2_disambig_campaign.sh" >/dev/null 2>&1; then
  echo "[k2-net] campaign running — ok."; exit 0
fi

# watchdog alive?
if pgrep -f "lcm_k2_autofire.sh" >/dev/null 2>&1; then
  echo "[k2-net] autofire watchdog alive — ok."; exit 0
fi

# Arm A still running? re-arm a watchdog on its pid.
ARMA_PID=$(pgrep -f "lcm_live_recovery.py .*--profile aegis" | head -1)
if [ -n "${ARMA_PID:-}" ]; then
  hb "♻️ **K=2 net** · watchdog was absent while Arm A (pid $ARMA_PID) still runs — re-arming."
  setsid bash scripts/lcm_k2_autofire.sh "$ARMA_PID" > /tmp/lcm-k2-autofire.out 2>&1 < /dev/null &
  echo "[k2-net] re-armed watchdog on Arm A pid $ARMA_PID."; exit 0
fi

# nothing alive, no powered report -> Arm A is gone; fire on sentinel pid 1
hb "♻️ **K=2 net** · no Arm A, no watchdog, no powered report after reboot/loss — gateway free, arming campaign now."
setsid bash scripts/lcm_k2_autofire.sh 1 > /tmp/lcm-k2-autofire.out 2>&1 < /dev/null &
echo "[k2-net] armed watchdog on sentinel pid 1 (immediate fire)."
exit 0
