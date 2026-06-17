#!/usr/bin/env bash
# Sequential Arm-A -> Arm-B campaign runner (PRD-7 boil-the-ocean).
# Waits for the Arm-A N>=180 gate to finish, then runs the Arm-B N>=180
# node-served gate. Sequential to avoid Aegis gateway / lcm.db WAL contention.
set -u
cd ~/.hermes/hermes-agent

ARM_A_PID="${1:-7198}"
REPORTS=docs/reports/lcm-qa
mkdir -p "$REPORTS"

echo "[campaign] waiting for Arm-A pid $ARM_A_PID to finish..."
while kill -0 "$ARM_A_PID" 2>/dev/null; do
  sleep 30
done
echo "[campaign] Arm-A finished at $(date). Starting Arm-B N=180..."

# Arm B: full statistical node-served gate. ~5 min/trial * 180 is long; this is
# the deliberate boil-the-ocean run. Writes incrementally per trial to stderr log.
python scripts/lcm_arm_b_node_recovery.py \
  --profile aegis --model claude-haiku-4-5 \
  --n 180 --filler-turns 44 --threshold 0.10 \
  --timeout-seconds 600 \
  --out "$REPORTS/arm-b-n180-haiku.md" \
  > /tmp/lcm-arm-b-n180.log 2>&1

echo "[campaign] Arm-B finished at $(date). Verdicts:"
echo "--- ARM A ---"; grep -iE "verdict|wilson|recall|correct" "$REPORTS/arm-a-n180-haiku.md" 2>/dev/null | head
echo "--- ARM B ---"; grep -iE "verdict|wilson|recall|condensation|preserved" "$REPORTS/arm-b-n180-haiku.md" 2>/dev/null | head
