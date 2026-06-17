#!/usr/bin/env bash
# Arm-A -> validate-Arm-B-batch -> full Arm-B N=180 campaign (PRD-7, batched).
# Sequential after Arm A to avoid Aegis gateway / lcm.db contention.
set -u
cd ~/.hermes/hermes-agent

ARM_A_PID="${1:-7198}"
REPORTS=docs/reports/lcm-qa
mkdir -p "$REPORTS"
K=10   # sentinels per session

echo "[campaign] $(date) waiting for Arm-A pid $ARM_A_PID ..."
while kill -0 "$ARM_A_PID" 2>/dev/null; do sleep 30; done
echo "[campaign] $(date) Arm-A finished."

# --- Validate the batched harness on ONE session of K sentinels first. ---
echo "[campaign] $(date) validating batched Arm-B (1 session, K=$K)..."
python scripts/lcm_arm_b_node_recovery.py \
  --profile aegis --model claude-haiku-4-5 \
  --n "$K" --sentinels-per-session "$K" \
  --filler-turns 44 --threshold 0.10 --timeout-seconds 600 \
  --out "$REPORTS/arm-b-validate-k${K}.md" \
  > /tmp/lcm-arm-b-validate.log 2>&1
VAL_RECALL=$(python -c "import json;d=json.load(open('$REPORTS/arm-b-validate-k${K}.json'));print(d['node_served_recall'])" 2>/dev/null || echo 0)
echo "[campaign] $(date) validation node_served_recall=$VAL_RECALL / $K"

# Require the batch to produce at least a few real node-served recoveries before
# committing to the full run (else the batched layout is broken; bail loudly).
if [ "${VAL_RECALL:-0}" -lt 3 ]; then
  echo "[campaign] VALIDATION WEAK ($VAL_RECALL/$K node-served). NOT launching full N=180."
  echo "[campaign] Inspect /tmp/lcm-arm-b-validate.log + $REPORTS/arm-b-validate-k${K}.md"
  exit 1
fi

# --- Full Arm-B N=180, batched K per session (~18 sessions). ---
echo "[campaign] $(date) validation OK. Launching full Arm-B N=180 (K=$K/session)..."
python scripts/lcm_arm_b_node_recovery.py \
  --profile aegis --model claude-haiku-4-5 \
  --n 180 --sentinels-per-session "$K" \
  --filler-turns 44 --threshold 0.10 --timeout-seconds 600 \
  --out "$REPORTS/arm-b-n180-haiku.md" \
  > /tmp/lcm-arm-b-n180.log 2>&1

echo "[campaign] $(date) DONE. Verdicts:"
echo "--- ARM A ---"; grep -iE "verdict|wilson|recall|correct" "$REPORTS/arm-a-n180-haiku.md" 2>/dev/null | head
echo "--- ARM B ---"; grep -iE "verdict|wilson|recall|condensation|preserved" "$REPORTS/arm-b-n180-haiku.md" 2>/dev/null | head
