#!/usr/bin/env bash
# Full LCM Phase-2 campaign (PRD-7): tightened Arm A (raw-store) -> Arm-B
# semantic validation (proven K=4) -> full Arm-B N=180 (DAG node-served).
# Sequential to avoid Aegis gateway / lcm.db contention. Self-contained: runs
# Arm A itself (does NOT wait on an external pid).
#
# CRITICAL: must run under the venv interpreter — 'python3' on this fleet is
# anaconda 3.7.4 (PATH-poison) which cannot import the LCM engine in-process.
set -u
cd ~/.hermes/hermes-agent

PY=~/.hermes/hermes-agent/venv/bin/python
REPORTS=docs/reports/lcm-qa
mkdir -p "$REPORTS"
K=4   # semantic probes per session — validated 4/4 node-served recall (v3)

HB_CHANNEL=1516341560118345738
hb() { /usr/bin/python3 ~/.hermes/scripts/notify.py --send "$1" --channel discord --target "$HB_CHANNEL" >/dev/null 2>&1 || true; }

# ---- ARM A: tightened raw-store/FTS gate (N=180) ------------------------------
# Root cause of the prior 13 unproven trials: only 4 filler turns followed the
# sentinel, but LCM keeps a ~32-message fresh tail, so the fact often never left
# the active window -> no store lookup forced. Deeper bury (36 filler turns)
# guarantees eviction so every trial must hit the store for tool-call evidence.
hb "🚀 **LCM campaign** · starting tightened Arm A (raw-store N=180, 36 filler turns to force eviction). ~2h."
echo "[campaign] $(date) ARM A (tightened) starting..."
$PY scripts/lcm_live_recovery.py --live --session-mode --profile aegis \
  --model claude-haiku-4-5 --n 180 --lcm-threshold 0.02 \
  --filler-turns 36 --filler-tokens 2500 \
  --out "$REPORTS/arm-a-n180-haiku.md" \
  > /tmp/lcm-arm-a-n180.log 2>&1
ARM_A_EXIT=$?
ARM_A_VERDICT=$(grep -iE "verdict|recall|wilson|evidence" "$REPORTS/arm-a-n180-haiku.md" 2>/dev/null | head -5 | tr '\n' ' ')
echo "[campaign] $(date) ARM A exit=$ARM_A_EXIT :: $ARM_A_VERDICT"
hb "💓 **LCM campaign** · Arm A done (exit $ARM_A_EXIT). ${ARM_A_VERDICT:-(see report)} — draining, then Arm-B validation."

# Drain: let the gateway flush its final session into lcm.db before Arm B's
# in-process engine reads it.
sleep 45

# ---- ARM B validation: proven semantic K=4 batch ------------------------------
echo "[campaign] $(date) Arm-B semantic validation (K=$K)..."
$PY scripts/lcm_arm_b_node_recovery.py \
  --profile aegis --model claude-haiku-4-5 \
  --n "$K" --sentinels-per-session "$K" \
  --filler-turns 44 --threshold 0.10 --timeout-seconds 600 \
  --out "$REPORTS/arm-b-validate-k${K}.md" \
  > /tmp/lcm-arm-b-validate.log 2>&1
VAL_RECALL=$($PY -c "import json;print(json.load(open('$REPORTS/arm-b-validate-k${K}.json'))['node_served_recall'])" 2>/dev/null || echo 0)
echo "[campaign] $(date) validation node_served_recall=$VAL_RECALL / $K"
if [ "${VAL_RECALL:-0}" -lt 3 ]; then
  echo "[campaign] VALIDATION WEAK ($VAL_RECALL/$K). NOT launching full N=180."
  hb "🛑 **LCM campaign** · Arm-B validation FAILED ($VAL_RECALL/$K node-served). Not launching full N=180. Logs: /tmp/lcm-arm-b-validate.log"
  exit 1
fi
hb "✅ **LCM campaign** · Arm-B validation OK ($VAL_RECALL/$K node-served). Launching full N=180 (~45 sessions, ~5h). Drain + go."
sleep 30

# ---- ARM B full: N=180 semantic node-served gate ------------------------------
echo "[campaign] $(date) full Arm-B N=180 (K=$K/session)..."
$PY scripts/lcm_arm_b_node_recovery.py \
  --profile aegis --model claude-haiku-4-5 \
  --n 180 --sentinels-per-session "$K" \
  --filler-turns 44 --threshold 0.10 --timeout-seconds 600 \
  --out "$REPORTS/arm-b-n180-haiku.md" \
  > /tmp/lcm-arm-b-n180.log 2>&1

echo "[campaign] $(date) DONE."
echo "--- ARM A ---"; grep -iE "verdict|wilson|recall|evidence" "$REPORTS/arm-a-n180-haiku.md" 2>/dev/null | head
echo "--- ARM B ---"; grep -iE "verdict|wilson|recall|preserved|condensation" "$REPORTS/arm-b-n180-haiku.md" 2>/dev/null | head
ARM_A_V=$(grep -iE "verdict|recall|wilson|evidence" "$REPORTS/arm-a-n180-haiku.md" 2>/dev/null | head -4 | tr '\n' ' ')
ARM_B_V=$(grep -iE "verdict|recall|wilson|preserved" "$REPORTS/arm-b-n180-haiku.md" 2>/dev/null | head -4 | tr '\n' ' ')
hb "🏁 **LCM campaign DONE**
**Arm A** (raw-store N=180): ${ARM_A_V:-see report}
**Arm B** (DAG node-served N=180): ${ARM_B_V:-see report}
Reports in $REPORTS/. Apollo cutover gated on your go."
