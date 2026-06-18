#!/usr/bin/env bash
# LCM Phase-2 campaign — PRD-8.1 probe-isolation contract.
#   Arm A: raw-store/FTS gate, EXACT probes only, per-run THROWAWAY lcm.db,
#          N=120 shakedown -> fixed extend predicate -> N=180.
#   Arm B: DAG node-served gate, K=1 (one fact per node), N=180.
# Sequential (Aegis gateway / lcm.db contention). Self-contained.
#
# CRITICAL: venv interpreter only — 'python3' on this fleet is anaconda 3.7.4
# (PATH-poison) which cannot import the LCM engine in-process.
#
# PRD-8.1 load-bearing, pinned params (AC-7) — do not edit without re-review:
PROBE_KIND=exact            # Arm A contract = exact-string FTS (PRD-7 §4.1)
FILLER_TURNS=36             # deep bury to force store eviction
SENTINELS_PER_SESSION=1     # Arm B: one (owner,phrase) per node, no intra-node confusion
ARMA_SHAKEDOWN_N=120        # staged Arm A
ARMA_FULL_N=180
ARMB_N=180
VOID_RATE_MAX=0.20          # AC-2 hard stop
MAX_USD=120                 # AC-6 hard cost ceiling — BELOW ~$150 worst case, a real checkpoint
set -u
cd ~/.hermes/hermes-agent

PY=~/.hermes/hermes-agent/venv/bin/python
REPORTS=docs/reports/lcm-qa
mkdir -p "$REPORTS"
ARMA_DB="/tmp/lcm-arma-throwaway-$(date +%s).db"   # C4: isolate Arm A FTS from prior sentinels

HB_CHANNEL=1516341560118345738
hb() { /usr/bin/python3 ~/.hermes/scripts/notify.py --send "$1" --channel discord --target "$HB_CHANNEL" >/dev/null 2>&1 || true; }

# ---- cost ceiling check (AC-6): cumulative Aegis Blackbox spend since t0 -------
T0=$(date +%s)
spend_since_t0() {
  $PY - "$T0" <<'PYEOF' 2>/dev/null || echo 0
import sqlite3, os, sys
t0=float(sys.argv[1])
db=os.path.expanduser("~/.hermes/profiles/aegis/blackbox/turns.db")
try:
    c=sqlite3.connect(f"file:{db}?mode=ro",uri=True)
    print(c.execute("select coalesce(sum(cost_usd),0) from turns where ts_start>=?",(t0,)).fetchone()[0])
except Exception:
    print(0)
PYEOF
}
ceiling_or_die() {
  local s; s=$(spend_since_t0)
  if $PY -c "import sys; sys.exit(0 if float('$s')>float('$MAX_USD') else 1)"; then
    hb "🛑 **LCM campaign** · cost ceiling hit (\$$s > \$$MAX_USD). Aborting."
    echo "[campaign] COST CEILING \$$s > \$$MAX_USD — abort."; exit 9
  fi
}

# ---- AC-7 enforcing param-pin: the report's recorded run_params MUST match the
#      pinned contract, else abort with no verdict (CB-2: raises, not prints) ----
assert_contract() {  # $1=report.json
  local rc; rc=$($PY - "$1" "$PROBE_KIND" "$FILLER_TURNS" <<'PYEOF' 2>/dev/null || echo FAIL
import json,sys
try:
    d=json.load(open(sys.argv[1])); rp=d.get("run_params",{})
    ok = (rp.get("probe_kind")==sys.argv[2]
          and str(rp.get("filler_turns"))==sys.argv[3]
          and rp.get("void_redraw") is True)
    print("OK" if ok else "MISMATCH:%s" % rp)
except Exception as e:
    print("FAIL:%s" % e)
PYEOF
)
  if [ "$rc" != "OK" ]; then
    hb "🛑 **LCM campaign** · AC-7 contract assert FAILED on $1 ($rc). Pinned probe=$PROBE_KIND filler=$FILLER_TURNS void_redraw=true. Aborting, no verdict."
    echo "[campaign] AC-7 CONTRACT MISMATCH ($rc) — abort."; exit 7
  fi
  echo "[campaign] AC-7 contract OK for $1"
}
run_arm_a() {  # $1=N  $2=report
  $PY scripts/lcm_live_recovery.py --live --session-mode --profile aegis \
    --model claude-haiku-4-5 --n "$1" --lcm-threshold 0.02 \
    --probe-kind "$PROBE_KIND" --filler-turns "$FILLER_TURNS" --filler-tokens 2500 \
    --lcm-db "$ARMA_DB" --void-redraw --void-rate-max "$VOID_RATE_MAX" \
    --out "$2"
}

hb "🚀 **LCM campaign (PRD-8.1)** · Arm A EXACT-only shakedown N=$ARMA_SHAKEDOWN_N, throwaway db, $FILLER_TURNS-turn bury. Ceiling \$$MAX_USD."
echo "[campaign] $(date) ARM A shakedown N=$ARMA_SHAKEDOWN_N (probe=$PROBE_KIND, db=$ARMA_DB)..."
run_arm_a "$ARMA_SHAKEDOWN_N" "$REPORTS/arm-a-n${ARMA_SHAKEDOWN_N}-exact.md" > /tmp/lcm-arm-a-shake.log 2>&1
ceiling_or_die
assert_contract "$REPORTS/arm-a-n${ARMA_SHAKEDOWN_N}-exact.json"

# fixed extend predicate (AC-6): recall>=0.95 AND wilson>=0.90 AND void_rate<=0.20
EXTEND=$($PY - "$REPORTS/arm-a-n${ARMA_SHAKEDOWN_N}-exact.json" "$VOID_RATE_MAX" <<'PYEOF' 2>/dev/null || echo no
import json,sys
d=json.load(open(sys.argv[1])); vmax=float(sys.argv[2])
rec=d.get("point_recall",0); wil=d.get("wilson_lower",0); vr=d.get("void_rate",0)
print("yes" if (rec>=0.95 and wil>=0.90 and vr<=vmax) else "no")
PYEOF
)
SHAKE_V=$(grep -iE "verdict|recall|wilson|void" "$REPORTS/arm-a-n${ARMA_SHAKEDOWN_N}-exact.md" 2>/dev/null | head -5 | tr '\n' ' ')
echo "[campaign] $(date) Arm A shakedown extend=$EXTEND :: $SHAKE_V"
if [ "$EXTEND" != "yes" ]; then
  hb "🛑 **Arm A** shakedown N=$ARMA_SHAKEDOWN_N did NOT meet extend predicate (recall≥0.95 ∧ wilson≥0.90 ∧ void≤$VOID_RATE_MAX). NOT extending. ${SHAKE_V} — finding, not a do-over."
  echo "[campaign] Arm A shakedown failed predicate; stopping per AC-6 (no re-draw)."
else
  hb "✅ **Arm A** shakedown passed predicate. Extending to N=$ARMA_FULL_N."
  echo "[campaign] $(date) ARM A full N=$ARMA_FULL_N..."
  run_arm_a "$ARMA_FULL_N" "$REPORTS/arm-a-n${ARMA_FULL_N}-exact.md" > /tmp/lcm-arm-a-full.log 2>&1
  ceiling_or_die
  ARM_A_V=$(grep -iE "verdict|recall|wilson|void" "$REPORTS/arm-a-n${ARMA_FULL_N}-exact.md" 2>/dev/null | head -4 | tr '\n' ' ')
  hb "💓 **Arm A** full N=$ARMA_FULL_N done. ${ARM_A_V:-see report}"
fi

# drain before Arm B in-process reads the live store
sleep 45
ceiling_or_die

# ---- ARM B: DAG node-served gate, K=1 -----------------------------------------
echo "[campaign] $(date) full Arm-B N=$ARMB_N (K=$SENTINELS_PER_SESSION/session)..."
hb "🚀 **Arm B** DAG node-served N=$ARMB_N, K=$SENTINELS_PER_SESSION (one fact/node)."
$PY scripts/lcm_arm_b_node_recovery.py \
  --profile aegis --model claude-haiku-4-5 \
  --n "$ARMB_N" --sentinels-per-session "$SENTINELS_PER_SESSION" \
  --filler-turns 44 --threshold 0.10 --timeout-seconds 600 \
  --out "$REPORTS/arm-b-n${ARMB_N}-k1-haiku.md" \
  > /tmp/lcm-arm-b-n180-k1.log 2>&1
ceiling_or_die

# ---- AC-3b: NON-GATING K=2 characterization (Opus GI-4) ------------------------
# Real Apollo windows will sometimes co-locate >=2 facts. Measure the intra-node
# disambiguation failure rate once (N=40, NOT a gate) so cutover sees the real
# co-location risk, not just the K=1 floor.
echo "[campaign] $(date) AC-3b K=2 characterization (N=40, non-gating)..."
hb "📊 **Arm B char** · non-gating K=2 N=40 (intra-node disambiguation rate, Opus GI-4)."
$PY scripts/lcm_arm_b_node_recovery.py \
  --profile aegis --model claude-haiku-4-5 \
  --n 40 --sentinels-per-session 2 \
  --filler-turns 44 --threshold 0.10 --timeout-seconds 600 \
  --out "$REPORTS/arm-b-char-k2-n40.md" \
  > /tmp/lcm-arm-b-char-k2.log 2>&1 || true
ceiling_or_die
CHAR_V=$(grep -iE "recall|confident|disambig" "$REPORTS/arm-b-char-k2-n40.md" 2>/dev/null | head -3 | tr '\n' ' ')

echo "[campaign] $(date) DONE."
ARM_B_V=$(grep -iE "verdict|recall|wilson|preserved|confident" "$REPORTS/arm-b-n${ARMB_N}-k1-haiku.md" 2>/dev/null | head -6 | tr '\n' ' ')
FINAL_SPEND=$(spend_since_t0)
hb "🏁 **LCM campaign (PRD-8.1) DONE** · spend \$$FINAL_SPEND
**Arm A** (raw-store EXACT): ${ARM_A_V:-skipped/failed predicate}
**Arm B** (DAG node-served K=1 N=$ARMB_N): ${ARM_B_V:-see report}
**Arm B char** (K=2 non-gating): ${CHAR_V:-see report}
Reports in $REPORTS/. Apollo cutover gated on your go."
rm -f "$ARMA_DB" "$ARMA_DB"-* 2>/dev/null || true
