#!/usr/bin/env bash
# PRD-8.3 — K>=2 multi-fact disambiguation cutover gate.
#
# Proves the identifier-fidelity + mandatory-escalation fix (commits 2d8c4bb40,
# 7c56b8f6d) drives intra-node confident-wrong to the statistical bar Opus set,
# WITHOUT being a fake-green from a neutered generator.
#
# Staged + fail-fast (cost-correct shape — never burn the powered arm on a
# broken harness):
#   STAGE 0  store reset (clean nodes; the harness matches nodes by text)
#   STAGE 1  BASELINE-REPRO  K=2 N=80  escalation OFF  -> MUST reproduce ~10% CW
#            (AC-5 anti-fake-green: if the generator can't reproduce the bug,
#             a later 0% is meaningless. <5% CW here ABORTS the campaign.)
#   STAGE 2  FIX-SHAKEDOWN   K=2 N=120 escalation ON   -> 0 CW gate to proceed
#   STAGE 3  POWERED GATE    K=2 N=600 escalation ON   -> rule-of-three <0.5%@95%
#            point recall >=0.95, zero confident-wrong (non-negotiable)
#   STAGE 4  K=1 RE-CERT     N=180 escalation ON       -> fix didn't regress K=1
#
# CRITICAL: venv interpreter only ('python3' = anaconda 3.7.4 PATH-poison).
# Haiku standing rule (claude-haiku-4-5); harness refuses Opus (exit 2).
# Apollo cutover stays gated on explicit user go — this only produces evidence.
set -u
cd ~/.hermes/hermes-agent

PY=~/.hermes/hermes-agent/venv/bin/python
REPORTS=docs/reports/lcm-qa
mkdir -p "$REPORTS"

# ---- pinned params (do not edit without re-review) ---------------------------
MODEL=claude-haiku-4-5
FILLER_TURNS=44
THRESHOLD=0.10
TIMEOUT=600
BASE_N=80           # baseline-repro (escalation OFF)
SHAKE_N=120         # fix shakedown (escalation ON)
POWERED_N=600       # rule-of-three: 0/600 => <0.5% CW @ 95%
RECERT_N=180        # K=1 non-regression
CW_REPRO_MIN=0.05   # AC-5: baseline must show >=5% CW or the generator is broken
MAX_USD=650         # hard cost ceiling (proj ~$566; backstop)

HB_CHANNEL=1516341560118345738
hb() { /usr/bin/python3 ~/.hermes/scripts/notify.py --send "$1" --channel discord --target "$HB_CHANNEL" >/dev/null 2>&1 || true; }

# ---- terminal-state marker (problem #2): the net cron must NEVER re-fire a
#      campaign that has already finished OR deliberately aborted. ANY exit from
#      here on writes the done-marker so a legitimate abort is a FINDING, not a
#      crash to recover from. Only an external kill (SIGKILL) skips this — and the
#      net cron's other guards (powered-report / running-proc) still hold.
DONE_MARKER=/tmp/lcm-k2-campaign.done
mark_done() { echo "$(date) $*" >> "$DONE_MARKER"; }
trap 'mark_done "exit trap rc=$?"' EXIT

# ---- per-stage clean store (problem #3): baseline-repro plants MERGED nodes;
#      if the fix stages run on the same store their sentinels collide with stale
#      baseline nodes (the harness matches nodes by text). Reset before EACH stage
#      so every stage starts from messages=0/nodes=0. Reset bounces the aegis
#      gateway (low blast radius; reversible — see lcm_reset_aegis_store.sh).
reset_store() {  # $1 = stage label
  echo "[k2] $(date) reset store before $1..."
  bash scripts/lcm_reset_aegis_store.sh --go > "/tmp/lcm-k2-reset-$1.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    hb "🛑 **K=2** store reset before $1 FAILED (rc=$rc). See /tmp/lcm-k2-reset-$1.log. Aborting."
    echo "[k2] reset before $1 failed"; tail -20 "/tmp/lcm-k2-reset-$1.log"; exit 1
  fi
  sleep 20  # let the aegis gateway settle after restart
}

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
    hb "🛑 **K=2 campaign** · cost ceiling hit (\$$s > \$$MAX_USD). Aborting."
    echo "[k2] COST CEILING \$$s > \$$MAX_USD — abort."; exit 9
  fi
}

# read a metric out of an Arm-B report json
jget() {  # $1=json $2=key
  $PY - "$1" "$2" <<'PYEOF' 2>/dev/null || echo ""
import json,sys
try:
    d=json.load(open(sys.argv[1])); print(d.get(sys.argv[2],""))
except Exception: print("")
PYEOF
}

run_k2() {  # $1=N  $2=report  $3=extra-flags
  $PY scripts/lcm_arm_b_node_recovery.py \
    --profile aegis --model "$MODEL" \
    --n "$1" --sentinels-per-session 2 \
    --filler-turns "$FILLER_TURNS" --threshold "$THRESHOLD" \
    --timeout-seconds "$TIMEOUT" --out "$2" $3
}

# ================= STAGE 0: store reset =======================================
# NOTE: the done-marker is cleared by the LAUNCHER (lcm_k2_autofire.sh) exactly
# once at fire time, NOT here — clearing here would race the 15m net cron (which
# treats a present marker as "finished"). We only ADD to it on exit (trap above).
hb "🧹 **PRD-8.3 K=2 campaign** starting · per-stage clean store, fidelity A/B. Ceiling \$$MAX_USD."
reset_store "S0-init"

# ================= STAGE 1: baseline-repro (escalation OFF) ====================
echo "[k2] $(date) STAGE 1 baseline-repro N=$BASE_N escalation OFF..."
hb "🔬 **K=2 STAGE 1** baseline-repro N=$BASE_N, **escalation OFF** — must reproduce ~10% confident-wrong (AC-5 anti-fake-green)."
run_k2 "$BASE_N" "$REPORTS/k2-baseline-repro-n${BASE_N}-noesc.md" "--no-escalation" > /tmp/lcm-k2-baseline.log 2>&1 || true
ceiling_or_die
B_CW=$(jget "$REPORTS/k2-baseline-repro-n${BASE_N}-noesc.json" confident_wrong)
B_CWRATE=$($PY -c "print(round(${B_CW:-0}/${BASE_N},4))" 2>/dev/null || echo 0)
echo "[k2] baseline CW=$B_CW rate=$B_CWRATE"
if $PY -c "import sys; sys.exit(0 if float('$B_CWRATE')>=float('$CW_REPRO_MIN') else 1)"; then
  hb "✅ **K=2 STAGE 1** baseline reproduced the bug: CW=$B_CW/$BASE_N ($B_CWRATE ≥ $CW_REPRO_MIN). Generator is sound → testing the fix."
else
  hb "🛑 **K=2 STAGE 1** baseline did NOT reproduce (CW=$B_CW/$BASE_N = $B_CWRATE < $CW_REPRO_MIN). Generator can't produce the bug → a 0% fixed result would be meaningless. ABORTING — finding, not a do-over."
  echo "[k2] baseline-repro below floor; aborting per AC-5."; exit 3
fi
ceiling_or_die

# ================= STAGE 2: fix shakedown (escalation ON) ======================
reset_store "S2-shakedown"
echo "[k2] $(date) STAGE 2 fix-shakedown N=$SHAKE_N escalation ON..."
hb "🛠️ **K=2 STAGE 2** fix-shakedown N=$SHAKE_N, escalation ON — gate to powered run = 0 confident-wrong."
run_k2 "$SHAKE_N" "$REPORTS/k2-fix-shakedown-n${SHAKE_N}.md" "" > /tmp/lcm-k2-shake.log 2>&1 || true
ceiling_or_die
S_CW=$(jget "$REPORTS/k2-fix-shakedown-n${SHAKE_N}.json" confident_wrong)
S_REC=$(jget "$REPORTS/k2-fix-shakedown-n${SHAKE_N}.json" node_served_recall_rate)
echo "[k2] shakedown CW=$S_CW recall=$S_REC"
if [ "${S_CW:-1}" != "0" ]; then
  hb "🛑 **K=2 STAGE 2** fix-shakedown still has CW=$S_CW (recall=$S_REC). The fix did NOT close confident-wrong at N=$SHAKE_N. NOT spending the powered arm. Finding, not a do-over — see report."
  echo "[k2] shakedown CW>0; not extending to powered."; exit 4
fi
hb "✅ **K=2 STAGE 2** fix-shakedown clean: 0 CW, recall=$S_REC at N=$SHAKE_N → extending to powered N=$POWERED_N."
ceiling_or_die

# ================= STAGE 3: powered gate (escalation ON) =======================
reset_store "S3-powered"
echo "[k2] $(date) STAGE 3 POWERED N=$POWERED_N escalation ON..."
hb "🚀 **K=2 STAGE 3** POWERED gate N=$POWERED_N, escalation ON (~28h). Rule-of-three: 0/$POWERED_N ⇒ <0.5% CW @95%."
run_k2 "$POWERED_N" "$REPORTS/k2-powered-gate-n${POWERED_N}.md" "" > /tmp/lcm-k2-powered.log 2>&1 || true
ceiling_or_die
P_CW=$(jget "$REPORTS/k2-powered-gate-n${POWERED_N}.json" confident_wrong)
P_REC=$(jget "$REPORTS/k2-powered-gate-n${POWERED_N}.json" node_served_recall_rate)
P_WLB=$(jget "$REPORTS/k2-powered-gate-n${POWERED_N}.json" wilson_lower_bound)
P_VERD=$(jget "$REPORTS/k2-powered-gate-n${POWERED_N}.json" verdict)
echo "[k2] powered verdict=$P_VERD CW=$P_CW recall=$P_REC wlb=$P_WLB"
ceiling_or_die

# ================= STAGE 4: K=1 re-cert (non-regression) =======================
reset_store "S4-recert"
echo "[k2] $(date) STAGE 4 K=1 re-cert N=$RECERT_N..."
hb "🔁 **K=2 STAGE 4** K=1 re-cert N=$RECERT_N — confirm the fix didn't regress the single-fact path."
$PY scripts/lcm_arm_b_node_recovery.py \
  --profile aegis --model "$MODEL" \
  --n "$RECERT_N" --sentinels-per-session 1 \
  --filler-turns "$FILLER_TURNS" --threshold "$THRESHOLD" \
  --timeout-seconds "$TIMEOUT" \
  --out "$REPORTS/k2-recert-k1-n${RECERT_N}.md" > /tmp/lcm-k2-recert.log 2>&1 || true
ceiling_or_die
R_CW=$(jget "$REPORTS/k2-recert-k1-n${RECERT_N}.json" confident_wrong)
R_REC=$(jget "$REPORTS/k2-recert-k1-n${RECERT_N}.json" node_served_recall_rate)

# ================= FINAL ======================================================
FINAL_SPEND=$(spend_since_t0)
hb "🏁 **PRD-8.3 K=2 disambiguation campaign DONE** · spend \$$FINAL_SPEND
**S1 baseline-repro** (esc OFF, N=$BASE_N): CW=$B_CW ($B_CWRATE) — bug reproduced ✅
**S2 fix-shakedown** (esc ON, N=$SHAKE_N): CW=$S_CW recall=$S_REC
**S3 POWERED gate** (esc ON, N=$POWERED_N): **verdict=$P_VERD** CW=$P_CW recall=$P_REC wilson=$P_WLB
**S4 K=1 re-cert** (N=$RECERT_N): CW=$R_CW recall=$R_REC
Reports in $REPORTS/. Apollo cutover gated on your explicit go."
echo "[k2] $(date) DONE spend=\$$FINAL_SPEND"
