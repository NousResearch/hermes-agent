#!/usr/bin/env bash
# Evidence Spine P3–P4 — committed controller selector manifest (R5).
#
# R5: the selector's cleanliness and portability gates cover BOTH
# repositories:
#   * exact core import provenance (this worktree);
#   * exact dashboard path/SHA bound via EVIDENCE_SPINE_DASHBOARD(+_SHA);
#   * core path/SHA exported for dashboard cross-repository tests
#     (EVIDENCE_SPINE_CORE / EVIDENCE_SPINE_CORE_SHA, dashboard tests
#     require the exact core commit SHA);
#   * pre/post status captured SEPARATELY for core and dashboard and both
#     must be unchanged or the selector fails;
#   * zero hardcoded governance-evidence-spine worktree prefixes enforced
#     across committed test/selector files in BOTH repositories;
#   * collection, execution, order-1/order-2 and dashboard-suite exits are
#     preserved independently and all reported;
#   * unique disposable HERMES_HOME and --basetemp per invocation.
#
# Run from the core candidate worktree:
#   scripts/evidence-spine-selector.sh
#
# Exit codes:
#   0 = all gates green (collection, execution, both orders, both-repo
#       cleanliness, provenance bindings all enforced)
#   non-zero otherwise (the failing stage is named on stderr)
set -u

SELECTOR_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SELECTOR_DIR/.." && pwd)"
cd "$REPO_ROOT" || { echo "SEL_FAIL: cannot cd REPO_ROOT" >&2; exit 2; }

PYTHON="${PYTHON:-/home/kensei/repos/KenseiAgent/.venv/bin/python3}"
EVIDENCE_DIR="${EVIDENCE_DIR:-/tmp/r5-evidence}"
mkdir -p "$EVIDENCE_DIR"
STAMP="$(date +%Y%m%dT%H%M%SZ)-$$"
EVIDENCE_FILE="$EVIDENCE_DIR/selector-collect-$STAMP.txt"
RUN_LOG="$EVIDENCE_DIR/selector-run-$STAMP.txt"

# ── disposable homes / basetemps (per invocation) ─────────────────────────
DISP_HOME="$(mktemp -d /tmp/r5-sel-home.XXXXXX)"
BASETEMP="$(mktemp -d /tmp/r5-sel-basetemp.XXXXXX)"
trap 'rm -rf "$DISP_HOME" "$BASETEMP"' EXIT
export HERMES_HOME="$DISP_HOME"

# ── provenance: core import must resolve to THIS checkout ──────────────────
CORE_SHA="$(git rev-parse HEAD)"
CORE_DIR="$(git rev-parse --show-toplevel)"
if [ "$CORE_DIR" != "$REPO_ROOT" ]; then
  echo "SEL_FAIL: git toplevel ($CORE_DIR) != selector repo root ($REPO_ROOT)" >&2
  exit 2
fi
export REPO_ROOT
IMPORT_PROVENANCE="$("$PYTHON" -c "
import sys, os
sys.path.insert(0, os.environ['REPO_ROOT'])
import hermes_cli
print(hermes_cli.__file__)
")"
case "$IMPORT_PROVENANCE" in
  "$REPO_ROOT"/*) echo "PROVENANCE: $IMPORT_PROVENANCE" ;;
  *) echo "SEL_FAIL: import provenance ($IMPORT_PROVENANCE) is not in $REPO_ROOT" >&2; exit 2 ;;
esac
echo "CORE_SHA=$CORE_SHA"

# ── provenance: dashboard checkout must match the expected tuple SHA ──────
# R3-5/R4-2: dashboard checkout is supplied EXPLICITLY; no baked-in absolute
# paths anywhere in the committed test/selector set.
if [ -z "${EVIDENCE_SPINE_DASHBOARD:-}" ]; then
  echo "SEL_FAIL: EVIDENCE_SPINE_DASHBOARD is required (bind the dashboard checkout explicitly)" >&2
  exit 2
fi
if [ -z "${EVIDENCE_SPINE_DASHBOARD_SHA:-}" ]; then
  echo "SEL_FAIL: EVIDENCE_SPINE_DASHBOARD_SHA is required (bind the dashboard commit explicitly)" >&2
  exit 2
fi
DASH_ROOT="$EVIDENCE_SPINE_DASHBOARD"
DASH_SHA_EXPECTED="$EVIDENCE_SPINE_DASHBOARD_SHA"
if [ ! -d "$DASH_ROOT" ]; then
  echo "SEL_FAIL: dashboard checkout missing: $DASH_ROOT" >&2
  exit 2
fi
DASH_SHA="$(git -C "$DASH_ROOT" rev-parse HEAD)"
if [ "$DASH_SHA" != "$DASH_SHA_EXPECTED" ]; then
  echo "SEL_FAIL: dashboard HEAD $DASH_SHA != expected $DASH_SHA_EXPECTED" >&2
  exit 2
fi
echo "DASH_SHA=$DASH_SHA"
export EVIDENCE_SPINE_DASHBOARD="$DASH_ROOT"
export EVIDENCE_SPINE_DASHBOARD_SHA="$DASH_SHA_EXPECTED"

# ── R4-2/R4-3: export exact core bindings for dashboard cross-repo tests ──
export EVIDENCE_SPINE_CORE="$REPO_ROOT"
export EVIDENCE_SPINE_CORE_SHA="$CORE_SHA"
echo "EVIDENCE_SPINE_CORE=$EVIDENCE_SPINE_CORE"
echo "EVIDENCE_SPINE_CORE_SHA=$EVIDENCE_SPINE_CORE_SHA"

# ── R5-3: broad hardcoded worktree prefix across ALL tracked content ────────
# Assemble at runtime so this selector does not contain the prohibited token.
SCAN_PAT="worktrees/governance-evidence-""spine-"
CORE_PREFIX_HITS="$(git -C "$REPO_ROOT" grep -n -I -e "$SCAN_PAT" -- . 2>&1)"
CORE_PREFIX_EXIT=$?
if [ "$CORE_PREFIX_EXIT" -gt 1 ]; then
  echo "SEL_FAIL: core hardcoded-prefix scan failed with exit $CORE_PREFIX_EXIT" >&2
  echo "$CORE_PREFIX_HITS" >&2
  exit 5
fi
DASH_PREFIX_HITS="$(git -C "$DASH_ROOT" grep -n -I -e "$SCAN_PAT" -- . 2>&1)"
DASH_PREFIX_EXIT=$?
if [ "$DASH_PREFIX_EXIT" -gt 1 ]; then
  echo "SEL_FAIL: dashboard hardcoded-prefix scan failed with exit $DASH_PREFIX_EXIT" >&2
  echo "$DASH_PREFIX_HITS" >&2
  exit 5
fi
PREFIX_HITS="${CORE_PREFIX_HITS}${CORE_PREFIX_HITS:+$'\n'}${DASH_PREFIX_HITS}"
if [ -n "$PREFIX_HITS" ]; then
  echo "SEL_FAIL: hardcoded governance-evidence-spine worktree prefix found:" >&2
  echo "$PREFIX_HITS" >&2
  exit 5
fi
echo "HARDCODED_PREFIX_SCAN: 0 matches (core+dashboard)"

# ── R4-3.4: pre-run status captured SEPARATELY for core and dashboard ──────
PRE_STATUS_CORE="$(git -C "$REPO_ROOT" status --porcelain --untracked-files=normal | sort)"
PRE_STATUS_DASH="$(git -C "$DASH_ROOT" status --porcelain --untracked-files=normal | sort)"

FILES=(
  # ── prior controller selector (34 files) ──
  tests/hermes_cli/test_profile_activity_ledger.py
  tests/gateway/test_delivery_profile_activity_ledger.py
  tests/cron/test_jobs_profile_activity_ledger.py
  tests/hermes_cli/test_profile_activity_ledger_kanban.py
  tests/hermes_cli/test_governance_findings.py
  tests/hermes_cli/test_governance_findings_kanban.py
  tests/scripts/test_w1_batch1_ops_wiring.py
  tests/scripts/test_governance_crossref.py
  tests/scripts/test_denji_self_eval_submit.py
  tests/scripts/test_denji_review_cycle.py
  tests/tools/test_delegation_activity_ledger.py
  tests/test_hermaguard_gate.py
  tests/test_hermaguard_gate_p13.py
  tests/test_hermaguard_release_gate.py
  tests/test_phase_a_pipeline.py
  tests/test_phase_d_audit_wiring.py
  tests/test_pipeline_fixes.py
  tests/test_pipeline_execution_integrity.py
  tests/test_pipeline_claim_cycle.py
  tests/test_dashboard_pipeline.py
  tests/hermes_cli/test_profile_registry.py
  tests/scripts/test_denji_review_four_dimensions.py
  tests/scripts/test_denji_self_eval_trigger.py
  tests/hermes_cli/test_skill_evidence.py
  tests/hermes_cli/test_shadow_classifier.py
  tests/hermes_cli/test_tiered_review_preservation.py
  tests/hermes_cli/test_hermaguard_events.py
  tests/hermes_cli/test_pilot_evidence.py
  tests/hermes_cli/test_hermaguard_corrections.py
  tests/hermes_cli/test_shadow_corrections.py
  tests/scripts/test_self_eval_honesty.py
  tests/hermes_cli/test_pilot_provenance.py
  tests/scripts/test_review_authority_corrections.py
  tests/hermes_cli/test_registry_failclosed.py
  # ── R2/R3/R4 correction tests ──
  tests/hermes_cli/test_r2_corrections.py
)

# ── collection gate: real exit, evidence file, exact count ─────────────────
COLLECT_EXIT=0
"$PYTHON" -m pytest --collect-only -q "${FILES[@]}" > "$EVIDENCE_FILE" 2>&1
COLLECT_EXIT=$?
if [ "$COLLECT_EXIT" -ne 0 ]; then
  echo "SEL_FAIL: collection exited $COLLECT_EXIT (log: $EVIDENCE_FILE)" >&2
  exit 3
fi
COLLECTED_COUNT="$(grep -E '^[0-9]+ tests? collected' "$EVIDENCE_FILE" | tail -1 | awk '{print $1}')"
if [ -z "$COLLECTED_COUNT" ]; then
  echo "SEL_FAIL: could not parse collected count from $EVIDENCE_FILE" >&2
  exit 3
fi
EXPECTED_NODES=490
echo "COLLECTED: $COLLECTED_COUNT (expected $EXPECTED_NODES, log: $EVIDENCE_FILE)"
if [ "$COLLECTED_COUNT" != "$EXPECTED_NODES" ]; then
  echo "SEL_FAIL: collected $COLLECTED_COUNT != expected $EXPECTED_NODES" >&2
  exit 3
fi

# ── R4-3.10: dashboard-native backend tests as a DISTINCT stage ────────────
DASH_LOG="$EVIDENCE_DIR/dashboard-tests-$STAMP.txt"
DASH_COLLECT_LOG="$EVIDENCE_DIR/dashboard-collect-$STAMP.txt"
(cd "$DASH_ROOT" && "$PYTHON" -m pytest --collect-only -q backend/tests/test_profile_registry_hierarchy.py) > "$DASH_COLLECT_LOG" 2>&1
DASH_COLLECT_EXIT=$?
DASH_COLLECTED="$(grep -E '^[0-9]+ tests? collected' "$DASH_COLLECT_LOG" | tail -1 | awk '{print $1}')"
DASH_EXPECTED_NODES=34
echo "DASH_COLLECTED: $DASH_COLLECTED (expected $DASH_EXPECTED_NODES, log: $DASH_COLLECT_LOG)"
if [ "$DASH_COLLECT_EXIT" -ne 0 ] || [ "$DASH_COLLECTED" != "$DASH_EXPECTED_NODES" ]; then
  echo "SEL_FAIL: dashboard collection exited $DASH_COLLECT_EXIT / collected $DASH_COLLECTED != expected $DASH_EXPECTED_NODES" >&2
  exit 6
fi
( cd "$DASH_ROOT" && EVIDENCE_SPINE_CORE="$REPO_ROOT" EVIDENCE_SPINE_CORE_SHA="$CORE_SHA" \
  "$PYTHON" -m pytest -q --basetemp="$BASETEMP" backend/tests/test_profile_registry_hierarchy.py \
) > "$DASH_LOG" 2>&1
DASH_EXEC_EXIT=$?
DASH_PASSED="$(grep -E '^[0-9]+ passed' "$DASH_LOG" | tail -1)"
echo "DASH_RUN: $DASH_PASSED (exit $DASH_EXEC_EXIT, log: $DASH_LOG)"

# ── execution (R3-6.7): exit preserved independently ───────────────────────
"$PYTHON" -m pytest -q --basetemp="$BASETEMP" "${FILES[@]}" 2>&1 | tee "$RUN_LOG"
EXEC_EXIT=${PIPESTATUS[0]}

# ── C9 order-isolation variants (mandatory, exits preserved independently) ─
"$PYTHON" -m pytest -q -p no:cacheprovider --basetemp="$BASETEMP" \
  tests/test_phase_d_audit_wiring.py \
  "tests/tools/test_g4_retained_fleet_smoke_matrix.py::TestAuthorityGuards::test_U1_nonspawnable_lead" >/dev/null 2>&1
ORDER1_EXIT=$?
"$PYTHON" -m pytest -q -p no:cacheprovider --basetemp="$BASETEMP" \
  "tests/tools/test_g4_retained_fleet_smoke_matrix.py::TestAuthorityGuards::test_U1_nonspawnable_lead" \
  tests/test_phase_d_audit_wiring.py >/dev/null 2>&1
ORDER2_EXIT=$?

# ── R4-3.5/6: post-run status captured separately; EITHER changed → fail ──
POST_STATUS_CORE="$(git -C "$REPO_ROOT" status --porcelain --untracked-files=normal | sort)"
POST_STATUS_DASH="$(git -C "$DASH_ROOT" status --porcelain --untracked-files=normal | sort)"
STATUS_CLEAN=1
if [ "$PRE_STATUS_CORE" != "$POST_STATUS_CORE" ]; then
  echo "SEL_FAIL: CORE worktree status changed by the run" >&2
  diff <(echo "$PRE_STATUS_CORE") <(echo "$POST_STATUS_CORE") >&2
  STATUS_CLEAN=0
fi
if [ "$PRE_STATUS_DASH" != "$POST_STATUS_DASH" ]; then
  echo "SEL_FAIL: DASHBOARD status changed by the run" >&2
  diff <(echo "$PRE_STATUS_DASH") <(echo "$POST_STATUS_DASH") >&2
  STATUS_CLEAN=0
fi

echo "EXEC_EXIT=$EXEC_EXIT"
echo "ORDER1_EXIT=$ORDER1_EXIT"
echo "ORDER2_EXIT=$ORDER2_EXIT"
echo "DASH_COLLECT_EXIT=$DASH_COLLECT_EXIT"
echo "DASH_EXEC_EXIT=$DASH_EXEC_EXIT"
if [ "$STATUS_CLEAN" != "1" ]; then
  exit 4
fi
[ "$EXEC_EXIT" = "0" ] && [ "$ORDER1_EXIT" = "0" ] && [ "$ORDER2_EXIT" = "0" ] && [ "$DASH_COLLECT_EXIT" = "0" ] && [ "$DASH_EXEC_EXIT" = "0" ]
