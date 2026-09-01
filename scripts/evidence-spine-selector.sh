#!/usr/bin/env bash
# Evidence Spine P3–P4 — committed controller selector manifest (R3).
#
# R3-6: the collection step is now a REAL gate, not a decoration:
#   * `set -u` plus explicit producer exit capture (no masked pipes);
#   * collect-only output is saved to a per-invocation evidence file and
#     the collection exit code is propagated, not swallowed by `| tail`;
#   * the exact expected collected-node count for the Round-3 SHA is
#     pinned and asserted;
#   * a unique disposable HERMES_HOME and --basetemp are created inside
#     the script per invocation;
#   * core import provenance (this worktree) and the dashboard SHA/path
#     are bound and validated BEFORE any test runs;
#   * execution, order-1 and order-2 exit codes are preserved
#     independently and all three are reported;
#   * the post-run worktree status must match the pre-run classified
#     status (tests must not dirty the candidate).
#
# Run from the core candidate worktree:
#   scripts/evidence-spine-selector.sh
#
# Exit codes:
#   0 = collection enforced, execution passed, both order tests passed,
#       and worktree status is unchanged
#   non-zero otherwise (the failing stage is named on stderr)
set -u

SELECTOR_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SELECTOR_DIR/.." && pwd)"
cd "$REPO_ROOT" || { echo "SEL_FAIL: cannot cd REPO_ROOT" >&2; exit 2; }

PYTHON="${PYTHON:-/home/kensei/repos/KenseiAgent/.venv/bin/python3}"
EVIDENCE_DIR="${EVIDENCE_DIR:-/tmp/r3-evidence}"
mkdir -p "$EVIDENCE_DIR"
STAMP="$(date +%Y%m%dT%H%M%SZ)-$$"
EVIDENCE_FILE="$EVIDENCE_DIR/selector-collect-$STAMP.txt"
RUN_LOG="$EVIDENCE_DIR/selector-run-$STAMP.txt"

# ── disposable homes / basetemps (per invocation) ─────────────────────────
DISP_HOME="$(mktemp -d /tmp/r3-sel-home.XXXXXX)"
BASETEMP="$(mktemp -d /tmp/r3-sel-basetemp.XXXXXX)"
trap 'rm -rf "$DISP_HOME" "$BASETEMP"' EXIT
export HERMES_HOME="$DISP_HOME"

# ── provenance: core import must resolve to THIS checkout ──────────────────
CORE_SHA="$(git rev-parse HEAD)"
CORE_DIR="$(git rev-parse --show-toplevel)"
if [ "$CORE_DIR" != "$REPO_ROOT" ]; then
  echo "SEL_FAIL: git toplevel ($CORE_DIR) != selector repo root ($REPO_ROOT)" >&2
  exit 2
fi
# The test process must import hermes_cli from THIS worktree, not the
# canonical checkout.  Prove it with the same interpreter the tests use.
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
# R3-5: the dashboard checkout is supplied EXPLICITLY (no baked-in absolute
# path — the committed selector set must contain zero absolute worktree
# prefixes).  EVIDENCE_SPINE_DASHBOARD is required; the expected SHA is
# bound by EVIDENCE_SPINE_DASHBOARD_SHA (default = frozen Round-2 dashboard).
if [ -z "${EVIDENCE_SPINE_DASHBOARD:-}" ]; then
  echo "SEL_FAIL: EVIDENCE_SPINE_DASHBOARD is required (bind the dashboard checkout explicitly)" >&2
  exit 2
fi
DASH_ROOT="$EVIDENCE_SPINE_DASHBOARD"
DASH_SHA_EXPECTED="${EVIDENCE_SPINE_DASHBOARD_SHA:-d6b8041526335c69fd9bf3c75b8399ab805de219}"
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

# ── pre-run classified worktree status (must be reproduced post-run) ──────
PRE_STATUS="$(git status --porcelain --untracked-files=normal | sort)"

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
  # ── R2/R3 correction tests ──
  tests/hermes_cli/test_r2_corrections.py
)

# ── collection gate (R3-6.2/3/4): real exit, evidence file, exact count ────
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
EXPECTED_NODES=477
echo "COLLECTED: $COLLECTED_COUNT (expected $EXPECTED_NODES, log: $EVIDENCE_FILE)"
if [ "$COLLECTED_COUNT" != "$EXPECTED_NODES" ]; then
  echo "SEL_FAIL: collected $COLLECTED_COUNT != expected $EXPECTED_NODES" >&2
  exit 3
fi

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

# ── post-run classified worktree status must match pre-run ──────────────────
POST_STATUS="$(git status --porcelain --untracked-files=normal | sort)"
STATUS_CLEAN=1
if [ "$PRE_STATUS" != "$POST_STATUS" ]; then
  echo "SEL_FAIL: worktree status changed by the run" >&2
  diff <(echo "$PRE_STATUS") <(echo "$POST_STATUS") >&2
  STATUS_CLEAN=0
fi

echo "EXEC_EXIT=$EXEC_EXIT"
echo "ORDER1_EXIT=$ORDER1_EXIT"
echo "ORDER2_EXIT=$ORDER2_EXIT"
if [ "$STATUS_CLEAN" != "1" ]; then
  exit 4
fi
[ "$EXEC_EXIT" = "0" ] && [ "$ORDER1_EXIT" = "0" ] && [ "$ORDER2_EXIT" = "0" ]
