#!/usr/bin/env bash
# Evidence Spine P3–P4 — committed controller selector manifest (R2).
#
# Byte-reproducible controller selector: the exact file list producing the
# documented collected-node count.  Run from the core candidate worktree:
#
#   HERMES_HOME=/tmp/selector-home scripts/evidence-spine-selector.sh
#
# Includes: the prior 34 controller files, all R2 test files, and the
# C9 order-isolation variants (as separate explicit invocations at the end).
set -u

SELECTOR_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC2034
REPO_ROOT="$(cd "$SELECTOR_DIR/.." && pwd)"
cd "$REPO_ROOT" || exit 1

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
  # ── R2 correction tests ──
  tests/hermes_cli/test_r2_corrections.py
)

# Deterministic collection: exact node count via --collect-only.
COLLECTED=$(/home/kensei/repos/KenseiAgent/.venv/bin/python3 -m pytest --collect-only -q "${FILES[@]}" 2>/dev/null | tail -1)
echo "COLLECTED: $COLLECTED"

# Execution.
/home/kensei/repos/KenseiAgent/.venv/bin/python3 -m pytest -q "${FILES[@]}"
EXEC_EXIT=$?

# ── C9 order-isolation variants (mandatory) ──
/home/kensei/repos/KenseiAgent/.venv/bin/python3 -m pytest -q -p no:cacheprovider \
  tests/test_phase_d_audit_wiring.py \
  tests/tools/test_g4_retained_fleet_smoke_matrix.py::TestAuthorityGuards::test_U1_nonspawnable_lead
ORDER1=$?
/home/kensei/repos/KenseiAgent/.venv/bin/python3 -m pytest -q -p no:cacheprovider \
  tests/tools/test_g4_retained_fleet_smoke_matrix.py::TestAuthorityGuards::test_U1_nonspawnable_lead \
  tests/test_phase_d_audit_wiring.py
ORDER2=$?

echo "EXEC_EXIT=$EXEC_EXIT"
echo "ORDER1_EXIT=$ORDER1"
echo "ORDER2_EXIT=$ORDER2"
[ "$EXEC_EXIT" = "0" ] && [ "$ORDER1" = "0" ] && [ "$ORDER2" = "0" ]