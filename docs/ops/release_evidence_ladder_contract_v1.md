# Release Evidence Ladder Contract v1

Date: 2026-03-21  
Status: active (Wave 6 release substrate formalization)

## Purpose
Provide one canonical, fail-closed release-governance ladder for deploy-vs-activate decisions.

This contract unifies stage evidence, rollback-proof recency, and compatibility lifecycle governance before canary/progressive/broad activation.

## Artifacts
- Contract: `docs/ops/release_evidence_ladder_contract_v1.md`
- Bundle schema/template:
  - `docs/ops/schemas/release_evidence_bundle.schema.json`
  - `docs/ops/templates/release_evidence_bundle.template.json`
- Runtime gate: `scripts/release_evidence_ladder_gate.py`
- Decision log: `state/continuity/release_governance/release_evidence_ladder_decisions.jsonl`

## Canonical stage ladder order
1. `local_determinism`
2. `presubmit`
3. `integration_replay`
4. `shadow`
5. `canary`
6. `progressive`
7. `broad_activation`

## Activation mode minimum stage coverage
- `shadow`: stages 1-4 must pass.
- `canary`: stages 1-5 must pass.
- `progressive`: stages 1-6 must pass.
- `broad_activation`: stages 1-7 must pass.

## Hard gates
1. **Stage order + coverage** must respect canonical ladder.
2. **Stage status** for required stages must be `pass`.
3. **Rollback proof recency** (`drilled_at` <= `max_age_hours`) must pass before canary/progressive/broad activation.
4. **Compatibility lifecycle evidence** must include a register reference and, when exceptions remain, at least one removal RFC reference.
5. **Evidence refs** must resolve to in-repo artifacts.
6. **Design gate stack (`XD-103`, conditional)**: when `lane_context` targets DesignOps (`XD` / `lane.designops*`), bundles must provide `design_gate_stack` (`G1_SCHEMA..G6_ALIGNMENT`) with all gates `pass` and resolved evidence refs.
7. **Health requirement coupling (`EX-05`)**: bundle-level `health_requirement` defines strict required-lane and minimum-layer expectations against `state/continuity/latest/layered_health_snapshot.json`; restore freshness (`SLO-4_RESTORE_DRILL_FRESHNESS`) is required by default.

## Commands
- Direct:
  - `python3 scripts/release_evidence_ladder_gate.py --bundle <bundle.json> --json`
- Continuity dispatcher:
  - `bash ops/openclaw/continuity.sh release-evidence-gate --bundle <bundle.json> --json`

## Notes
- This contract governs release decisions; it does not replace model qualification rollout policy contracts.
- Automation over this ladder can grow later, but gate semantics stay deterministic and fail-closed.

## XR-007 promoted asset checklist (C2 canonical promotion pack)

Archive/runtime asset promoted to canonical C2 checklist:

1. **Live observability/error-budget rollback trigger coupling**
   - Trigger artifact (latest): `state/continuity/latest/release_error_budget_rollback_trigger_latest.json`
   - Trigger history: `state/continuity/release_governance/release_error_budget_rollback_trigger_history.jsonl`
   - Slice evidence: `reports/c2_runtime_coupling_to_live_observability_error_budgets_slice_2026-03-28.md`
   - Required verification refs:
     - `tests/test_release_evidence_ladder_gate.py`
     - `tests/test_cockpit_summary_failclose.py`

Promotion rule (fail-closed): C2 release progression is not canonicalized unless A6-fail gate paths emit an active rollback trigger with deterministic rollback command projection.
