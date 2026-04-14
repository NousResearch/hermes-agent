# Error-Budget Rollout Policy v1

Date: 2026-03-21
Status: active (Wave 8 C2 DevEx Substrate)
Parent doctrine: `docs/ops/unified_operating_doctrine_v1.md`

## 0) Purpose
To safely advance structural upgrades (Waves, architectural migrations, pool changes) through the Release Evidence Ladder. A release stage MUST NOT advance unless both A6 surfaces are fully green: system SLO snapshot status is `pass`, and layered health status is `pass`.

## 1) Evaluation Criteria
Before `release_evidence_ladder_gate.py` advances a phase (e.g., Phase B `Runtime Safety Validation` to Phase C `Broad Enrollment`), it MUST read `state/continuity/latest/slo_snapshot.json` and `layered_health_snapshot.json`.

1. **SLO Constraint:** If `.status` != `"pass"`, the rollout gate evaluation is hard-halted.
2. **Health Constraint:** If `.status` != `"pass"`, the rollout gate evaluation is hard-halted.
3. **Required lane constraint (EX-05 bounded MVP):** `A1_CONTROL_PLANE`, `A2_RUNTIME_CONTINUITY`, `A3_MODEL_ROUTING`, `A6_OPS_OBSERVABILITY`, `C1_OPERATOR_SURFACE`, and `C2_RELEASE_SUBSTRATE` must be present and `pass` at or above the required minimum layer (`truthful` by default).
4. **Restore evidence constraint:** `SLO-4_RESTORE_DRILL_FRESHNESS` must be explicitly present and `pass` in `slo_snapshot.json`.

The release bundle may override the lane policy through `health_requirement`, but default mode is `strict` and fail-closed.

## 2) Replay/Rollback Mandate
If an error budget is breached *during* a Release Ladder sequence, the new configuration or architectural update is presumed unsafe. The DevEx Substrate is required to immediately halt the sequence, and the operator must be presented with the `verify_then_resume.sh --run-rollback` remediation path via the Cockpit summary.
