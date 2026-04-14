# Release Migration Playbook v1

Date: 2026-03-21  
Status: active (Wave 6 continuity-safe upgrade mechanics)

## Purpose
Provide a minimal canonical playbook for continuity-safe release progression and rollback readiness.

## Flow
1. Build release evidence bundle (`release_evidence_bundle.schema.json`).
2. Run `release-evidence-gate` and record decision.
3. If PASS, activate only up to declared `activation_mode`.
4. Observe health/replay monitors for bounded window.
5. If widening activation, regenerate evidence bundle with updated stages.
6. If gate blocks or monitor thresholds breach, execute rollback using latest rollback proof artifacts.

## Required evidence references
- replay evidence
- shadow/canary/progressive evidence
- rollback drill proof (recent)
- compatibility register + removal RFC references

## Mutations and authority
- Activation widening is a mutation and must remain action-token gated.
- Rollback actions are emergency-safe but still require logged operator reason and evidence references.

## Operator checklist
- [ ] bundle schema valid
- [ ] required stages present + pass
- [ ] rollback proof within allowed recency window
- [ ] compatibility register + RFC refs attached
- [ ] post-activation monitor window observed
