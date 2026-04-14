# Unified Doctrine Promotion — Implementation Note

Date: 2026-03-19

## Canonical doctrine (new)
- `docs/ops/unified_operating_doctrine_v1.md`
  - Promoted as the top-level, authoritative doctrine for control-plane operation.
  - Anchored to source PDF: `/home/yeqiuqiu/.openclaw/media/inbound/Unified_Operating_Doctrine_for_a_Technical_AI_Operator_Syste---a1a3ed9d-8fec-43a2-82ce-8ed7da87af8a.pdf`.

## Subordinate doctrine/modules (retained, now explicitly under canonical)
- `docs/ops/phase1_control_plane_doctrine_v1.md`
- `docs/ops/model_routing_no_llm_matrix_v1.md`
- `docs/ops/blocker_burndown_control_loop_v1.md`
- `docs/ops/continuity_queue_state_model_v1.md`
- `docs/ops/subagent_slot_fill_protocol_v1.md`
- `docs/ops/swarm_operating_contract_runbook_v1.md`
- `docs/ops/templates/worker_slice_spec.template.json`
- `docs/ops/schemas/evidence_closeout.schema.json`

## Minimum supporting scaffolding added
- `docs/ops/verify_before_resume_gate_checklist_v1.md`
  - Explicit `allowed | caution | forbidden` resume gate with required evidence fields.
- `docs/ops/session_mode_blocker_vs_throughput_runbook_v1.md`
  - Deterministic entry/exit criteria and switch procedure for `BLOCKER_BURNDOWN` vs `THROUGHPUT`.

## Discoverability updates
- Updated `WORKING_PROTOCOL.md` doctrine pointer block to lead with the new canonical doctrine and include resume/mode gate runbooks.

## Deferred (intentional)
- Runtime enforcement wiring (scripts/scheduler/watchdog implementation changes).
- Historical doctrine normalization beyond minimal backlinking/pointer updates.
