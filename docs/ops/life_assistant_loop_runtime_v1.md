# Life Assistant Loop Runtime v1 (`XP-303`)

Date: 2026-03-29  
Status: active (canonical for `XP-303`)  
Owner: Architect  
Scope: Personal OS / life-assistant daily+weekly planning-review runtime (`XP-*`) in `/home/yeqiuqiu/clawd-architect`

---

## 1) Purpose

`XP-303` lands the bounded Personal OS runtime loop that was explicitly deferred by `XP-301` and `XP-302`.

This slice defines deterministic runtime behavior for:
1. daily and weekly planning-review intervention generation,
2. per-session/channel isolation of Personal OS planning loops,
3. low-noise heartbeat suppression with bounded action-card emission,
4. advisory-only outputs that remain inside the `XP-301` autonomy envelope.

This slice does **not** authorize autonomous external actions, PX2/PX3 execution, or cross-session context sharing.

---

## 2) Canonical inputs

- `state/continuity/latest/true_expanded_roadmap_queue_layer.json` (`XP-303` authoritative queue contract)
- `docs/ops/personal_os_scope_boundary_contract_v1.md` (`XP-301` refusal/escalation + approval envelope)
- `docs/ops/personal_context_graph_schema_pack_v1.md` (`XP-302` typed personal context objects)
- `docs/ops/low_noise_interaction_policy_v1.md`
- `docs/ops/model_routing_no_llm_matrix_v1.md`
- `docs/ops/session_topology_transport_contract_v1.md`
- `docs/ops/verify_before_resume_gate_checklist_v1.md`
- `docs/ops/invalid_output_retry_relaunch_contract_v1.md`
- `tests/fixtures/xp/personal_context_graph_objects_fixture_v1.json`
- `tests/fixtures/xp/life_assistant_loop_runtime_fixture_v1.json`

---

## 3) Execution-shape record

Per `docs/ops/model_routing_no_llm_matrix_v1.md`, `XP-303` executes with the following control record:

- `selected_route`: `NO_LLM`
- `reason`: deterministic queue/dependency gating + machine-validated JSON outputs are required
- `fallback_route`: `NO_LLM`
- `task_class`: `implementation`
- `risk_tier`: `high`
- `scope_shape`: `multi_surface_coupled`
- `worker_topology`: `single`
- `verification_class`: `validator_required`
- `fold_in_target`: `queue_continuity`

---

## 4) Runtime contract semantics

### 4.1 Loop windows
The runtime emits two bounded loop classes only:
- `daily_review`: near-term reminders, review prompts, and commitment guards.
- `weekly_review`: compact planning packets linking goals, routines, and learning objects.

### 4.2 Session isolation
1. Every intervention is owned by one resolved session key.
2. A session may only reference record ids declared visible inside its runtime fixture scope.
3. Cross-session probes must fail closed (`blocked`) and appear in negative-test artifacts.
4. Topic/session isolation follows the topology expectations in `docs/ops/session_topology_transport_contract_v1.md`.

### 4.3 Actionable-only emission
1. Runtime outputs must be advisory-only action cards or batched review packets.
2. Informational “all green” heartbeats are suppressed.
3. Every emitted intervention must include:
   - `summary`,
   - one or more concrete `action_items`,
   - referenced `record_ids`,
   - `required_approval_tier`,
   - `escalation_level`,
   - `advisory_only=true`.

### 4.4 Low-noise budget
1. `heartbeat_notification_budget = 0` for this runtime baseline.
2. `immediate_intervention_budget_per_session` is fixed and validated.
3. Extra actionable items may only appear in batched review packets; they do not bypass the immediate budget.
4. `E1..E4` refusal/escalation conditions bypass silent-success behavior and must remain explicit.

### 4.5 Boundary + approval alignment
1. Runtime may operate only on records whose governance tier is `PX0_INFO` or `PX1_ASSIST`.
2. Any referenced `PX2_HIGH_IMPACT` or `PX3_SAFETY_CRITICAL` record fails closed.
3. Runtime outputs do not execute external actions; they surface proposals and review prompts only.
4. Approval requirements are surfaced, not consumed or satisfied by the runtime itself.

### 4.6 Failure semantics
1. Missing dependency closure or queue-state mismatch blocks runtime generation.
2. Foreign-record leakage across sessions is a deterministic failure.
3. Invalid-output handling must fail closed and follow `docs/ops/invalid_output_retry_relaunch_contract_v1.md` when this runtime is supervised by broader execution loops.
4. Mutation work that re-enables or resumes the loop must record a verify-before-resume decision packet.

---

## 5) Canonical artifacts

### 5.1 Runtime generator
- `scripts/life_assistant_loop_runtime.py`

### 5.2 Runtime fixture
- `tests/fixtures/xp/life_assistant_loop_runtime_fixture_v1.json`

### 5.3 Generated operator/runtime evidence
- `state/continuity/latest/xp_303_life_assistant_loop_runtime_2026-03-29.json`
- `state/continuity/latest/xp_303_noise_budget_metrics_2026-03-29.json`
- `state/continuity/latest/xp_303_session_isolation_regression_tests_2026-03-29.json`
- `state/continuity/latest/xp_303_cross_session_contamination_negative_test_2026-03-29.json`
- `state/continuity/latest/xp_303_weekly_review_packet_2026-03-29.json`
- `state/continuity/latest/xp_303_verify_before_resume_gate_2026-03-29.json`
- `state/continuity/latest/xp_303_runtime_artifact_manifest_2026-03-29.json`
- `state/continuity/latest/xp_303_runtime_validation_2026-03-29.json`

---

## 6) Validation entrypoints

- `python scripts/life_assistant_loop_runtime.py --repo-root /home/yeqiuqiu/clawd-architect --stamp 2026-03-29 --json`
- `python -m json.tool tests/fixtures/xp/life_assistant_loop_runtime_fixture_v1.json`
- `python -m json.tool state/continuity/latest/xp_303_life_assistant_loop_runtime_2026-03-29.json`
- `python -m json.tool state/continuity/latest/xp_303_noise_budget_metrics_2026-03-29.json`
- `python -m json.tool state/continuity/latest/xp_303_session_isolation_regression_tests_2026-03-29.json`
- `python -m json.tool state/continuity/latest/xp_303_cross_session_contamination_negative_test_2026-03-29.json`
- `python -m json.tool state/continuity/latest/xp_303_weekly_review_packet_2026-03-29.json`
- `python -m json.tool state/continuity/latest/xp_303_verify_before_resume_gate_2026-03-29.json`
- `python -m json.tool state/continuity/latest/xp_303_runtime_artifact_manifest_2026-03-29.json`
- `python -m json.tool state/continuity/latest/xp_303_runtime_validation_2026-03-29.json`
- `python ops/openclaw/continuity/check_source_of_truth_map_regressions.py --repo-root /home/yeqiuqiu/clawd-architect --map-path /home/yeqiuqiu/clawd-architect/reports/openclaw_system_source_of_truth_map_2026-03-20.md --json`
- `pytest -q tests/test_xp_303_life_assistant_loop_runtime.py`

---

## 7) Closeout condition for `XP-303`

`XP-303` is complete only when:
1. this runtime contract is canonical,
2. deterministic runtime generation emits advisory-only daily/weekly interventions,
3. session-isolation regression tests pass,
4. noise-budget/heartbeat metrics prove silent-success suppression and bounded intervention counts,
5. cross-session contamination probes pass with fail-closed blocking,
6. verify-before-resume gate evidence exists,
7. queue slice `XP-303` transitions to `DONE` with bounded evidence refs,
8. no claim is made that Personal OS can autonomously execute external high-impact actions.
