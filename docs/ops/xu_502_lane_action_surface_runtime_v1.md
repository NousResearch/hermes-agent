# Lane Action Surface Runtime v1 (`XU-502`)

Date: 2026-03-29  
Status: active (canonical for `XU-502`)  
Owner: Architect  
Scope: Frontend/operator UX action-surface runtime for governed expanded lanes in `/home/yeqiuqiu/clawd-architect`

---

## 1) Purpose

`XU-502` lands the bounded runtime layer that turns the `XU-501` IA/state model into deterministic operator action surfaces.

This slice defines:
1. composable slice workspaces that map queue truth into standard action widgets,
2. dispatch bindings from widgets to lane-contract packet types (`signal`, `ticket`, `deep_review`),
3. permission/audit semantics for operator-triggered actions,
4. human-first, low-noise renderings for `READY`, `DEPENDENCY_BLOCKED`, `DONE`, and fail-closed review states.

This slice does **not** implement explainability/provenance trace rendering (`XU-503`) and does **not** authorize autonomous runtime mutation outside explicit approval-gated ticket previews.

---

## 2) Canonical inputs

- `state/continuity/latest/true_expanded_roadmap_queue_layer.json`
- `docs/ops/xu_501_operator_ux_information_architecture_v2.md`
- `state/continuity/latest/xu_501_navigation_state_model_2026-03-28.json`
- `docs/ops/cockpit_action_card_design_v1.md`
- `docs/ops/human_first_observability_v1.md`
- `docs/ops/low_noise_interaction_policy_v1.md`
- `docs/ops/lane_boundary_contract_v1.md`
- `docs/ops/c3_activation_governance_contract_v1.md`
- `state/continuity/latest/xg_801_c3_activation_risk_matrix_2026-03-28.json`
- `state/continuity/latest/xg_801_c3_activation_owner_registry_2026-03-28.json`
- `tests/fixtures/xu/lane_action_surface_runtime_fixture_v1.json`

`XB-402` is a required upstream dependency because the action runtime assumes deterministic lane inventory and health semantics exist. However, **queue truth remains the actionability authority** if a previously-generated registry snapshot is older than the queue layer.

---

## 3) Execution-shape record

Per `docs/ops/model_routing_no_llm_matrix_v1.md`, `XU-502` executes with the following control record:

- `selected_route`: `NO_LLM`
- `reason`: deterministic queue/risk/owner inputs and machine-validated operator artifacts are required
- `fallback_route`: `NO_LLM`
- `task_class`: `implementation`
- `risk_tier`: `medium`
- `scope_shape`: `multi_surface_coupled`
- `worker_topology`: `single`
- `verification_class`: `validator_required`
- `fold_in_target`: `queue_continuity`

---

## 4) Runtime contract semantics

### 4.1 Action-surface hierarchy
`XU-502` preserves the `XU-501` IA layers and adds runtime actions only where governance allows:

1. **Program Overview**
   - queue posture only;
   - no mutation widgets.
2. **Lane Board**
   - lane summary + owner/risk overlays;
   - may expose a primary action class label, not direct mutation.
3. **Slice Workspace**
   - canonical action-widget surface;
   - shows dependencies, closeout condition, evidence refs, permission posture, and remediation commands.
4. **Evidence / Walkthrough View**
   - read-only evidence drill-down;
   - used for `DONE` review and blocked-lane diagnostics.

### 4.2 Composable widget taxonomy
Standard widget ids are:
- `prepare_closeout_ticket`
- `inspect_evidence_bundle`
- `review_lane_contracts`
- `inspect_dependency_chain`
- `request_dependency_unblock_review`
- `review_closeout_bundle`
- `replay_validation_bundle`

Widget rules:
1. Widgets are chosen strictly from normalized operator state.
2. `ACTIONABLE_NOW` may emit exactly one approval-gated `ticket` widget plus read-only review widgets.
3. `WAITING_ON_DEPENDENCY` may emit only read-only `signal`/`deep_review` widgets.
4. `CLOSED_VERIFIED` is read-only and evidence-first.
5. `FAIL_CLOSED_REVIEW` suppresses the action runtime entirely and surfaces remediation hints only.

### 4.3 Packet binding to lane contracts
Every actionable widget must bind to one lane-boundary packet type:
- `signal` for dependency and status inspection,
- `ticket` for bounded execution proposals,
- `deep_review` for evidence or governance review.

Generated packet previews must include the packet-type required fields from `docs/ops/lane_boundary_contract_v1.md` plus contamination-guard metadata.

### 4.4 Permission and approval posture
1. No widget may auto-execute external mutations.
2. `ticket` widgets must declare `promotion_gate=human_required` and `requires_operator_confirmation=true`.
3. `signal` and `deep_review` widgets are read-only operator aids.
4. Missing owner tuple, missing risk assignment, unknown queue state, or malformed slice payload maps to `FAIL_CLOSED_REVIEW` and zero runtime widgets.
5. `RG3_CRITICAL` blocked lanes may be rendered for diagnostics, but must not expose mutation-capable widgets.

### 4.5 Human-first and low-noise rendering
1. Every rendered workspace must include:
   - headline status,
   - failing constraints,
   - immediate action/remediation,
   - telemetry footer.
2. Remediation text must include a full command or artifact pointer.
3. Inline-button payloads must remain compact and deterministic.
4. Passive `DONE` refreshes remain silent; this slice emits artifacts, not unsolicited notifications.

### 4.6 Failure semantics
The runtime fails closed when any condition is true:
1. `XU-502` queue state is neither `READY` nor `DONE`.
2. `XU-502` dependencies are not all `DONE`.
3. Risk/owner overlays required for the rendered workspace are missing.
4. A generated `ticket` preview lacks `allowed_write_paths[]` or `verification_commands[]`.
5. A blocked or critical workspace exposes a mutation-capable widget.

---

## 5) Canonical artifacts

### 5.1 Runtime generator
- `scripts/lane_action_surface_runtime.py`

### 5.2 Fixture + test
- `tests/fixtures/xu/lane_action_surface_runtime_fixture_v1.json`
- `tests/test_xu_502_lane_action_surface_runtime.py`

### 5.3 Generated operator/runtime evidence
- `state/continuity/latest/xu_502_lane_action_surface_runtime_2026-03-29.json`
- `state/continuity/latest/xu_502_operator_workspace_views_2026-03-29.json`
- `state/continuity/latest/xu_502_action_simulation_tests_2026-03-29.json`
- `state/continuity/latest/xu_502_permission_audit_trace_2026-03-29.json`
- `state/continuity/latest/xu_502_verify_before_resume_gate_2026-03-29.json`
- `state/continuity/latest/xu_502_runtime_artifact_manifest_2026-03-29.json`
- `state/continuity/latest/xu_502_runtime_validation_2026-03-29.json`
- `state/continuity/latest/xu_502_source_of_truth_map_guard_2026-03-29.json`

---

## 6) Validation entrypoints

- `python scripts/lane_action_surface_runtime.py --repo-root /home/yeqiuqiu/clawd-architect --stamp 2026-03-29 --json`
- `python -m py_compile scripts/lane_action_surface_runtime.py tests/test_xu_502_lane_action_surface_runtime.py`
- `python -m json.tool tests/fixtures/xu/lane_action_surface_runtime_fixture_v1.json`
- `python -m json.tool state/continuity/latest/xu_502_lane_action_surface_runtime_2026-03-29.json`
- `python -m json.tool state/continuity/latest/xu_502_operator_workspace_views_2026-03-29.json`
- `python -m json.tool state/continuity/latest/xu_502_action_simulation_tests_2026-03-29.json`
- `python -m json.tool state/continuity/latest/xu_502_permission_audit_trace_2026-03-29.json`
- `python -m json.tool state/continuity/latest/xu_502_verify_before_resume_gate_2026-03-29.json`
- `python -m json.tool state/continuity/latest/xu_502_runtime_artifact_manifest_2026-03-29.json`
- `python -m json.tool state/continuity/latest/xu_502_runtime_validation_2026-03-29.json`
- `python -m json.tool state/continuity/latest/xu_502_source_of_truth_map_guard_2026-03-29.json`
- `python ops/openclaw/continuity/check_source_of_truth_map_regressions.py --repo-root /home/yeqiuqiu/clawd-architect --map-path /home/yeqiuqiu/clawd-architect/reports/openclaw_system_source_of_truth_map_2026-03-20.md --json`
- `pytest -q tests/test_xu_502_lane_action_surface_runtime.py`

---

## 7) Closeout condition for `XU-502`

`XU-502` is complete only when:
1. this runtime contract is canonical,
2. deterministic slice workspaces render standardized action widgets from queue truth,
3. action simulation tests prove packet bindings are correct for ready/blocked/done/fail-closed cases,
4. permission audit traces prove no auto-execute mutation path exists,
5. verify-before-resume evidence exists,
6. source-of-truth map registers the runtime implementation/tests/operator surfaces,
7. queue slice `XU-502` transitions to `DONE` only after validation passes,
8. no claim is made that `XU-503` explainability runtime is already complete.
