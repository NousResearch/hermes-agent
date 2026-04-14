# Human Explainability Panel for Domain Lanes v1 (`XU-503`)

Date: 2026-03-29  
Status: active (canonical for `XU-503`)  
Owner: Architect  
Scope: Frontend/operator UX explainability panel for governed domain-lane slices in `/home/yeqiuqiu/clawd-architect`

---

## 1) Purpose

`XU-503` lands the bounded explainability layer that turns queue truth, domain-registry projections, and closeout artifacts into a human-readable provenance panel.

This slice defines:
1. explainability panels for current domain-lane slices using canonical queue truth as the authority,
2. explicit drift surfacing when older derived projections disagree with queue truth,
3. dependency-trace and artifact-provenance rendering for operator review,
4. remediation paths that point to exact commands or artifact refs instead of narrative-only summaries.

This slice does **not** mutate prior lane runtimes automatically, does **not** override queue truth with stale projections, and does **not** authorize autonomous runtime mutation.

---

## 2) Canonical inputs

- `state/continuity/latest/true_expanded_roadmap_queue_layer.json`
- `docs/ops/xu_501_operator_ux_information_architecture_v2.md`
- `state/continuity/latest/xu_501_navigation_state_model_2026-03-28.json`
- `docs/ops/xu_502_lane_action_surface_runtime_v1.md`
- `state/continuity/latest/xb_402_domain_capability_registry_runtime_2026-03-29.json`
- `docs/ops/cockpit_action_card_design_v1.md`
- `docs/ops/human_first_observability_v1.md`
- `docs/ops/low_noise_interaction_policy_v1.md`
- `docs/ops/lane_boundary_contract_v1.md`
- `docs/ops/c3_activation_governance_contract_v1.md`
- `state/continuity/latest/xg_801_c3_activation_risk_matrix_2026-03-28.json`
- `state/continuity/latest/xg_801_c3_activation_owner_registry_2026-03-28.json`
- `tests/fixtures/xu/lane_explainability_panel_fixture_v1.json`
- Reusable promoted precedent from `XR-007`: `state/continuity/latest/evidence_trace_viewer_latest.json`

`XB-402` is a required upstream dependency because `XU-503` explains domain-lane state using the capability-registry projection. However, **queue truth remains authoritative** whenever the registry snapshot is older or disagreeing.

---

## 3) Execution-shape record

Per `docs/ops/model_routing_no_llm_matrix_v1.md`, `XU-503` executes with the following control record:

- `selected_route`: `NO_LLM`
- `reason`: queue/projection/artifact provenance must remain deterministic and machine-verifiable
- `fallback_route`: `NO_LLM`
- `task_class`: `implementation`
- `risk_tier`: `medium`
- `scope_shape`: `multi_surface_coupled`
- `worker_topology`: `single`
- `verification_class`: `validator_required`
- `fold_in_target`: `queue_continuity`

---

## 4) Explainability runtime contract

### 4.1 Panel hierarchy
`XU-503` extends the `XU-501` IA layers with a dedicated explainability overlay:

1. **Program Overview**
   - required queue summary and first-launch recommendation.
2. **Lane Projection Summary**
   - one row per relevant lane comparing queue counts vs registry counts.
3. **Explainability Panel**
   - focused per-slice provenance card with direct dependency trace.
4. **Artifact Provenance Strip**
   - bounded list of closeout/evidence refs reachable from the traced dependency chain.

### 4.2 Authority and drift semantics
1. Queue truth is the state authority.
2. Derived registry projections are diagnostic only.
3. If queue truth and registry projection disagree, the panel must render explicit drift (`projection_drift`) and cite both sources.
4. Stale projection drift is not a blocker by itself once queue truth shows the slice `READY` or `DONE`.

### 4.3 Explainability payload requirements
Every rendered panel must include:
- queue state,
- derived operator state,
- owners and risk posture,
- dependency states from queue truth,
- dependency trace rows,
- artifact provenance refs,
- blocker causes (including projection drift if present),
- immediate remediation paths,
- telemetry footer pointers.

### 4.4 Fail-closed conditions
The explainability panel fails closed when any condition is true:
1. queue slice payload is missing or malformed,
2. owner tuple is missing,
3. risk assignment is missing,
4. queue state cannot be mapped by the `XU-501` state model.

Fail-closed rendering must still provide a remediation hint, but must not claim a valid explainability panel was rendered.

### 4.5 Human-first and low-noise rendering
1. Markdown must follow the unified action-card structure.
2. Drift/blocker causes must be listed with exact contract-style ids.
3. Remediation must be a full copy-pasteable command or explicit artifact path.
4. Passive `DONE` refreshes remain silent; this slice emits artifacts only.

---

## 5) Canonical artifacts

### 5.1 Runtime generator
- `scripts/lane_explainability_panel_runtime.py`

### 5.2 Fixture + test
- `tests/fixtures/xu/lane_explainability_panel_fixture_v1.json`
- `tests/test_xu_503_lane_explainability_panel_runtime.py`

### 5.3 Generated explainability evidence
- `state/continuity/latest/xu_503_lane_explainability_panel_runtime_2026-03-29.json`
- `state/continuity/latest/xu_503_operator_explainability_views_2026-03-29.json`
- `state/continuity/latest/xu_503_provenance_parity_tests_2026-03-29.json`
- `state/continuity/latest/xu_503_operator_usability_checks_2026-03-29.json`
- `state/continuity/latest/xu_503_verify_before_resume_gate_2026-03-29.json`
- `state/continuity/latest/xu_503_runtime_artifact_manifest_2026-03-29.json`
- `state/continuity/latest/xu_503_runtime_validation_2026-03-29.json`
- `state/continuity/latest/xu_503_source_of_truth_map_guard_2026-03-29.json`

---

## 6) Validation entrypoints

- `python scripts/lane_explainability_panel_runtime.py --repo-root /home/yeqiuqiu/clawd-architect --stamp 2026-03-29 --json`
- `python -m py_compile scripts/lane_explainability_panel_runtime.py tests/test_xu_503_lane_explainability_panel_runtime.py`
- `python -m json.tool tests/fixtures/xu/lane_explainability_panel_fixture_v1.json`
- `python -m json.tool state/continuity/latest/xu_503_lane_explainability_panel_runtime_2026-03-29.json`
- `python -m json.tool state/continuity/latest/xu_503_operator_explainability_views_2026-03-29.json`
- `python -m json.tool state/continuity/latest/xu_503_provenance_parity_tests_2026-03-29.json`
- `python -m json.tool state/continuity/latest/xu_503_operator_usability_checks_2026-03-29.json`
- `python -m json.tool state/continuity/latest/xu_503_verify_before_resume_gate_2026-03-29.json`
- `python -m json.tool state/continuity/latest/xu_503_runtime_artifact_manifest_2026-03-29.json`
- `python -m json.tool state/continuity/latest/xu_503_runtime_validation_2026-03-29.json`
- `python -m json.tool state/continuity/latest/xu_503_source_of_truth_map_guard_2026-03-29.json`
- `python ops/openclaw/continuity/check_source_of_truth_map_regressions.py --repo-root /home/yeqiuqiu/clawd-architect --map-path /home/yeqiuqiu/clawd-architect/reports/openclaw_system_source_of_truth_map_2026-03-20.md --json`
- `pytest -q tests/test_xu_503_lane_explainability_panel_runtime.py`

---

## 7) Closeout condition for `XU-503`

`XU-503` is complete only when:
1. this explainability contract is canonical,
2. deterministic explainability panels render queue-truth state, drift causes, and artifact provenance,
3. provenance parity tests pass for ready/done/fail-closed cases,
4. operator usability checks prove markdown/remediation ergonomics,
5. verify-before-resume evidence exists,
6. source-of-truth map registers the XU-503 implementation/tests/operator surfaces,
7. queue slice `XU-503` transitions to `DONE` only after validation passes.
