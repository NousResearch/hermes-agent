# Health ingestion/support runtime v1 (`XH-703`)

Date: 2026-03-29  
Status: active (canonical for `XH-703`)  
Owner: Architect  
Scope: deterministic health ingestion/support MVP runtime inside `/home/yeqiuqiu/clawd-architect`

---

## 1) Purpose

`XH-703` lands the bounded health-runtime layer that turns the `XH-701` safety boundary and `XH-702` typed schema pack into deterministic ingest/support artifacts.

This slice implements:
1. manual + wearable + lab intake under schema/privacy gates,
2. non-diagnostic support-card generation from accepted health records,
3. fail-closed rejection of policy-breaching records,
4. deterministic `XG-803` incident emission for safety/policy trips.

This slice remains **supportive and advisory-only**.
It does **not** diagnose, prescribe treatment, replace licensed clinicians, or authorize autonomous external health actions.

---

## 2) Canonical inputs

- `state/continuity/latest/true_expanded_roadmap_queue_layer.json` (`XH-703` authoritative queue state)
- `docs/ops/health_lane_boundary_safety_contract_v1.md` (`XH-701` non-diagnostic boundary + escalation contract)
- `docs/ops/health_typed_record_schema_pack_v1.md` (`XH-702` typed health-record schema pack)
- `docs/ops/schemas/health_typed_record.schema.json`
- `docs/ops/xg_803_domain_failclose_incident_contract_v1.md`
- `docs/ops/verify_before_resume_gate_checklist_v1.md`
- `docs/ops/invalid_output_retry_relaunch_contract_v1.md`
- `state/continuity/latest/xg_801_c3_activation_risk_matrix_2026-03-28.json`
- `state/continuity/latest/xg_801_c3_activation_owner_registry_2026-03-28.json`
- `tests/fixtures/xh/health_runtime_fixture_v1.json`
- bounded fixture inputs referenced by that runtime fixture

---

## 3) Runtime artifacts (normative)

### 3.1 Runtime snapshot
- `state/continuity/latest/xh_703_health_runtime_2026-03-29.json`

Must publish:
- authoritative queue precondition,
- dependency + governance posture,
- advisory-only boundary and escalation ceiling,
- accepted/rejected intake summary,
- refs to support and incident operator surfaces.

### 3.2 Ingest audit
- `state/continuity/latest/xh_703_ingest_audit_2026-03-29.json`

Must prove:
- manual / wearable / lab sources were evaluated deterministically,
- schema-valid records preserving `advisory_only=true` were accepted,
- privacy/policy-breaching records were rejected fail-closed,
- source-level observed results match fixture expectation.

### 3.3 Support workspace
- `state/continuity/latest/xh_703_support_workspace_2026-03-29.json`

Must provide:
- non-diagnostic support cards,
- clinician question-prep outputs for lab records,
- tracking / adherence support for symptom and protocol records,
- explicit disclaimers and escalation routing.

### 3.4 Incident fail-close surfaces
- `state/continuity/latest/xh_703_failclose_incident_packet_2026-03-29.json`
- `state/continuity/latest/xh_703_failclose_incident_gate_decision_2026-03-29.json`
- `state/continuity/latest/xh_703_incident_to_lesson_handoff_packet_2026-03-29.json`
- `state/continuity/latest/xh_703_knowledge_queue_ingestion_trace_2026-03-29.json`
- `state/continuity/latest/xh_703_operator_remediation_plan_2026-03-29.json`
- `state/continuity/latest/xh_703_remediation_closure_verification_2026-03-29.json`

Must prove:
- fail-close packet emission for a health policy/safety trip,
- `XG-803` gate PASS on the emitted incident packet,
- explicit owner/timebound remediation and incident-to-learning handoff.

### 3.5 Validation and manifest surfaces
- `state/continuity/latest/xh_703_end_to_end_ingestion_tests_2026-03-29.json`
- `state/continuity/latest/xh_703_verify_before_resume_gate_2026-03-29.json`
- `state/continuity/latest/xh_703_runtime_artifact_manifest_2026-03-29.json`
- `state/continuity/latest/xh_703_runtime_validation_2026-03-29.json`
- `state/continuity/latest/xh_703_source_of_truth_map_guard_2026-03-29.json`

---

## 4) Mandatory runtime controls

1. **Queue-truth gate**
   - `XH-703` may execute only when queue truth is `READY` or already `DONE`.
   - Required queue dependencies: `XH-702`, `XB-402`, `XU-502`, `XG-803` must be `DONE`.

2. **Advisory-only health boundary**
   - Accepted records must preserve `governance.route_class=advisory` and `advisory_only=true`.
   - Runtime outputs must state that they are non-diagnostic and do not replace professional care.

3. **Manual + wearable + lab modality coverage**
   - The MVP runtime must ingest at least one accepted source from each modality class: `manual`, `wearable`, `lab`.
   - Accepted support surfaces must remain deterministic and evidence-backed.

4. **Privacy / policy fail-close**
   - PHI-bearing records that violate confidentiality constraints must be rejected.
   - Rejected policy-breach sources must not appear in support cards.
   - At least one bounded policy/safety trip must emit a valid `XG-803` fail-close incident packet.

5. **Support-card constraints**
   - Every support card must include:
     - `summary`,
     - one or more `action_items`,
     - `record_ids`,
     - `escalation_code`,
     - `advisory_only=true`,
     - a non-diagnostic disclaimer.
   - Lab support cards are question-prep / referral-oriented only.

6. **Invalid-output supervision compatibility**
   - If this runtime is embedded in broader execution loops, invalid/junk downstream outputs must fail closed and follow `docs/ops/invalid_output_retry_relaunch_contract_v1.md`.

---

## 5) Validation entrypoints

- `python scripts/health_ingestion_support_runtime.py --repo-root /home/yeqiuqiu/clawd-architect --stamp 2026-03-29 --json`
- `python -m py_compile scripts/health_ingestion_support_runtime.py tests/test_xh_703_health_ingestion_support_runtime.py`
- `python tests/test_xh_703_health_ingestion_support_runtime.py`
- `python scripts/domain_failclose_incident_gate.py --repo-root /home/yeqiuqiu/clawd-architect --packet state/continuity/latest/xh_703_failclose_incident_packet_2026-03-29.json --json`
- `python -m json.tool tests/fixtures/xh/health_runtime_fixture_v1.json`
- `python -m json.tool state/continuity/latest/xh_703_health_runtime_2026-03-29.json`
- `python -m json.tool state/continuity/latest/xh_703_ingest_audit_2026-03-29.json`
- `python -m json.tool state/continuity/latest/xh_703_support_workspace_2026-03-29.json`
- `python -m json.tool state/continuity/latest/xh_703_failclose_incident_packet_2026-03-29.json`
- `python -m json.tool state/continuity/latest/xh_703_failclose_incident_gate_decision_2026-03-29.json`
- `python -m json.tool state/continuity/latest/xh_703_end_to_end_ingestion_tests_2026-03-29.json`
- `python -m json.tool state/continuity/latest/xh_703_verify_before_resume_gate_2026-03-29.json`
- `python -m json.tool state/continuity/latest/xh_703_runtime_artifact_manifest_2026-03-29.json`
- `python -m json.tool state/continuity/latest/xh_703_runtime_validation_2026-03-29.json`
- `python ops/openclaw/continuity/check_source_of_truth_map_regressions.py --repo-root /home/yeqiuqiu/clawd-architect --map-path /home/yeqiuqiu/clawd-architect/reports/openclaw_system_source_of_truth_map_2026-03-20.md --json`

---

## 6) Closeout condition for `XH-703`

`XH-703` is complete only when:
1. this runtime contract is canonical,
2. manual + wearable + lab ingest is published via deterministic runtime surfaces,
3. support outputs remain advisory-only and non-diagnostic,
4. policy/safety breaches are rejected fail-closed and emit valid incident evidence,
5. verify-before-resume and end-to-end validation artifacts pass,
6. source-of-truth map is updated to register `XH-703` runtime assets,
7. queue layer transitions `XH-703` from `READY` to `DONE` with bounded evidence refs,
8. no diagnosis/treatment authority is introduced.
