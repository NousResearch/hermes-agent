# XG-803 Domain Fail-Close Incident Contract v1

Date: 2026-03-28  
Status: active (canonical XG-803 incident fail-close + learning-loop contract)  
Owner: Architect  
Scope: Cross-domain incident artifacts, remediation closure, and incident-to-learning handoff for XG-governed downstream lanes

---

## 1) Purpose

`XG-803` closes the incident-governance gap left after `XG-801` and `XG-802`.

It defines mandatory contract behavior for domain-lane incidents:
1. deterministic fail-close incident packet emission,
2. explicit owner + timebound remediation loop,
3. verified remediation closure artifacts,
4. incident-to-learning handoff into knowledge ingestion/promotion queues.

This slice is contract + schema/template + gate/runtime + evidence closeout.

---

## 2) Canonical dependencies

- `docs/ops/c3_activation_governance_contract_v1.md` (`XG-801` foundation)
- `docs/ops/xg_802_domain_release_evidence_ladder_extension_contract_v1.md`
- `docs/ops/release_evidence_ladder_contract_v1.md`
- `docs/ops/controlled_cross_lane_bridge_contract_v1.md`
- `docs/ops/lane_boundary_contract_v1.md`
- `docs/ops/knowledge_review_approval_promotion_queue_v1.md`
- `docs/ops/shared_memory_fabric_lifecycle_contract_v1.md`
- `state/continuity/latest/true_expanded_roadmap_queue_layer.json` (`XG-803` authoritative queue entry)

Formal fold-in constraints:
- `reports/true_expanded_roadmap_formal_layers_foldin_2026-03-28.md`
  - Domain Safety Layer: fail-close incident semantics
  - Learning/Adaptation Layer: incident-to-learning handoff packets

---

## 3) Contracted artifacts

### 3.1 Canonical schema/template
- Schema: `docs/ops/schemas/domain_failclose_incident_packet.schema.json`
- Template: `docs/ops/templates/domain_failclose_incident_packet.template.json`

### 3.2 Validation/runtime entrypoint
- `python scripts/domain_failclose_incident_gate.py --packet <packet.json> --json`

The gate is fail-closed and must emit explicit block reasons on first failing gate.

---

## 4) Normative requirements

A domain incident is valid only when all requirements pass:

1. **Schema-valid deterministic packet**
   - Packet must validate against `clawd.domain_failclose_incident_packet.v1`.

2. **Mandatory fail-close action semantics**
   - If incident class is `policy_violation`, `boundary_breach`, or `safety_guard_trip`, then `failclose_action.triggered` must be `true`.
   - Missing trigger for these classes is a hard block.

3. **Owner + timebound remediation contract**
   - `remediation.owner` must be non-empty.
   - `remediation.due_at` must be valid and strictly after `detected_at`.
   - `remediation.status` must be explicit (`open|in_progress|verified|expired`).

4. **Incident-to-learning handoff (mandatory)**
   - `lesson_handoff.required` must be `true`.
   - `incident_to_lesson_handoff_ref` and `knowledge_queue_ingestion_trace_ref` must resolve to existing in-repo artifacts.
   - `ingestion_status=blocked` is a hard block for closeout.

5. **Remediation closure verification**
   - If remediation status is `verified`, both `closure_verified_at` and `closure_verification_ref` are mandatory.
   - `closure_verification_ref` must resolve to an existing in-repo artifact.

---

## 5) Canonical block reasons

- `incident_schema_invalid`
- `incident_schema_gate_unavailable`
- `failclose_action_missing`
- `operator_surface_ref_unresolved`
- `evidence_ref_unresolved`
- `remediation_owner_missing`
- `remediation_due_missing_or_invalid`
- `remediation_not_timebound`
- `lesson_handoff_missing`
- `incident_to_lesson_ref_unresolved`
- `knowledge_queue_trace_missing_or_unresolved`
- `ingestion_trace_status_blocked`
- `remediation_closure_unverified`
- `remediation_verification_ref_unresolved`

---

## 6) Required closeout evidence for XG-803

- incident schema artifacts,
- runtime incident simulation output,
- operator remediation references,
- incident-to-lesson handoff packet,
- knowledge queue ingestion trace,
- remediation closure verification packet.

All evidence artifacts must be in canonical continuity surfaces under `state/continuity/latest/`.

---

## 7) Closeout criteria for XG-803

`XG-803` is complete only when:
1. this contract is canonicalized,
2. incident schema/template are published,
3. runtime gate emits PASS on a fully-populated incident packet,
4. negative simulation proves fail-close block behavior,
5. incident-to-learning handoff + knowledge ingestion trace artifacts exist,
6. remediation closure verification evidence is present,
7. source-of-truth map is updated with XG-803 references,
8. queue-layer `XG-803` is transitioned to `DONE` with evidence refs.
