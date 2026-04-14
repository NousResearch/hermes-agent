# Health Lane Boundary & Safety Contract v1 (`XH-701`)

Date: 2026-03-28  
Status: active (canonical XH lane safety-boundary foundation)  
Owner: Architect  
Scope: Health subsystem lane (`XH-*`) in `/home/yeqiuqiu/clawd-architect`

---

## 1) Purpose

`XH-701` defines the fail-closed boundary and safety contract required before any Health lane schema/runtime activation.

This contract canonicalizes:
1. non-diagnostic scope boundaries,
2. risk-tier and refusal/escalation semantics,
3. emergency and incident escalation linkage,
4. governance dependency ceilings inherited from `XG-801`/`XG-803`.

This slice is foundation-only. It does **not** claim schema-pack completion (`XH-702`) or runtime activation (`XH-703`).

---

## 2) Canonical dependencies

- `state/continuity/latest/true_expanded_roadmap_queue_layer.json` (`XH-701..XH-703`)
- `docs/ops/c3_activation_governance_contract_v1.md`
- `state/continuity/latest/xg_801_c3_activation_risk_matrix_2026-03-28.json`
- `docs/ops/xg_803_domain_failclose_incident_contract_v1.md`
- `docs/ops/lane_boundary_contract_v1.md`
- `docs/ops/controlled_cross_lane_bridge_contract_v1.md`
- `reports/future_personal_health_system_implementation_queue_2026-03-20.md`

---

## 3) Scope boundary (allow / deny / escalate)

### 3.1 Allowed (in-boundary)
- Non-diagnostic health-support summaries using user-provided records.
- Habit, recovery, and wellness planning support with explicit uncertainty/provenance.
- Symptom/lab trend organization and question-prep for licensed clinicians.
- Safety-aware reminders to seek professional or emergency care when risk signals appear.

### 3.2 Denied (out-of-boundary)
- Medical diagnosis, treatment prescription, or replacing licensed clinician judgment.
- Emergency triage claims as authoritative medical advice.
- Autonomous medication/dosage recommendations.
- Hidden cross-lane writes or unsafe external actions without governed bridge contracts.
- Runtime activation when dependencies and safety gates are unresolved.

### 3.3 Escalate-before-proceed classes
- Any emergency or potentially life-threatening signal.
- Any request for diagnosis/treatment decisions.
- Any request with missing provenance, ambiguous identity/authority, or missing consent context.
- Any unknown risk class.

Fail-close default for unknown class: `REFUSE_AND_ESCALATE`.

---

## 4) Risk tiers and XG-801 alignment

Health lane action tiers:
- `HX0_INFORMATIONAL`: low-risk informational organization/summarization.
- `HX1_SUPPORTIVE`: bounded planning/reflection support with no medical decision claims.
- `HX2_CLINICAL_ADJACENT`: potentially consequential interpretation support requiring strict caveats and escalation posture.
- `HX3_SAFETY_CRITICAL`: emergency/safety-sensitive context.

Alignment to XG risk classes:
- `HX0_INFORMATIONAL` -> `RG0_LOW`
- `HX1_SUPPORTIVE` -> `RG1_MODERATE`
- `HX2_CLINICAL_ADJACENT` -> `RG2_HIGH`
- `HX3_SAFETY_CRITICAL` -> `RG3_CRITICAL`

Activation ceilings:
- `HX0/HX1`: governance validation only; no autonomous health action execution.
- `HX2`: blocked from runtime activation until `XH-702` + `XG-802` + `XG-803` obligations are satisfied.
- `HX3`: hard-blocked from runtime activation until domain safety contract + incident fail-close readiness are proven (`XH-703` prerequisite posture).

---

## 5) Refusal and escalation contract

Escalation levels:
1. `HE1_BOUNDARY_REFUSAL` — denied/out-of-scope health request.
2. `HE2_APPROVAL_OR_CONTEXT_REQUIRED` — insufficient consent/authority/provenance context.
3. `HE3_MEDICAL_PROFESSIONAL_ESCALATION` — clinical/diagnostic request must be redirected to licensed care.
4. `HE4_EMERGENCY_ESCALATION` — potential acute risk; immediate emergency-care routing guidance, no advisory delay.
5. `HE5_GOVERNANCE_BLOCK` — dependency/release/owner obligations unresolved.

Fail-close rules:
- Unknown risk class -> `HE5_GOVERNANCE_BLOCK`.
- Diagnostic/treatment request -> `HE3_MEDICAL_PROFESSIONAL_ESCALATION`.
- Emergency indicators -> `HE4_EMERGENCY_ESCALATION`.
- Missing owner/dependency context -> `HE5_GOVERNANCE_BLOCK`.

---

## 6) Escalation playbook linkage (required)

Health-lane escalation handling must link to canonical incident playbooks/contracts:

- `docs/ops/incident_playbooks/blindness_recovery.md` (signal-loss/operator escalation baseline)
- `docs/ops/incident_playbooks/restore_drill.md` (recovery/restore operating baseline)
- `docs/ops/xg_803_domain_failclose_incident_contract_v1.md` (domain incident fail-close packet and remediation contract)

Normative rule:
- Any `HE4` or repeated `HE3` pattern in runtime slices must emit a fail-close incident packet compliant with `XG-803` before any activation expansion claim.

---

## 7) Required `XH-701` artifacts

- this boundary/safety contract,
- safety gate test packet,
- escalation playbook linkage artifact,
- closeout report and queue evidence refs.

Deferred to follow-on slices:
- `XH-702`: typed health-record schema/template pack.
- `XH-703`: ingestion/support runtime under safety gates.

---

## 8) Validation entrypoints

- `python -m json.tool state/continuity/latest/xh_701_health_lane_safety_gate_tests_2026-03-28.json`
- `python -m json.tool state/continuity/latest/xh_701_escalation_playbook_linkage_2026-03-28.json`
- `python ops/openclaw/continuity/check_source_of_truth_map_regressions.py --repo-root . --map-path reports/openclaw_system_source_of_truth_map_2026-03-20.md --json`

---

## 9) Closeout condition for `XH-701`

`XH-701` is complete only when:
1. this boundary/safety contract is canonical,
2. safety gate tests and escalation playbook linkage artifacts are published,
3. source-of-truth map references this contract/artifacts under XH lane,
4. queue slice `XH-701` is `DONE` with bounded evidence refs,
5. no schema/runtime completion claims are made for `XH-702`/`XH-703`.
