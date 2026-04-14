# Health Typed Record Schema Pack v1 (`XH-702`)

Date: 2026-03-29  
Status: active (canonical for `XH-702`)  
Owner: Architect  
Scope: typed health-record schema/template pack for `XH-*` lane in `/home/yeqiuqiu/clawd-architect`

---

## 1) Purpose

`XH-702` lands the typed health-record schema pack required before health-runtime activation (`XH-703`).

This slice defines machine-validated contracts for:
1. measurements,
2. lab results,
3. symptoms,
4. protocols,
5. provenance and integrity anchoring,
6. confidentiality/privacy constraints,
7. governance/escalation metadata under non-diagnostic advisory boundaries.

This slice is schema-pack only. It does **not** activate runtime ingestion/support execution paths.

---

## 2) Canonical inputs

- `state/continuity/latest/true_expanded_roadmap_queue_layer.json` (`XH-702` authoritative queue contract)
- `docs/ops/health_lane_boundary_safety_contract_v1.md` (`XH-701` fail-closed boundary and escalation contract)
- `docs/ops/downstream_capability_backend_contract_v1.md` (`XB-401` provenance/governance/privacy envelope dependency)
- `docs/ops/c3_activation_governance_contract_v1.md`
- `docs/ops/lane_boundary_contract_v1.md`
- `docs/ops/controlled_cross_lane_bridge_contract_v1.md`
- `reports/future_personal_health_system_implementation_queue_2026-03-20.md`

---

## 3) Schema-pack artifacts (normative)

### 3.1 Typed record schema
- `docs/ops/schemas/health_typed_record.schema.json`

Defines required envelope fields:
- identity/lifecycle (`record_id`, `record_type`, `status`, `confidence`, `recorded_at`),
- typed payload branches (`measurement`, `lab_result`, `symptom`, `protocol`),
- provenance (`source_kind`, `source_ref`, `captured_at`, `payload_sha256`),
- confidentiality (`classification`, `contains_phi`, `sharing_policy`, `redaction_required`),
- governance (`risk_class`, `auth_tier`, `escalation_code`, advisory-only route).

### 3.2 Template
- `docs/ops/templates/health_typed_record.template.json`

Provides canonical starter payload aligned with non-diagnostic advisory posture and deidentified-sharing-safe defaults.

### 3.3 Fixtures
- `tests/fixtures/xh/health_typed_record_fixture_v1.json`
- `tests/fixtures/xh/health_typed_record_pack_fixture_v1.json`
- `tests/fixtures/xh/health_privacy_constraint_fixture_v1.json`

Fixtures provide representative valid records, cross-type coverage, and pass/fail privacy constraint cases.

### 3.4 Schema pack manifest
- `ops/openclaw/architecture/health_typed_record_schema_pack.v1.json`

Pins schema/template/fixture pointers for deterministic validation and evidence replay.

---

## 4) Mandatory control semantics for `XH-702`

1. **Non-diagnostic advisory boundary**
   - `governance.route_class` is hard-constant `advisory`.
   - `governance.advisory_only` is hard-constant `true`.

2. **Provenance and integrity anchoring**
   - every record must include immutable `payload_sha256` and explicit source metadata.

3. **Confidentiality fail-close rules**
   - PHI-bearing records (`contains_phi=true`) must require redaction and cannot use `deidentified_only` sharing.
   - `deidentified_only` records must assert `contains_phi=false` and `redaction_required=false`.
   - lab records must be `health_sensitive` or `health_restricted` (never `health_standard`).

4. **Escalation parity for critical risk**
   - `RG3_CRITICAL` records require `HE4_EMERGENCY_ESCALATION`.

5. **Canonical confidence floor**
   - records marked `canonical` must carry confidence >= 0.7.

---

## 5) Validation entrypoints

- `pytest -q tests/test_xh_702_health_typed_record_schema_pack.py`
- `bash ops/openclaw/architecture/validate_health_typed_record_schema_pack.sh --json`
- `python -m json.tool docs/ops/schemas/health_typed_record.schema.json`
- `python -m json.tool docs/ops/templates/health_typed_record.template.json`
- `python -m json.tool tests/fixtures/xh/health_privacy_constraint_fixture_v1.json`
- `python ops/openclaw/continuity/check_source_of_truth_map_regressions.py --repo-root /home/yeqiuqiu/clawd-architect --map-path /home/yeqiuqiu/clawd-architect/reports/openclaw_system_source_of_truth_map_2026-03-20.md --json`

---

## 6) Closeout condition for `XH-702`

`XH-702` is complete only when:
1. schema/template/fixture pack is canonical and valid,
2. privacy constraint checks pass,
3. provenance/confidentiality constraints are enforced fail-closed,
4. queue slice `XH-702` transitions to `DONE` with evidence refs,
5. no runtime completion claims are made for `XH-703`.
