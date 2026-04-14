# Downstream Capability Backend Contract v1 (`XB-401`)

Date: 2026-03-28  
Status: active (canonical for `XB-401`)  
Owner: Architect  
Scope: downstream capability backend ingress contracts (connector, adapter, domain-event) with policy-as-code security envelope

---

## 1) Purpose

`XB-401` defines the single-ingress backend contract required before any downstream capability runtime can integrate with OpenClaw continuity/governance lanes.

The contract standardizes:
1. connector identity and auth-tier semantics,
2. adapter/runtime boundary behavior,
3. domain-event envelope requirements,
4. policy-as-code controls for privileged actions,
5. credential-isolation and deny-by-default egress posture.

This slice is contract foundation only.
It does **not** implement capability registry runtime (`XB-402`) or eval/release gate pack (`XB-403`).

---

## 2) Canonical inputs

- `state/continuity/latest/true_expanded_roadmap_queue_layer.json` (`XB-401` authoritative slice contract)
- `docs/ops/c3_activation_governance_contract_v1.md` (`XG-801` activation governance dependency)
- `docs/ops/true_expanded_out_of_core_scope_filter_rules_v1.md`
- `docs/ops/lane_boundary_contract_v1.md`
- `docs/ops/controlled_cross_lane_bridge_contract_v1.md`

---

## 3) Contract bundle (normative)

### 3.1 Connector contract

Each connector declaration MUST include:
- `connector_id` (stable id),
- `domain_lane_id` (target lane scope),
- `auth_tier` (`ADMIN | OBSERVABILITY | INTERNAL | PUBLIC`),
- `route_class` (`advisory | state_mutation | sensitive_action`),
- `sensitivity` (`standard | privileged`),
- `allowed_event_types` (non-empty allowlist).

Fail-closed rules:
- unknown auth tier, route class, or sensitivity blocks ingress;
- empty event allowlist blocks ingress.

### 3.2 Adapter contract

Every adapter runtime declaration MUST enforce:
- `inference_proxy_required = true`,
- `credential_source = isolated_secret_store`,
- `worker_credential_access = false`,
- bounded retries with deterministic idempotency key,
- explicit timeout budget.

Fail-closed rules:
- direct provider calls without inference proxy are blocked;
- worker runtime credential materialization is prohibited.

### 3.3 Domain-event envelope

Each ingress domain event MUST include:
- deterministic identity (`event_id`, `event_type`, `occurred_at`),
- source provenance (`source_connector_id`, `producer_id`),
- governance metadata (`risk_class`, `route_class`, `auth_tier`),
- immutable integrity hash (`payload_sha256`).

Fail-closed rules:
- missing provenance/governance metadata blocks ingestion;
- malformed hash blocks ingestion.

### 3.4 Policy-as-code security envelope

Policy envelope MUST define:
- approval requirements for privileged connector actions,
- deny-by-default egress policy,
- explicit allowlisted destinations for any outbound path.

Minimum policy obligations:
- privileged actions require approval step chain (`governance_owner` + `release_owner`);
- non-allowlisted egress targets are denied;
- policy decision records are attributable and auditable.

---

## 4) Required artifacts for strict DONE

- contract doc:
  - `docs/ops/downstream_capability_backend_contract_v1.md`
- schema pack:
  - `docs/ops/schemas/downstream_capability_backend_contract_bundle.schema.json`
  - `docs/ops/templates/downstream_capability_backend_contract_bundle.template.json`
- security evidence artifacts:
  - `state/continuity/latest/xb_401_policy_as_code_contract_examples_2026-03-28.json`
  - `state/continuity/latest/xb_401_credential_isolation_test_packet_2026-03-28.json`
  - `state/continuity/latest/xb_401_deny_by_default_egress_verification_sample_2026-03-28.json`
- validation artifact:
  - `state/continuity/latest/xb_401_downstream_backend_contract_validation_2026-03-28.json`
- closeout report:
  - `reports/xb_401_downstream_capability_backend_contract_closeout_2026-03-28.md`

---

## 5) Validation entrypoints

- `pytest -q tests/test_xb_401_downstream_backend_contract_pack.py`
- `python -m json.tool docs/ops/schemas/downstream_capability_backend_contract_bundle.schema.json`
- `python -m json.tool docs/ops/templates/downstream_capability_backend_contract_bundle.template.json`
- `python -m json.tool state/continuity/latest/xb_401_policy_as_code_contract_examples_2026-03-28.json`
- `python -m json.tool state/continuity/latest/xb_401_credential_isolation_test_packet_2026-03-28.json`
- `python -m json.tool state/continuity/latest/xb_401_deny_by_default_egress_verification_sample_2026-03-28.json`
- `python ops/openclaw/continuity/check_source_of_truth_map_regressions.py --repo-root /home/yeqiuqiu/clawd-architect --map-path /home/yeqiuqiu/clawd-architect/reports/openclaw_system_source_of_truth_map_2026-03-20.md --json`

---

## 6) Closeout condition for `XB-401`

`XB-401` is complete only when:
1. this backend contract is canonical,
2. schema/template pack validates,
3. policy-as-code examples prove privileged approval path and deny-by-default egress posture,
4. credential-isolation packet proves worker runtimes cannot access connector credentials,
5. queue slice `XB-401` is transitioned to `DONE` with evidence refs and bounded status reason.
