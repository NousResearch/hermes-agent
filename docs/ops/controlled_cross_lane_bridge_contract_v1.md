# Controlled Cross-Lane Bridge Contract v1

Date: 2026-03-20  
Status: active (wave-4 slice-4 contract)  
Parent doctrine: `docs/ops/unified_operating_doctrine_v1.md`  
Aligned contracts:
- `docs/ops/lane_boundary_contract_v1.md`
- `docs/ops/promotion_protocol_contract_v1.md`

## 0) Intent and non-goals

This contract defines how **approved artifacts** cross lane boundaries using a deterministic bridge envelope, without raw-context leakage.

Goals in v1:
- allow only explicit bridge object classes,
- require provenance + approval metadata,
- enforce contamination guards,
- fail closed with machine-readable rejection reasons.

Non-goals in v1:
- no runtime queue/scheduler refactor,
- no automatic cross-lane writes,
- no broad retrieval/routing redesign.

## 1) Core invariants (normative)

1. **Promotion, not leakage**: bridge traffic carries structured object refs + bounded metadata, not raw transcript/context dumps.
2. **Lane boundary remains authoritative**: lane identity/epoch checks from Lane Boundary Contract v1 still apply.
3. **Promotion protocol remains authoritative**: doctrine/playbook/memory promotion semantics still require Promotion Protocol gates.
4. **Reference-first transfer**: bridge payloads are pointer-led (`path` + `content_hash`) by default.
5. **Fail closed**: unknown class/version/fields or unavailable checks must be rejected.

## 2) Bridge object model (v1)

All cross-lane artifact transfers MUST use:
- schema: `docs/ops/schemas/cross_lane_bridge_object.schema.json`
- schema version: `lane.bridge_object.v1`
- template: `docs/ops/templates/cross_lane_bridge_object.template.json`

The bridge envelope must include:
- lane tuple (`from_lane_id`, `from_lane_epoch`, `to_lane_id`, `to_lane_epoch`),
- work anchor (`work_item_id`),
- object class + deterministic object ref (`object_id`, `path`, `content_hash`, `schema_ref`),
- approval metadata,
- promotion/leakage metadata,
- contamination guard metadata.

## 3) Allowed bridge object classes (v1)

Allowed classes only:
1. `doctrine_object`
2. `promotion_candidate`
3. `lane_crossover_packet`
4. `evidence_closeout`
5. `approved_artifact_ref`

Any other class is invalid and must be rejected.

### 3.1 Class-specific gate requirements

#### `doctrine_object`
- `object_ref.schema_ref` must be `docs/ops/schemas/doctrine_object.schema.json`.
- Requires `promotion.promotion_required=true`.
- Requires `promotion.promotion_id` (`prom_*`).
- Requires `promotion.promotion_state in {APPROVED, PROMOTED}`.
- `promotion.leakage_check` must be `pass`.

#### `promotion_candidate`
- `object_ref.schema_ref` must be `docs/ops/schemas/promotion_candidate.schema.json`.
- `promotion.promotion_state` must not be `LOCAL_ONLY`.
- `promotion.leakage_check` must be `pass`.

#### `lane_crossover_packet`
- `object_ref.schema_ref` must be `docs/ops/schemas/lane_crossover_packet.schema.json`.
- Object is informational/request governance transfer; no implicit write authority.

#### `evidence_closeout`
- `object_ref.schema_ref` must be `docs/ops/schemas/evidence_closeout.schema.json`.
- Must include at least one source/evidence ref and reviewer decision ref.

#### `approved_artifact_ref`
- For approved canonical artifacts that do not yet have a dedicated schema.
- Requires explicit `schema_ref` string, `content_hash`, and promoted/approved decision linkage.
- Treated as reference-only object class in v1.

## 4) Contamination guard requirements

Every bridge object MUST include `contamination_guard` with:
- `source_memory_scope`: `lane_local_only | shared_contract_only | mixed`
- `contains_unverified_content`: boolean
- `cross_lane_write_requested`: boolean
- `promotion_gate`: `none | validator_required | human_required`
- `max_inline_context_bytes`: integer `0..512`
- `redaction_applied`: boolean
- `allow_inline_excerpt`: boolean

v1 policy constraints:
1. `max_inline_context_bytes` hard-capped at 512.
2. `inline_excerpt` is optional and bounded; reference-only transfer is default.
3. `contains_unverified_content=true` is not bridge-eligible in v1.
4. `classification=secret` is not bridge-eligible in v1.
5. `classification=restricted` requires redaction (`promotion.redaction_applied=true` and `contamination_guard.redaction_applied=true`).
6. `cross_lane_write_requested=true` requires an explicit `packet_type=ticket` crossover packet; non-ticket packets must fail closed.
7. Ticket crossover packets must carry explicit lease metadata (`operation_id`, `lease_mode`, `risk_tier`, `fencing_term`, `attestation_refs`) and comply with the approved fast-path topology policy (`state/continuity/latest/core_roadmap_dependency_unblock_policy_pack_v1.json`, slice 12).

## 5) Fail-closed rejection reasons (canonical v1)

Receivers/validators should emit one or more of:
- `schema_invalid`
- `schema_version_unsupported`
- `unknown_object_class`
- `lane_identity_mismatch`
- `lane_epoch_mismatch`
- `object_not_found`
- `content_hash_missing`
- `content_hash_mismatch`
- `object_schema_mismatch`
- `promotion_missing`
- `promotion_state_invalid`
- `promotion_gate_not_satisfied`
- `review_not_approved`
- `leakage_risk`
- `classification_forbidden`
- `redaction_required`
- `inline_context_over_limit`
- `unverified_content_blocked`
- `cross_lane_write_scope_violation`
- `expired_bridge_object`
- `gate_unavailable`

Unknown rejection reasons should be normalized to `schema_invalid` or `gate_unavailable` depending failure stage.

## 6) Deterministic validation order (v1)

1. Schema + required fields + enum checks.
2. Lane tuple validation (lane IDs + epoch compatibility).
3. Object-ref resolution (`path` exists, hash present/matching, schema_ref match by class).
4. Class-specific promotion/review gates.
5. Contamination + classification checks.
6. Expiry/time-window checks.
7. Consumer-local write-scope authorization.

On any gate failure: reject; do not partially apply.

## 7) v1 implementation artifacts

- Contract: `docs/ops/controlled_cross_lane_bridge_contract_v1.md`
- Schema: `docs/ops/schemas/cross_lane_bridge_object.schema.json`
- Template: `docs/ops/templates/cross_lane_bridge_object.template.json`

## 8) Out of scope for v1

- Automatic bridge ingestion into queue runtime.
- Cross-lane mutation execution engine.
- Policy exceptions for secret-class artifacts.
