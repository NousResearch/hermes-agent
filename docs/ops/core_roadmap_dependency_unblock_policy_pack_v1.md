# Core Roadmap Dependency-Unblock Policy Pack v1 (Canonical Governance Contract)

Date: 2026-03-28  
Status: active (**canonical governance contract** for dependency-definition unblock policy)  
Owner lane scope: A4 + B2 + B6 policy governance

Machine artifacts:
- Contract payload: `state/continuity/latest/core_roadmap_dependency_unblock_policy_pack_v1.json`
- Contract schema: `docs/ops/schemas/core_roadmap_dependency_unblock_policy_pack.schema.json`

## Purpose

This contract promotes the dependency-unblock policy pack from bounded support policy to canonical governance authority for three previously policy-definition-blocked slices:

- Slice 12 (`a4_broad_enforcement_expansion_and_lease_generalization`) needed an explicit cross-lane fast-path topology policy map.
- Slice 22 (`b2_multi_case_capacity_orchestration`) needed concrete starvation and concurrency limits tied to node capability.
- Slice 30 (`b6_multi_provider_bakeoff_governance_and_cockpit_action_ux`) needed explicit graduation criteria across cost/speed/accuracy/safety.

The contract is intentionally bounded: it unblocks readiness only for policy-definition blockers and does not claim downstream runtime implementation completion.

## Canonical contract decisions

### 1) A4 fast-path lane topology policy (Slice 12)

- Default cross-lane posture is `ticket_required` (fail-closed for unknown paths).
- Fast-path exceptions are explicitly allowlisted and constrained:
  - low/medium risk only,
  - strict TTL (`<=120s`),
  - fencing-term match required,
  - attestation required,
  - deny unknown callsites.
- High/critical risk cross-lane mutation remains non-fast-path only.

### 2) B2 multi-case capacity + starvation control (Slice 22)

- Scheduler policy is fixed to weighted round-robin with aging.
- Hard concurrency maxima are tied to node class (`small|medium|large`) and globally capped.
- Starvation is explicitly bounded with wait-time and skip-count thresholds, plus deterministic alert names.
- Preemption trigger is defined for persistent starvation pressure.

### 3) B6 provider bakeoff graduation thresholds (Slice 30)

- Graduation requires bounded evaluation windows and consecutive pass windows.
- Thresholds explicitly encode:
  - accuracy tolerances vs incumbent,
  - latency ratio caps,
  - cost ratio/budget constraints,
  - safety floor (abstention and error-budget discipline).
- Cockpit action UX governance is explicit: threshold failure auto-rejects, and action-card approval + scorecard attachment are mandatory before promotion.

## Schema guard (required)

The payload at
`state/continuity/latest/core_roadmap_dependency_unblock_policy_pack_v1.json`
**must validate** against:
`docs/ops/schemas/core_roadmap_dependency_unblock_policy_pack.schema.json`.

Fail-closed contract posture:
- schema mismatch => contract is non-canonical for execution claims,
- missing required slice policy (`12|22|30`) => contract invalid,
- unknown/malformed critical fields => contract invalid.

## Operator-surface visibility (required)

Operator Mission Control must expose this contract as a first-class projection via:
- `state/continuity/latest/operator_mission_control.json` → `dependency_policy_pack`
- mission-control headline rollups:
  - `dependency_policy_pack_status`
  - `dependency_policy_pack_schema_ok`
  - `dependency_policy_pack_slice_count`
  - `dependency_policy_pack_required_slice_coverage`

This keeps dependency-governance truth visible and auditable from the canonical operator surface.

## Queue-truth interaction

This policy pack unblocks readiness only for slices whose blocker reason was missing policy/definition.

It does **not** override dependency completion semantics:
- dependent slices remain blocked until upstream dependencies are actually `DONE`.

## XR-005 canonical promotion closeout evidence (2026-03-28)

Promotion evidence bundle for `XR-005 dependency_policy_pack_canonical_promotion`:

- Schema validation packet:
  - `state/continuity/latest/xr_005_dependency_policy_pack_schema_validation_2026-03-28.json`
- Operator-surface snapshot packet:
  - `state/continuity/latest/xr_005_dependency_policy_pack_operator_surface_snapshot_2026-03-28.json`

Deterministic validation command:

```bash
bash ops/openclaw/architecture/validate_contracts.sh --json
```

Pass criteria (fail-closed):
- `state/continuity/latest/core_roadmap_dependency_unblock_policy_pack_v1.json` validates against `docs/ops/schemas/core_roadmap_dependency_unblock_policy_pack.schema.json`.
- `state/continuity/latest/operator_mission_control.json` publishes `dependency_policy_pack` projection and headline rollups with schema/coverage OK.

## Canonical references

- `docs/ops/lane_topology_authority_contract_v1.md`
- `docs/ops/controlled_cross_lane_bridge_contract_v1.md`
- `docs/ops/shared_memory_fabric_lifecycle_contract_v1.md`
- `docs/ops/model_qualification_rollout_gate_contract_v1.md`
- `docs/ops/model_pool_policy_v1.json`
- `state/continuity/latest/core_roadmap_execution_queue.json`
- `state/continuity/latest/core_roadmap_slice_queue_2026-03-28.json`
- `state/continuity/latest/core_roadmap_queue_layer.json`
