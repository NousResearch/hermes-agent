# Lane Boundary Contract v1

Date: 2026-03-20  
Status: active (wave-4 slice-1 contract)  
Parent doctrine: `docs/ops/unified_operating_doctrine_v1.md`

## 0) Intent and non-goals

This contract defines deterministic lane boundaries for multi-lane operation:
- canonical lane identity
- lane-local memory scope
- allowed cross-lane inputs/outputs
- contamination boundaries
- explicit crossover packet types: `signal`, `ticket`, `deep_review`

Authority lease/fencing and mutation-ticket governance are defined in:
- `docs/ops/lane_topology_authority_contract_v1.md`

Non-goals in v1:
- no scheduler policy rewrite
- no queue schema migration
- no mutation of existing role contract semantics

## 1) Column-A preservation (explicit)

This contract **does not change** current Column A doctrine (no-nudge trust/autonomy first).

Preserved invariants:
1. Column A remains the highest-priority trust/control lane.
2. Fail-close and no-false-green posture remain mandatory.
3. Cross-lane exchanges are additive governance, not authority replacement.

## 2) Canonical lane identity

A lane instance is identified by the tuple:

- `lane_id` (stable lane class): `lane.column_<a|b|c>.<slug>`
- `lane_epoch_id` (reset epoch): `epoch_<id>`
- `work_item_id` (objective anchor)

Canonical primary lanes:
- `lane.column_a.no_nudge_autonomy`
- `lane.column_b.swarm_orchestration`
- `lane.column_c.upgrade_substrate`

Identity rules:
1. `lane_id` is stable across normal progress for the same lane class.
2. `lane_epoch_id` changes only on hard reset/re-seed.
3. Cross-lane packets must include both sender and receiver lane IDs + epochs.
4. Any packet with unknown lane ID or epoch mismatch is invalid and must be rejected.

## 3) Lane-local memory scope

Each lane has lane-local working memory with strict ownership:

- **Local-only memory:** transient notes, tentative hypotheses, in-flight scratch state.
- **Shared-contract memory:** approved artifacts/schemas/reports already in canonical paths.
- **External/unverified memory:** raw captures or model output not yet validated.

Scope rules:
1. A lane may write only to its declared output sinks for the current slice.
2. Lane-local memory is **not** globally authoritative.
3. Another lane may consume lane-local content only through an explicit crossover packet.
4. Raw lane-local context must not be injected into another lane implicitly.

## 4) Allowed lane inputs/outputs

### Allowed inputs (for any lane)
1. Canonical doctrine/contracts/schemas.
2. Own lane-local memory.
3. Explicit crossover packets that pass contract validation.

### Allowed outputs (for any lane)
1. Lane-local artifacts in declared scope.
2. Canonical artifacts in approved shared sinks.
3. Crossover packets (`signal`, `ticket`, `deep_review`).

### Forbidden outputs
1. Direct mutation of another lane's local memory/scratch state.
2. Implicit handoff via chat-only narrative without packetized fields.
3. Cross-lane authority escalation without packet + evidence refs.

## 5) Contamination boundaries

A crossover packet is the only allowed cross-lane contamination boundary.

Mandatory boundary controls:
1. `contamination_guard` metadata is required on all crossover packets.
2. `max_inline_context_bytes` must be <= 2048.
3. Unverified content must be marked (`contains_unverified_content=true`) and cannot be promoted without gate.
4. Requested cross-lane writes must be explicit (`cross_lane_write_requested=true`) and constrained to declared paths.
5. Promotion gate must be declared (`none | validator_required | human_required`).

Fail-close rules:
- invalid schema => reject
- unknown packet type => reject
- missing required evidence for `ticket` / `deep_review` => reject
- stale/expired packet (when `expires_at` present and in past) => reject

## 6) Crossover packet types (v1)

### `signal`
Purpose: lightweight state/health/attention transfer.  
Semantics: informational or warning path; no implicit execution grant.

Required payload fields:
- `signal_id`, `signal_code`, `severity`, `status`, `observed_at`

### `ticket`
Purpose: request bounded execution from one lane to another.  
Semantics: executable ask with explicit done condition and verification contract.

Required payload fields:
- `ticket_id`, `requested_outcome`, `definition_of_done`
- `allowed_write_paths[]`, `verification_commands[]`, `due_at`

### `deep_review`
Purpose: request deeper independent review/decision support.  
Semantics: read-heavy review lane crossover; does not authorize mutation by itself.

Required payload fields:
- `review_id`, `question`, `decision_type`
- `required_artifacts[]`, `response_deadline`

## 7) Implementation artifact (v1)

Schema landed:
- `docs/ops/schemas/lane_crossover_packet.schema.json`

Templates landed:
- `docs/ops/templates/lane_crossover_signal.template.json`
- `docs/ops/templates/lane_crossover_ticket.template.json`
- `docs/ops/templates/lane_crossover_deep_review.template.json`

Contracted schema version:
- `lane.crossover_packet.v1`

## 8) Adoption checklist

- [ ] Producers emit only schema-valid crossover packets.
- [ ] Receivers reject invalid/mismatched packets fail-closed.
- [ ] `ticket` and `deep_review` packets always carry evidence refs.
- [ ] No cross-lane work starts from narrative-only handoff.
