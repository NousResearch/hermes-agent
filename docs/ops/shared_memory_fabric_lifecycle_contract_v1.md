# Shared Memory Fabric Lifecycle Contract v1

Date: 2026-03-21  
Status: active (Wave 7 lane closeout contract)  
Parent lanes: B1 shared memory fabric, B2 research lifecycle, B5 artifact/evidence OS

## Purpose
Define deterministic typed-memory lifecycle semantics for promoted knowledge:
- canonical typed object materialization after successful promotion,
- contradiction conflict-set recording with explicit ownership,
- staleness/divergence demotion records for canonical authority control.

## Canonical runtime + contracts
- Runtime: `scripts/shared_memory_fabric.py`
- Object schema/template:
  - `docs/ops/schemas/shared_memory_object.schema.json`
  - `docs/ops/templates/shared_memory_object.template.json`
- Conflict record schema/template:
  - `docs/ops/schemas/shared_memory_conflict_record.schema.json`
  - `docs/ops/templates/shared_memory_conflict_record.template.json`
- Demotion record schema/template:
  - `docs/ops/schemas/shared_memory_demotion_record.schema.json`
  - `docs/ops/templates/shared_memory_demotion_record.template.json`

## Entrypoint parity contract (EX-09 closure slice)
- Mutating commands (`promote|conflict|demote`) are **wrapper-only** and fail closed on direct entrypoint usage.
- Required wrapper provenance env for mutators:
  - `OPENCLAW_INTERNAL_MUTATION=1`
  - `OPENCLAW_INTERNAL_MUTATION_CALLSITE=<allowlisted>`
- Default allowlisted callsite: `continuity.sh:shared-memory`.
- Override hook for controlled expansion: `OPENCLAW_SHARED_MEMORY_FABRIC_ALLOWED_CALLSITES` (comma-separated).
- Non-mutating utility commands (`status|compact|retrieve|benchmark`) remain directly callable.

## Typed object model
Every canonical shared-memory object MUST include:
1. promotion trace (`promotion_id`, queue entry, candidate/workflow hashes),
2. memory payload (`object_type`, title/statement, canonical state, target path),
3. freshness policy (`stale_after_days`, `demote_after_days`, accountable owner),
4. traceability refs (`source_refs`, decision refs, implementation refs).

## Lifecycle states
Canonical state enum:
- `PROMOTED_CANONICAL`
- `CONFLICTED`
- `DEMOTED_STALE`
- `DEMOTED_SUPERSEDED`
- `DEMOTED_INVALIDATED`
- `ARCHIVED`

## Conflict-set workflow
`shared_memory_fabric.py conflict` emits append-only contradiction records with:
- both object IDs,
- owner identity + role,
- reason,
- status (`pending|resolved_keep|resolved_merge|resolved_demote`),
- optional resolution notes.

Pending conflict sets move the primary object to `CONFLICTED`.
Resolved-demote conflict decisions may move object state to `DEMOTED_SUPERSEDED`.

## Staleness/demotion policy
`shared_memory_fabric.py demote` emits append-only demotion records with:
- demotion kind (`stale|superseded|invalidated|manual`),
- explicit owner/role,
- previous state and new state.

Demotion updates freshness state to `demoted` and records a deterministic timestamp.

## Fail-closed expectations
- Invalid object IDs/roles/states block.
- Unknown demotion/conflict statuses block.
- Schema validation failures block writes.
- Unsafe repo-outside paths block.

## Integration expectation
Canonical memory promotions should route through:
1. promotion gate PASS,
2. queue promote,
3. shared-memory object materialization,
4. promotion trace manifest emission.

This keeps source→synthesis→decision→implementation traceability machine-verifiable.

## Wave 8 extension — compaction + retrieval ergonomics
The shared-memory runtime now supports deterministic compaction and retrieval benchmarking:

- `shared_memory_fabric.py compact`
  - emits a compacted assertion snapshot (`state/continuity/shared_memory/compaction/latest.json` by default),
  - groups semantically equivalent typed assertions,
  - preserves source object IDs + relationship references (conflicts/demotions),
  - reports byte reduction metrics.

- `shared_memory_fabric.py compact --compaction-strategy {legacy|signature}`
  - `legacy` (default): deterministic token-order-preserving grouping keyed by normalized object_type + title/statement.
  - `signature` (optional): stop-word-reduced token-signature grouping for paraphrase deduplication.

- `shared_memory_fabric.py retrieve --query ...`
  - default retrieval strategy is token-overlap over compacted assertions,
  - supported strategies:
    - `--retrieval-strategy token_overlap`
    - `--retrieval-strategy tfidf_hybrid` (token-weighted + optional phrase match boost)
  - returns ranked compacted assertions and the union of matched source object IDs.

- `shared_memory_fabric.py benchmark --golden-queries ...`
  - validates compaction reduction and retrieval recall against golden queries,
  - supports `--compaction-strategy` and `--retrieval-strategy` for benchmark mode,
  - fails closed when configured thresholds are not met.

### Closeout target for this extension
- Compaction reduction threshold: `>= 50%`
- Retrieval recall threshold: `>= 95%` on golden expected object IDs
