# Knowledge Review / Approval / Promotion Queue v1

Date: 2026-03-20  
Status: active (bounded queue runtime + contract)  
Parent doctrine: `docs/ops/unified_operating_doctrine_v1.md`

## Purpose
Provide deterministic queue governance for promotion candidates:
1. enqueue candidate packets
2. capture explicit review/approval decisions
3. trigger promotion workflow gate in a controlled queue step

This closes the gap between contract-only promotion packets and operational review backlog handling.

## Artifacts
- queue entry schema: `docs/ops/schemas/knowledge_promotion_queue_entry.schema.json`
- queue entry template: `docs/ops/templates/knowledge_promotion_queue_entry.template.json`
- canonical runtime path: `scripts/knowledge_promotion_queue.py`
- promotion trace manifest schema/template: `docs/ops/schemas/promotion_trace_manifest.schema.json`, `docs/ops/templates/promotion_trace_manifest.template.json`
- shared memory lifecycle runtime/contract (memory target promotions):
  - `scripts/shared_memory_fabric.py`
  - `docs/ops/shared_memory_fabric_lifecycle_contract_v1.md`
- legacy compatibility helper (non-canonical): `scripts/knowledge_review_queue.py`

Upstream ingestion substrate (Wave 6):
- ingestion package contract: `docs/ops/knowledge_ingestion_package_workflow_v1.md`
- ingestion package schema/template: `docs/ops/schemas/knowledge_ingestion_package.schema.json`, `docs/ops/templates/knowledge_ingestion_package.template.json`
- ingestion helper runtime: `scripts/knowledge_ingestion_package.py`

## Queue states (v1)
- `QUEUED_REVIEW`
- `APPROVED`
- `PROMOTED`
- `REJECTED`
- `BLOCKED`

## Commands
- enqueue candidate (wrapper-dispatched mutator):
  - `bash ops/openclaw/continuity.sh --action-token <...> promotion-queue enqueue --candidate <promotion_candidate.json> --json`
- review decision (wrapper-dispatched mutator):
  - `bash ops/openclaw/continuity.sh --action-token <...> promotion-queue review --entry-id <id> --decision approved --reviewer-role VALIDATOR --reviewer-id <id> --json`
- promote approved entry (wrapper-dispatched mutator):
  - `bash ops/openclaw/continuity.sh --action-token <...> promotion-queue promote --entry-id <id> --json`
- list queue state (non-mutating; direct entrypoint allowed):
  - `python3 scripts/knowledge_promotion_queue.py list --json`

## Entrypoint parity contract (EX-09 closure)
- Mutating commands (`enqueue|review|promote`) are **wrapper-only** and fail closed on direct entrypoint usage.
- Required wrapper provenance env for mutators:
  - `OPENCLAW_INTERNAL_MUTATION=1`
  - `OPENCLAW_INTERNAL_MUTATION_CALLSITE=<allowlisted>`
- Default allowlisted callsites:
  - `continuity.sh:promotion-queue`
  - `knowledge_ingestion_package.py:enqueue`
- Override hook for controlled expansion: `OPENCLAW_KNOWLEDGE_PROMOTION_QUEUE_ALLOWED_CALLSITES` (comma-separated).
- Non-mutating command (`list`) remains directly callable.

## Fail-closed behavior
- Candidate path must resolve inside repo.
- Candidate packet must validate against promotion candidate schema before enqueue.
- Review updates mutate candidate packet review fields and promotion state atomically.
- Promotion step only runs from `APPROVED`; all other states block.
- Promotion workflow result must be machine-readable; parse failures block.
- Promotion trace manifest generation is mandatory on PASS; manifest failure blocks (`BLOCKED`).
- Memory-target promotions must materialize shared-memory typed object via `shared_memory_fabric.py`; materialization failure blocks (`BLOCKED`).

## Runtime outputs
Default state/log paths:
- queue state: `state/continuity/knowledge_promotion_queue/state.json`
- append-only events: `state/continuity/knowledge_promotion_queue/events.jsonl`
- promotion workflow snapshots: `state/continuity/knowledge_promotion_queue/workflow/<entry_id>.json`
- promotion trace manifests: `state/continuity/knowledge_promotion_queue/manifests/<entry_id>.json`
- shared memory objects/conflicts/demotions (for memory-target promotions):
  - `state/continuity/shared_memory/objects/*.json`
  - `state/continuity/shared_memory/conflicts/*.json`
  - `state/continuity/shared_memory/demotions/*.json`

## Out of scope (v1)
- distributed locking and concurrent writers across hosts
- SLA scheduling or automatic reviewer assignment
- automatic canonical-surface file mutation outside existing promotion gates
