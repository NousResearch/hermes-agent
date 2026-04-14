# Knowledge Ingestion Package Workflow v1

Date: 2026-03-20  
Status: active (wave-6 slice-3 bounded substrate)  
Depends on: `docs/ops/knowledge_review_approval_promotion_queue_v1.md`

## Purpose
Introduce a bounded **production ingestion package** layer in front of Wave 5 queue/review/promotion runtime.

This package layer preserves evidence and fixity metadata so queue admission is backed by deterministic provenance rather than ad-hoc narrative context.

## Scope (v1)
- Define ingestion package schema + template.
- Capture preserved evidence bundle metadata (path/hash/bytes/timestamp).
- Capture provenance metadata (`capture_method`, tool traces, source ref linkage).
- Enforce fixity checks before queue handoff.
- Provide minimal runtime helper for create/validate/enqueue.

## Non-goals (v1)
- No archive backend redesign.
- No vector-store ingestion refactor.
- No automatic promotion write bypassing existing Wave 5 gate workflow.
- No distributed locking across hosts.

## Contracts
- Ingestion package schema:
  - `docs/ops/schemas/knowledge_ingestion_package.schema.json`
- Ingestion package template:
  - `docs/ops/templates/knowledge_ingestion_package.template.json`
- Runtime helper:
  - `scripts/knowledge_ingestion_package.py`

## Package model summary
Top-level contract fields:
- `schema_version`: `clawd.knowledge_ingestion.package.v1`
- `package_id`: `kip_*`
- `ingestion_state`: `DRAFT | READY_FOR_QUEUE | QUEUED_REVIEW | BLOCKED`
- `promotion_candidate_ref`: candidate path + hash + promotion id
- `preserved_evidence`: itemized evidence package with content hashes
- `provenance`: capture metadata + source ref mapping
- `fixity`: verification status + mismatch counters
- `handoff`: queue linkage (`queue_entry_id`, `queue_status`, timestamps)

## Runtime commands
Build package from a promotion candidate:
- `python3 scripts/knowledge_ingestion_package.py create --candidate <promotion_candidate.json> --json`

Validate package (schema + provenance/fixity runtime checks):
- `python3 scripts/knowledge_ingestion_package.py validate --package <ingestion_package.json> --json`

Validate + hand off to Wave 5 queue runtime:
- `python3 scripts/knowledge_ingestion_package.py enqueue --package <ingestion_package.json> --json`

## Fail-closed behavior
- Candidate/package path outside repo => block.
- Missing validator/schema => block.
- Any schema mismatch => block.
- Candidate hash mismatch => block.
- Evidence file missing or hash mismatch => block.
- Queue handoff parse/runtime failures => block.
- Already-queued package (`queue_entry_id` present) => block (requires explicit operator intent for replay/requeue).

## Alignment to Wave 5 stack
`enqueue` does not invent a new promotion lane. It delegates to:
- `scripts/knowledge_promotion_queue.py enqueue --candidate ...`

This keeps review/approval/promotion authority in the existing queue contract and uses ingestion package only as a bounded evidence/provenance substrate.

## Wave 7 alignment update
Wave 7 now hardens downstream traceability by emitting promotion trace manifests (`clawd.promotion_trace_manifest.v1`) and shared-memory objects for memory targets during queue promotion. The deterministic chain is now:

`package_id -> queue_entry_id -> promotion_id -> workflow_decision -> promotion_trace_manifest -> shared_memory_object(optional)`

## Next recommended slice
- Add append-only replay checker that cross-verifies full Wave 7 chain above across multiple cases and time windows.
