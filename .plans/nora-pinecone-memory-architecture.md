# Nora Pinecone Memory Architecture / Rollout

## Current rollout state

- Phase 0 — dark launch: supported by `PINECONE_RECALL_ENABLED=0` or `PINECONE_RECALL_PHASE=0`.
- Phase 1 — debug retrieval only: supported by `PINECONE_RECALL_ENABLED=1` and `PINECONE_RECALL_PHASE=1`; prompt injection remains off.
- Phase 2 — limited prompt recall: supported by `PINECONE_RECALL_ENABLED=1` and `PINECONE_RECALL_PHASE=2`; defaults to at most 2 snippets and can be restricted with allowlists.
- Phase 3 — broader prompt recall: supported by `PINECONE_RECALL_ENABLED=1` and `PINECONE_RECALL_PHASE=3`; supports wider source coverage and explicit limits.

## Config flags

Prompt-time Pinecone recall is gated entirely by config/env flags:

- `PINECONE_RECALL_ENABLED` — master on/off switch for prompt retrieval.
- `PINECONE_RECALL_PHASE` — rollout phase integer.
- `PINECONE_RECALL_PLATFORMS` — comma-separated platform allowlist for phase 2+.
- `PINECONE_RECALL_SCOPES` — comma-separated scope allowlist for phase 2+.
- `PINECONE_RECALL_SOURCE_TYPES` — comma-separated `memory_type` values to query in phase 3+ (for example `project_context,profile`).
- `PINECONE_RECALL_MAX_ITEMS` — explicit snippet cap override.
- `PINECONE_RECALL_TOP_K` — explicit Pinecone candidate fetch override.

## Phase guidance

### Phase 0
- Ingest content into Pinecone only.
- Do not inject Pinecone recall into prompts.
- Inspect indexed content quality out of band.

### Phase 1
- Query Pinecone manually with admin/debug tools.
- Compare quality against `session_search`, repo files, and durable memory before allowing model-visible recall.
- Keep prompt injection disabled.

### Phase 2
- Enable prompt recall only for selected platforms/scopes first.
- Default to 1–2 snippets unless an explicit override is configured.
- Use quality review before widening beyond initial cohorts.

### Phase 3
- Increase `PINECONE_RECALL_MAX_ITEMS` and/or widen `PINECONE_RECALL_SOURCE_TYPES` after quality review.
- Keep the master switch and allowlists available for fast rollback.

## Verification checklist

- Integration coverage for:
  - no Pinecone configured
  - Pinecone configured with zero hits
  - Pinecone configured with relevant hits
  - stale hit vs fresher local source
  - long user message with multiple candidates
- Run `pytest tests/agent/test_context_retrieval.py -q`.
- Run targeted memory/session-search regressions with recall disabled and enabled.
- Run full `pytest tests/ -q` in both rollout modes before broad enablement.

## Explicit quality gate

Phase 2 should receive an explicit quality review before Phase 3 broad enablement. The safe rollback path is to set `PINECONE_RECALL_ENABLED=0` or lower `PINECONE_RECALL_PHASE` below 2.
