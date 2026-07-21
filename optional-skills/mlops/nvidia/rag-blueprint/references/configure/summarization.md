# Document Summarization

## When to Use
- User wants to generate summaries during document ingestion
- User asks about summarization strategies or options
- User wants to check summary status or progress

## Restrictions
- Not supported in lite mode (containerless/library-only deployment)
- Requires Redis for status tracking and rate limiting
- Collection must exist before uploading with `generate_summary: true`

## Process
1. Detect the deployment mode. Docker: edit the active env file. Helm: configure under `ingestor-server.envVars` in `values.yaml`. Library: use the upload API parameters directly (no env vars needed)
2. Read `docs/summarization.md` for full configuration, env vars, and prompt customization
3. Set `generate_summary: true` in the upload payload (per-request, no global toggle)
4. Optionally configure `summary_options`: strategy, shallow mode, page filter
5. Retrieve summary via `GET /v1/summary?collection_name=...&file_name=...`

## Decision Table

| Goal | Strategy | Notes |
|------|----------|-------|
| Fastest overview | `"single"` + `shallow_summary=true` + `page_filter` | Quick text-only extraction |
| Best quality | `null` (iterative, default) + `shallow_summary=false` | Sequential refinement |
| Balanced | `"hierarchical"` + `shallow_summary=true` | Parallel tree-based |

## Agent-Specific Notes
- `CONVERSATION_HISTORY` prerequisite does not apply — that's for query rewriting only
- `SUMMARY_LLM_SERVERURL=""` (empty) routes to NVIDIA cloud; `"nim-llm:8000"` for self-hosted
- `SUMMARY_LLM_MAX_CHUNK_LENGTH` should be below the model's context window to leave room for prompt + output
- Redis semaphore auto-resets on ingestor startup (prevents stale values from crashes)
- If Redis is unavailable, summaries still generate but no real-time status tracking
- Status entries have 24-hour TTL in Redis

## Notebooks
- `notebooks/summarization.ipynb` — complete examples for all strategies, status polling, library mode usage

## Source Documentation
- `docs/summarization.md` — env var reference, prompt customization, rate limiting, chunking details
