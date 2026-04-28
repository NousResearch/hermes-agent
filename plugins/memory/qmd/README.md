# QMD memory provider

Read-only Hermes `MemoryProvider` plugin that queries a self-hosted
**QMD-compatible HTTP API**: a small JSON gateway over a vector index
(e.g. sqlite-vec) with optional cross-encoder reranking.

> This plugin connects to a remote QMD-compatible HTTP API. It does
> not vendor or depend on `@tobilu/qmd`. The protocol is OpenClaw's
> wire format — `POST /search` and `GET /health`.

This plugin only **reads**. Writes still flow through:

- the built-in `MEMORY.md` / `USER.md` path on the Hermes side, and
- the QMD ingest pipeline running alongside the QMD server.

Treat this provider as a remote RAG layer on top of curated memory
collections.

## Tools exposed

### `qmd_search(query, top_k=5, index=None, collection_filter=None)`
Semantic search against the remote QMD index. Returns ranked hits with
`docid`, `file` (a `qmd://collection/path` URI), `title`, `score`,
`rerank_score`, plus a truncated `context` and `snippet`.

- `query` (string, required) — natural-language query.
- `top_k` (int, default 5, max 25) — number of hits to return.
- `index` (string, default `default`) — index name. Use `qmd_status`
  to enumerate available indexes on your deployment.
- `collection_filter` (string, optional) — restrict hits whose
  `qmd://<collection>/...` URI's collection equals or starts with this
  prefix (e.g. `session-logs`, `global-facts`).

### `qmd_status()`
Hits the remote `/health` endpoint and returns the JSON: uptime,
monitored indexes, per-index document counts, embedding dimensions,
and the active rerank URL. Use to diagnose retrieval problems.

## System-prompt block
Three-line markdown summary identifying the active default index and
host, plus a one-liner reminder that the provider is read-only.

## Prefetch
Each turn, if the user message is at least
`MIN_PREFETCH_QUERY_LEN` (12) characters and the circuit breaker is
closed, a background thread fires a `top_k=3` search against the
default index. The result is stitched into the next-turn prefetch
context as `## QMD Memory` followed by `- **title** (uri)\n  snippet`
lines. Overlapping prefetches are guarded — if a previous thread is
still alive, the new one is skipped.

## Circuit breaker
Mirrors the mem0 plugin: 5 consecutive failures pauses calls for
120 seconds. Successes reset the counter. Cooldown reads
`time.monotonic()`.

## Configuration

Secrets live in `$HERMES_HOME/.env`:

```
QMD_REMOTE_API_TOKEN=<bearer token>
QMD_REMOTE_API_BASE_URL=http://localhost:18181   # optional
QMD_DEFAULT_INDEX=default                         # optional
QMD_TIMEOUT=30                                    # optional
```

Non-secret overrides go to `$HERMES_HOME/qmd.json`:

```json
{
  "base_url": "http://localhost:18181",
  "default_index": "default",
  "timeout": 30,
  "prefetch_top_k": 3,
  "manual_top_k": 5,
  "snippet_max": 300
}
```

`save_config()` only ever writes to `qmd.json`. The `api_token` field
is filtered out — it must come from the environment (the `hermes
memory setup` wizard sends it to `.env` because the schema marks it
`secret: True`).

## Wire protocol

The remote service must implement two endpoints:

- `POST /search` with body `{"query": str, "topK": int, "index": str}`
  returning `{"results": [{"docid", "file", "title", "score",
  "externalRerankScore" (optional), "context", "snippet"}]}`.
- `GET /health` returning a JSON object with at minimum `indexes`
  (per-index metadata) and `uptime_s`.

Both endpoints expect `Authorization: Bearer <QMD_REMOTE_API_TOKEN>`.

## Activation

Set the active provider in `$HERMES_HOME/config.yaml`:

```yaml
memory:
  provider: qmd
```

Or run `hermes memory setup` and select **qmd**.

## Limitations
- Read-only. Writes are out of scope.
- Single index per call; cross-index fan-out is intentionally not
  implemented — pick the right index or use `qmd_status` to enumerate
  what's available.
- The `collection_filter` is a simple prefix match on the
  `qmd://collection/...` URI segment — it is **not** a server-side
  filter. The full `top_k` results are fetched first, then filtered
  client-side. If you need precise scoping for a small collection,
  bump `top_k` accordingly.
- HTTP timeout defaults to 30 s. The underlying `httpx.Client` keeps
  up to 8 connections (4 keep-alive) for thread-safe concurrent use.
