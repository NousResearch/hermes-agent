# Cognee Memory Plugin (V2)

Knowledge graph memory with session-aware storage, auto-routing recall, and a learning loop that bridges session data into the permanent graph. Built on the [Cognee V2 API](https://github.com/topoteretes/cognee).

## What changed from V1

- **sync_turn** now writes to session cache (fast) instead of running add+cognify every turn
- **recall** auto-routes to the best search strategy and searches session memory first
- **forget** tool added for deleting memories
- **on_session_end** runs `improve()` to bridge session data into the permanent knowledge graph
- Remote/cloud mode via `cognee.serve()`

## Setup

### Install

```bash
pip install cognee
```

### Local mode (no server)

Cognee runs locally using SQLite + LanceDB + Kuzu. Requires an LLM API key for entity extraction and graph reasoning.

```bash
export LLM_API_KEY="sk-..."       # OpenAI or other LLM provider key
export LLM_MODEL="gpt-4o-mini"    # optional
```

Or configure via `$HERMES_HOME/cognee.json`:

```json
{
  "llm_api_key": "sk-...",
  "llm_model": "gpt-4o-mini",
  "dataset": "hermes"
}
```

### Server mode (legacy)

Point to a running Cognee server:

```bash
export COGNEE_BASE_URL="http://localhost:8000"
export COGNEE_API_KEY="your-api-key"
```

### Remote/Cloud mode (V2)

Connect to a Cognee Cloud or remote instance:

```bash
export COGNEE_SERVICE_URL="https://your-instance.cognee.ai"
export COGNEE_API_KEY="your-api-key"
```

### Activate

Set `memory.provider` in your Hermes config:

```yaml
memory:
  provider: cognee
```

Or run `hermes memory setup` and select Cognee.

## How it works

### Per-turn: session cache write

Each conversation turn is stored in Cognee's session cache via `remember(data, session_id=...)`. This is a lightweight write -- no entity extraction or graph processing runs. The Hermes session ID is mapped to a Cognee session ID with a `hermes_` prefix.

### On recall: session-first with graph fallback

When the agent calls `cognee_recall`, the query goes through Cognee's `recall()` which:
1. Searches the session cache by keyword matching (fast)
2. If no session results match, falls through to the permanent knowledge graph
3. Auto-routes to the best search strategy (GRAPH_COMPLETION, RAG, CHUNKS, etc.)

Each result is tagged with `_source: "session"` or `_source: "graph"`.

### On session end: improve loop

When the Hermes session ends, `improve(session_ids=[...])` runs a 4-stage pipeline:
1. Apply feedback weights from session to graph nodes/edges
2. Persist session Q&A into the permanent knowledge graph
3. Enrich graph with triplet embeddings (memify)
4. Sync enriched graph back to session cache

This replaces V1's costly add+cognify on every turn with one batch at session end.

## Config Reference

| Key | Env Var | Default | Description |
|-----|---------|---------|-------------|
| `llm_api_key` | `LLM_API_KEY` | -- | LLM provider API key (local mode) |
| `llm_model` | `LLM_MODEL` | -- | LLM model name |
| `base_url` | `COGNEE_BASE_URL` | -- | Cognee server URL (legacy server mode) |
| `api_key` | `COGNEE_API_KEY` | -- | Cognee server/remote API key |
| `serve_url` | `COGNEE_SERVICE_URL` | -- | Cognee Cloud/remote URL (V2 mode) |
| `dataset` | `COGNEE_DATASET` | `hermes` | Default dataset name |
| `top_k` | -- | `5` | Default max results |
| `auto_route` | -- | `true` | Auto-select search strategy in recall |
| `improve_on_end` | -- | `true` | Run improve() on session end |
| `session_prefix` | -- | `hermes` | Prefix for Cognee session IDs |

## Tools

| Tool | Description |
|------|-------------|
| `cognee_recall` | Search knowledge graph and session memory. Auto-routes to best strategy. Supports `scope` parameter: auto/session/graph. |
| `cognee_remember` | Permanently store content in the knowledge graph (full add+cognify+improve pipeline). |
| `cognee_forget` | Delete data -- target a specific dataset or delete everything. |
