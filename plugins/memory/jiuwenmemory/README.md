# JiuwenMemory Memory Provider

JiuwenMemory is a Hermes `MemoryProvider` plugin for the official JiuwenMemory memory-server HTTP API.

## Configuration

Enable it as the active memory provider:

```yaml
memory:
  provider: jiuwenmemory
```

Hermes plugin environment variables:

- `JIUWENMEMORY_API_KEY` - optional bearer token for authenticated memory-server deployments.
- `JIUWENMEMORY_BASE_URL` - memory-server URL. Defaults to `http://127.0.0.1:8000`.
- `JIUWENMEMORY_SCOPE_ID` - JiuwenMemory scope id. Defaults to `hermes`.
- `JIUWENMEMORY_USER_ID` - optional JiuwenMemory user id override. If unset, Hermes uses the runtime gateway user id when one is available.

The JiuwenMemory server has its own configuration, separate from the Hermes plugin. In the default server package this is loaded from `~/.jiuwenmemory/.env`; memory search/write depends on a working embedding backend there. The common server-side knobs are:

- `MODEL_PROVIDER`, `MODEL_NAME`, `API_KEY`, `API_BASE` for the LLM used by JiuwenMemory.
- `EMBED_MODEL_NAME`, `EMBED_API_KEY`, `EMBED_API_BASE` for embeddings used by vector search.

`EMBED_API_BASE` must point to an embeddings endpoint, not just the chat-completions base URL. For OpenAI-compatible providers this usually looks like `https://provider.example/v1/embeddings`; the model in `EMBED_MODEL_NAME` must be an embedding model that provider actually serves.

`jiuwenmemory_status` checks both `GET /health` and a lightweight memory-search probe. If `/health` is healthy but the probe reports `memory_search: failed`, fix the server-side embedding settings rather than the Hermes plugin switch.

When server vector search fails with the known JiuwenMemory embedding error, the plugin enables a degraded local fallback:

- Search first tries JiuwenMemory `/search_memory/`.
- If that fails because embeddings are unavailable, Hermes ranks memories from `/get_user_mem_by_page/` plus a local fallback journal with lexical matching.
- Failed `jiuwenmemory_store`, `sync_turn()`, and built-in memory mirror writes are preserved in `$HERMES_HOME/jiuwenmemory_fallback.jsonl`.
- `jiuwenmemory_status` reports `fallback_search: available` and includes `local_fallback` details when this degraded path can be used.

This fallback makes memory recall usable without new credentials, but it is not a replacement for JiuwenMemory native vector search. Configure `EMBED_MODEL_NAME`, `EMBED_API_KEY`, and `EMBED_API_BASE` on the server for full semantic search and server-side memory extraction.

Advanced settings live at `$HERMES_HOME/jiuwenmemory.json`:

```json
{
  "base_url": "http://127.0.0.1:8000",
  "scope_id": "hermes",
  "user_id": "",
  "threshold": 0.3,
  "max_recall_results": 8,
  "max_summary_results": 4,
  "auto_recall": true,
  "auto_capture": true,
  "local_fallback": true,
  "fallback_page_size": 50,
  "fallback_max_pages": 4,
  "enable_long_term_mem": true,
  "enable_user_profile": true,
  "enable_semantic_memory": true,
  "enable_episodic_memory": true,
  "enable_summary_memory": true
}
```

## API Usage

The plugin uses the official memory-server routes:

- `GET /health` for status.
- `POST /search_memory/` for long-term memory recall.
- `POST /search_user_history_summary/` for summary recall.
- `POST /add_messages/` for turn capture and explicit memory writes.
- `POST /get_user_mem_by_page/` through the internal client helper.

Recall payloads:

```json
{
  "query": "What does the user prefer?",
  "num": 8,
  "scope_id": "hermes",
  "user_id": "optional-user-id",
  "threshold": 0.3
}
```

Turn capture and explicit stores use `POST /add_messages/`:

```json
{
  "messages": [
    {"role": "user", "content": "hello"},
    {"role": "assistant", "content": "hi"}
  ],
  "scope_id": "hermes",
  "user_id": "optional-user-id",
  "enable_long_term_mem": true,
  "enable_user_profile": true,
  "enable_semantic_memory": true,
  "enable_episodic_memory": true,
  "enable_summary_memory": true
}
```

If no user id is configured or supplied by Hermes, the plugin omits `user_id` and lets the memory server apply its default.

## Tools

- `jiuwenmemory_search` - search long-term memory.
- `jiuwenmemory_store` - store an explicit durable memory through `/add_messages/`.
- `jiuwenmemory_status` - check memory-server health and report the configured scope.

`prefetch()` searches both memories and user-history summaries and injects them as `<jiuwenmemory-context>`. `sync_turn()` and built-in memory write mirroring run in daemon threads so normal chat turns do not block on HTTP latency.

## Development Artifacts

`scripts/benchmark_jiuwenmemory_tokens.py` and `scripts/jiuwenmemory_server_audit.py` are source files and should remain visible in `git status`. Benchmark JSON reports written under `benchmark_results/` are generated runtime output and are ignored by git.
