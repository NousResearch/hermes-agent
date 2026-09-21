# deja-recall

Opt-in, local prior-session recall for Hermes. It indexes clean user/assistant transcript content from canonical `state.db` files into a separate SQLite FTS5 cache, then injects a small deterministic context block on the first turn of a new session. It makes no LLM, embedding, or network calls.

Enable it in `config.yaml`:

```yaml
plugins:
  enabled: [deja-recall]
  entries:
    deja-recall:
      enabled: true
      session_paths: [~/.hermes/state.db]
      index_path: ~/.cache/hermes/recall/index.sqlite3
      result_count: 3       # 1..10
      recency_days: 30      # bounded ranking signal, lexical match remains primary
      max_injected_chars: 2400  # 128..8000
```

The index is incremental per source and per session. Cache changes are transactional. Missing, busy, malformed, or unsupported stores fail open and never prevent Hermes startup. `on_session_finalize` only queues background refresh work. Retrieval excludes the active session, deduplicates identical excerpts, and uses only `messages.content`; `api_content`, inactive messages, tool rows, and prior `<filesystem-recall>` blocks are not indexed.

The cache is rebuildable and contains transcript excerpts. Protect it with the same filesystem permissions as `$HERMES_HOME`. Delete the configured index file to force a cold rebuild.

See `USAGE.md` for setup, configuration details, injected-context format, retention/privacy guidance, troubleshooting, and how to clear or rebuild the index.
