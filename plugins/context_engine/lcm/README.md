# Hermes LCM: context engine plugin (B04)

Opt-in, DAG-based context management for Hermes Agent. It replaces the built-in
`ContextCompressor` when the user asks for it, and gets out of the way
completely when they do not.

```yaml
# config.yaml
context:
  engine: "lcm"     # omit (or set "compressor") for the built-in engine
```

## What it does

When the live context approaches the model's limit, the engine compacts the
middle of the conversation into a **summary node** and leaves a marker message
behind:

```
[LCM summary node n2213a4...] Compacted 8 message(s) at DAG level 1. The raw
messages are retained unchanged and recoverable in bounded pages: call
lcm_expand with node_id="n2213a4..." (page_size <= 50).
```

The originals are **not** deleted. Every node records one lineage edge per
message it absorbed, so the compacted range can be reconstructed later through
bounded pages. Compacting an already-summarized range points at the superseded
*node* rather than at its marker, so the structure stays a DAG instead of a
chain of copies.

| File | Role |
|---|---|
| `__init__.py` | entry point: `LCMEngine` + `register(ctx)` (+ `/lcm`) |
| `engine.py` | the `ContextEngine` implementation and the agent-facing tools |
| `storage.py` | SQLite store: raw messages, nodes, lineage, FTS index, migration, backup |
| `compaction.py` | DAG compaction and the deterministic digest |
| `recall.py` | bounded recall: `search` / `expand` / `page` / `status` |
| `config.py` | `context.lcm` settings with hard ceilings |
| `skills/hermes-lcm/` | the recall skill and its policy references |

## Agent-facing tools

`lcm_search`, `lcm_expand`, `lcm_page`, `lcm_status`. See
`skills/hermes-lcm/references/recall-tools.md` for arguments and response
shapes, and `recall-policy.md` for why recall is session-scoped.

## Bounds are contractual

`page_size` (default 10 / ceiling 50), `max_search_results` (default 8 /
ceiling 25) and `body_chars` (default 2000 / ceiling 8000) are enforced in the
storage layer. A config or a caller may lower them. Nothing can raise them.
There is no call that returns a whole session. `lcm_page` walks it one page at
a time with a cursor, and node lineage is paged the same way.

## Rollback

Unset `context.engine`, set it to `compressor`, misspell the engine name, or
have the plugin fail to import or construct: in every case the host keeps the
built-in `ContextCompressor` and the `lcm_*` tools are absent. Nothing needs to
be repaired or reconciled. The store is a read-only record of what happened.
See `tests/plugins/context_engine/test_lcm_rollback.py`.

## Design notes

- **No model calls.** The digest is deterministic and extractive, so compaction
  cannot fail on a provider error and its output is testable.
- **Fail-open.** `compress` returns the caller's list unchanged on any internal
  failure. The ingest hook swallows storage errors. Tool calls return JSON
  errors rather than raising.
- **Thread-safe.** Compression can run on a pooled daemon thread
  (`compression.context_timeout_seconds`), so the SQLite connection is opened
  with `check_same_thread=False` and every statement is serialized by a lock.
- **No `select_context()` replacement.** Per-turn context rewriting would
  change the prompt-cache prefix every turn. Recall is an explicit tool call
  instead, so installing the engine does not by itself alter caching.
