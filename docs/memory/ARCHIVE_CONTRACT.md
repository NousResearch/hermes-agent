# Memory Archive Contract — Phase 1.5 (updated through Phase 3)

Status: SPEC + implementation notes. Phases 2 and 3 are implemented; this
document is the binding contract between the session subsystem (owner of raw
transcripts) and the memory subsystem (reader/indexer).

Hard invariants (non-negotiable):

- Raw conversations are the SOURCE OF TRUTH. The SQLite index is a DERIVED
  cache that can always be regenerated from the raw archive files.
- No summarization. The index stores RAW message chunks verbatim.
- No automatic extraction. Nothing is pulled, rewritten, or inferred.
- No embeddings. Search is lexical (FTS5) + sqlite-like fallback only.
- The memory subsystem ONLY READS raw transcripts. It never creates, mutates,
  compacts, or "optimizes" them. (See §0 ownership rule.)

## 0. Session vs memory ownership (Phase 3)

The session subsystem OWNS creation and mutation of raw transcripts
(`<HERMES_HOME>/sessions/<id>.jsonl`). The memory subsystem ONLY reads and
indexes them.

This rule exists to prevent a future contributor from reasoning: "Since memory
owns archives, let's optimize the JSON / rewrite transcripts / strip fields."
Memory does NOT own archives. If a transcript needs to change, the session
subsystem changes it; memory re-indexes (idempotent DELETE+INSERT by
`source_file`) on the next refresh. Nothing in the memory subsystem may write
to a `sessions/*.jsonl` path.

## 1. Archive storage location

- Root: `<HERMES_HOME>/sessions/` (existing convention; Phase 2 indexes this
  in place — no copy, no `archive/` migration). Phase 2 also recognizes
  `<HERMES_HOME>/archive/` if present, for forward compatibility.
- One file per session: `<session_id>.jsonl`.

## 2. File format

- One file per session. Format: JSON Lines (`.jsonl`), one JSON object per
  line, UTF-8. This is append-friendly and stream-parseable without loading the
  whole file.
- Each line is one event. Recognized event shapes (all include `role` and a
  timestamp):
  - `{"role": "user", "content": "...", "ts": "<iso8601>"}`
  - `{"role": "assistant", "content": "...", "ts": "<iso8601>", "tool_calls": [...]}`
  - `{"role": "system", "content": "...", "ts": "<iso8601>"}`
  - `{"role": "tool", "content": "...", "ts": "<iso8601>", "name": "..."}`
- The existing `~/.hermes/sessions/*.jsonl` files already match this shape, so
  Phase 2 can either (a) point the indexer at that existing location, or
  (b) copy into `<HERMES_HOME>/archive/`. Contract does not mandate which; it
  mandates the shape.

## 3. Metadata requirements

Per archive FILE, the index must capture (stored as columns, not inferred):

- `source_file` — absolute path (provenance, required).
- `session_id` — parsed from filename.
- `memory_layer` — constant `"L3-archive"` for every chunk from archive files.
- `created_at` — file mtime (deterministic, stable across rebuilds).
- `event_ts` — the per-line `ts` when present (for recency sort), else mtime.
- `role` — the line's `role` (user/assistant/system/tool) for filtering.

Optional enrichment (added in Phase 2, best-effort — NEVER blocks indexing if
unavailable):

- `session_id` — parsed from filename (required for archive; always present).
- `event_ts` — the per-line `ts` when present, else file mtime.
- `chunk_index` — ordinal within the file (preserves order; aids provenance).
- `project_context` — optional; current project focus if discoverable.
- `hermes_version` — optional; hermes_cli package version if importable.
- `git_commit` — optional; short HEAD sha if a git repo is discoverable from cwd.
- `working_directory` — optional; cwd when git_commit was captured.

All optional fields are gathered without raising; absence simply omits them
from `extra`. They enrich provenance but never prevent indexing.

Per CHUNK (row in `notes`):

- `content` — the raw line content, NOT summarized.
- `chunk_index` — ordinal within the file (preserves order; aids provenance).
- `extra` — JSON blob carrying `session_id`, `role`, `event_ts`, `chunk_index`.

## 4. Indexing behavior

- The unified `notes` table (Phase 1) is extended, NOT split. Archive chunks
  are inserted with `memory_layer = "L3-archive"`. The `memory_layer` column is
  the only routing discriminant — no schema migration needed for new layers.
- Indexer discovery (`_discover_sources`) gains an archive glob:
  `<HERMES_HOME>/archive/**/*.jsonl` (and/or the existing `sessions/*.jsonl`,
  decided in Phase 2). Each `.jsonl` line becomes one chunk.
- FTS5: archive chunks participate in the SAME `notes_fts` virtual table as
  markdown chunks. No separate FTS index. Search ranks across all layers.
- `retrieval_method` provenance for an archive hit = `"fts5"` or
  `"sqlite-like"` (same as today). Every result still carries full provenance
  (source_file, memory_layer, timestamp, snippet, role, session_id).
- Rebuild determinism: archive files processed in sorted (path, line) order;
  chunks keep source position. Rebuild is idempotent.

## 5. Retention policy

- Raw archive files: retained indefinitely by default (source of truth). No
  automatic deletion in Phase 2.
- SQLite index: regenerable; safe to delete and rebuild. Not a retention
  target.
- Optional trim: a future cron MAY prune archive files older than N days, but
  ONLY after explicit user approval and ONLY by deleting raw files (the index
  is then rebuilt). The contract forbids pruning the index while keeping raw
  files orphaned, and forbids "summarized then discard raw".
- Privacy-driven redaction (see §6) is the only in-Phase-2 transform, and it
  applies to the INDEX copy, never mutates the raw file.

## 6. Privacy considerations

- Raw archive = full fidelity, including any secrets Joe typed. Treated as
  trusted, local, source-of-truth.
- The index is a LOCAL cache only. No archive chunk or extracted content ever
  leaves the machine via the index path.
- Redaction: the indexer MAY redact a fixed denylist of high-risk patterns
  (API keys, private key blocks, passwords) from the indexed `content` copy
  ONLY, leaving the raw file untouched. Redaction is lexical/mask-based (e.g.
  replace matched secret with `***REDACTED***`), never LLM-based. This is the
  single allowed transform and is opt-in via config (`memory.index.redact`).
- Provenance still points to the true `source_file` even for redacted chunks,
  so a hit can be audited against the raw (authorized) source.
- Holographic / Graphiti / any external backend: NOT wired. Archive data is
  never sent to them. This contract does not authorize any external sync.

## 7. Relationship between archive files and SQLite

```
  ~/.hermes/sessions/<session>.jsonl   (RAW, truth, owned by session subsystem)
        │   discovered + chunked (1 line -> 1 chunk)  [read-only]
        ▼
  memory/index.db  (DERIVED cache, gitignored, rebuildable)
     notes (memory_layer="L3-archive", content=raw line, extra={session_id,role,...})
     notes_fts (FTS5 over notes.content)
     index_pending (source_file, enqueued_at, attempts, last_error, last_attempt, status)
        │   search
        ▼
  provenance: source_file + session_id + role + event_ts + chunk_index
```

- Delete `index.db` -> no data loss. Rebuild from archive + markdown.
- Delete an archive file -> that session's chunks vanish from the index on next
  rebuild; the markdown layers (L1) are unaffected.
- The index NEVER becomes authoritative. Any divergence is resolved by rebuild.

## 8. Archive lifecycle management (Phase 3)

Goal: make the existing session archive lifecycle EXPLICIT — a closed session
becomes searchable automatically, without changing the memory trust model and
without the memory subsystem ever touching a raw transcript.

### 8.1 Trigger — `on_session_end`

The ONLY hook the memory subsystem listens to is the existing plugin hook
`on_session_end(session_id, completed, interrupted, model, platform)`. The
session/memory subsystem fires this at every close boundary:

- TUI: `tui_gateway/server.py::_finalize_session`
- CLI: `cli.py` exit handler (interrupt path) + per-turn via `run_conversation`
- Gateway: per-turn via `run_conversation` + final boundary via expiry watcher
- ACP: `acp_adapter/session.py::SessionManager.cleanup` (emit added in Phase 3)

The listener must NOT live inside any close path. Close paths stay clean
("Session ended.") and only emit the hook. All archive logic is in the
registered listener (`hermes_cli/memory_index/archive_lifecycle.py`).

Verified close-path emit map (Phase 3 audit):
- TUI: `tui_gateway/server.py::_finalize_session` (direct emit)
- ACP: `acp_adapter/session.py::SessionManager.cleanup` (direct emit, added Phase 3)
- CLI / Gateway expiry / oneshot / /reset: all route through
  `run_agent.py::AIAgent.shutdown_memory_provider`, which emits the hook
  (added Phase 3). A single emit there covers all of them.

### 8.1b Semantics — event is an indexing opportunity, not finalization

The archive lifecycle consumer treats `on_session_end` as an **indexing
opportunity**, not a cryptographic finalization event. A future developer must
NOT assume:

    on_session_end  ==  "this archive is immutable forever"

It currently works because (a) the raw `sessions/<id>.jsonl` is stable once the
session closes, and (b) indexing is idempotent (DELETE+INSERT by `source_file`)
and lag-tolerant. None of that makes the event a permanent seal. If a transcript
is later corrected, reopened, or re-exported, re-indexing the same `source_file`
is the correct and expected recovery — not an anomaly to "guard against." The
hook firing is a signal to *try to make the session searchable*, not a verdict
that the indexing outcome is authoritative or irreversible.

Note on semantics: in this codebase `on_session_end` is run/turn-scoped, not
strictly lifetime. That is fine — the listener is idempotent (DELETE+INSERT by
`source_file`) and lag-tolerant, and the lazy safety net (§8.3) guarantees a
closed session is searchable by the next `hermes memory search` / `status`.

### 8.2 Incremental indexing — boring on purpose

- `index_session(jsonl_path)`: `DELETE FROM notes WHERE source_file = ?` then
  re-INSERT that file's rows. Idempotent, retry-safe, no diffing, no partial
  updates, no migration.
- On `on_session_end(session_id, ...)`: resolve `source_file =
  <HERMES_HOME>/sessions/<session_id>.jsonl`; if it exists, enqueue it into
  `index_pending` (UPSERT: insert or reset `status='pending'`, `attempts=0`).

### 8.3 Pending queue (SQLite `index_pending`)

```
index_pending
--------------
source_file   TEXT PRIMARY KEY
enqueued_at   TEXT
attempts      INTEGER DEFAULT 0
last_error    TEXT
last_attempt  TEXT
status        TEXT  -- 'pending' | 'failed' | 'done'
```

- `refresh_pending()`: drain every `status IN ('pending','failed')` row:
  `index_session()` it; on success `status='done'`; on failure increment
  `attempts`, set `last_error`, `status='failed'`. Errors are caught per-row so
  one bad file never blocks the others.
- Lazy safety net: `search()` / `archive_stats()` call `refresh_pending()`
  first (cheap when empty). So even if the async flush never ran, the next
  query makes pending sessions searchable — no daemon required.
- Async flush: a fire-and-forget background thread (or `threading.Timer`) drains
  `refresh_pending()` after enqueue. It MUST NEVER block the close path; if it
  raises, it is swallowed. Memory can lag slightly; memory can never interrupt
  work.

### 8.4 Failure behavior

- Indexing fails? Logged; `last_error` + `attempts` recorded; raw file untouched.
- Recoverable? Always — raw transcript was never mutated (ownership rule §0).
- Retry? Yes — `refresh_pending()` retries `failed` rows on the next trigger;
  `hermes memory index --retry` forces an immediate drain. Idempotent DELETE+
  INSERT makes retries safe.

### 8.5 `hermes memory status` — L3 archive block

```
L3 conversation archive:
  Indexed sessions: 542
  Pending:          0
  Failed:           0
  Last refresh:     2026-07-09 08:32
```

When `pending>0` or `failed>0`, a trailing note shows the last error. Idle
(zero indexed) shows the block with zeros and no error line.

## Resolved questions

1. Index location: EXISTING `~/.hermes/sessions/*.jsonl` in place (Phase 2). No
   `archive/` copy required; `archive/` still recognized if present.
2. Intent: extend `HISTORICAL` to include L3 (Phase 2). No separate `ARCHIVE`
   intent.
3. Redaction: off-by-default, explicit opt-in config (`memory.index.redact`).
   Lexical/mask-only, index copy only, raw file never touched.
