# Hermes Mem Implementation Plan

Date: 2026-04-17

## Goal

Add a native, local-first operational memory subsystem to Hermes that continuously captures useful observations from agent work, stores them with stable IDs, supports progressive recall, and exposes a local viewer for audit/debugging.

This plan is inspired by the product shape of claude-mem, but it is deliberately designed around Hermes's current internals rather than as a direct clone.

## Why This Should Exist

Hermes already has:

- durable compact memory through the built-in `memory` tool and `USER PROFILE`
- cross-session recall through `session_search`
- a provider abstraction for external memory backends
- context compression and persistent session storage

But Hermes does not yet have a native operational memory layer that turns activity into reusable observations with:

- stable observation IDs
- layered recall
- timeline navigation
- local auditability
- explicit separation between transcript history and reusable memory units

## Design Direction

Implement a new memory provider:

- `plugins/memory/hermes_mem`

Use the existing provider lifecycle:

- `initialize`
- `prefetch`
- `queue_prefetch`
- `sync_turn`
- `on_session_end`
- `on_pre_compress`
- `on_delegation`

Use local storage first:

- SQLite
- FTS5

Add progressive recall tools:

- `memory_search`
- `memory_timeline`
- `memory_get`

Add a local viewer:

- read-only first
- audit/debug focused rather than polished

## Core Concepts

### Observation

The primary durable unit should be an `observation`, not a raw transcript chunk.

Every observation should have at least:

- `id`
- `session_id`
- `project`
- `kind`
- `title`
- `summary`
- `detail`
- `source_type`
- `source_ref`
- `created_at`
- `importance`
- `tags`

Recommended observation kinds:

- `fact`
- `decision`
- `bugfix`
- `investigation`
- `tool_result`
- `file_change`
- `user_preference`
- `delegation_result`
- `session_summary`

### Progressive recall

The retrieval flow should work in 3 layers:

1. `memory_search`
   - compact index of candidate memories
2. `memory_timeline`
   - chronological context around a selected memory or query
3. `memory_get`
   - full detail fetch only for selected IDs

This keeps token usage low and makes memory auditable.

## Proposed Work Breakdown

### Phase 1: provider scaffold

Create:

- `plugins/memory/hermes_mem/__init__.py`
- `plugins/memory/hermes_mem/plugin.yaml`
- `plugins/memory/hermes_mem/README.md`

The provider should be loadable through the existing plugin system and selectable by `memory.provider`.

### Phase 2: storage layer

Create:

- `plugins/memory/hermes_mem/store.py`
- `plugins/memory/hermes_mem/schema.sql`  ← initial draft already committed

Add tests:

- `tests/plugins/memory/test_hermes_mem_store.py`

Initial tables should cover:

- `observations`
- `observation_fts`
- `session_summaries`
- `recall_log`

### Phase 3: observation extraction

Create:

- `plugins/memory/hermes_mem/extraction.py`

Add tests:

- `tests/plugins/memory/test_hermes_mem_extraction.py`

Sources to extract from:

- completed turns via `sync_turn`
- session end via `on_session_end`
- delegation results via `on_delegation`
- pre-compression state via `on_pre_compress`

### Phase 4: retrieval tools

Add provider tools:

- `memory_search(query, kind=None, session_id=None, limit=10)`
- `memory_timeline(observation_id=None, query=None, window=5)`
- `memory_get(ids=[...])`

Add tests:

- `tests/plugins/memory/test_hermes_mem_search.py`
- `tests/plugins/memory/test_hermes_mem_timeline.py`
- `tests/plugins/memory/test_hermes_mem_get.py`

### Phase 5: automatic prefetch

Use `prefetch()` for compact reinjection of relevant observations.

Rules:

- inject no more than a short fenced context block
- prefer summaries, not full payloads
- include IDs for later citation/debugging
- keep the default result stable and cheap

### Phase 6: local viewer

Create:

- `plugins/memory/hermes_mem/viewer.py`

Add tests:

- `tests/plugins/memory/test_hermes_mem_viewer.py`

Suggested minimal endpoints:

- `GET /health`
- `GET /observations`
- `GET /observations/{id}`
- `GET /search?q=...`
- `GET /sessions/{id}/summary`
- `GET /recall/latest`

## Non-Goals For MVP

Do not include in v1:

- mandatory vector embeddings
- SaaS dependency for storage or retrieval
- polished dashboard UI
- broad collaborative knowledge-sharing workflows
- indiscriminate storage of raw tool outputs

## Important Design Rules

1. Keep built-in durable memory separate from operational memory.
2. Do not treat raw tool output as the memory unit.
3. Make every observation addressable by stable ID.
4. Keep the MVP local-first.
5. Optimize recall for token efficiency.
6. Make reinjected memory easy to inspect and debug.

## Relationship To Existing Features

### Built-in memory

Keep using the existing built-in memory for concise durable facts.

### `session_search`

Keep `session_search` transcript-centric.

`hermes_mem` should focus on reusable observations rather than conversation recap.

### Existing providers

Do not remove support for external providers.

`hermes_mem` should become the native local-first option in the same provider ecosystem.

## Open Questions

1. Should `hermes_mem` become the default provider later?
2. Should embeddings be optional in v1 behind a config gate, or deferred entirely?
3. Should the viewer ship inside the provider or as a separate memory-inspection module?
4. Should session-derived summaries reuse any `session_search` internals?

## Recommended First Implementation Order

1. provider scaffold
2. storage + schema
3. turn/session extraction
4. progressive recall tools
5. tests
6. viewer

That order delivers the core memory product without requiring a full UX layer first.
