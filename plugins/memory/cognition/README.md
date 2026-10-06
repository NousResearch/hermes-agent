# Cognition memory provider

Session-end durable memory for Hermes: one active provider, facts routed
internally to configured repos.

- **Push is fixed to session end.** Turns accumulate; `on_session_end`
  extracts durable facts onto a SQLite FIFO queue. `sync_turn` persists
  nothing. `Pending` means unpushed (extracted, not yet committed).
- **Fetch-first merge, keep-both conflicts.** Push compares against committed
  state first; a changed base keeps both versions under deterministic refs
  (`cog-{sessionId}-{seq}`, conflicts `{ref}-conflict-{unix_ts}`).
- **Mandatory provenance.** Every entry carries `source_session`, `author`,
  `created_at`, `origin_repo`, `trace_id`. Eviction archives first
  (JSONL, never deleted by lifecycle).
- **Single-writer lock per entry id.** Dream reads committed state only.
- **Prefetch budget** (default 8000 tokens). Truncation order: system-pin >
  provenance > recency > salience, always with a marker.
- **Dream is read-only.** `cognition_dream` synthesizes committed state and
  surfaces conflicts; no write-back path exists by design.

Config (DRAFT key names): `memory.provider=cognition` with
`memory.cognition={repos, budget_tokens, queue, archive}`. Setup asks for the
author identity last — the configured author is authoritative for provenance;
empty falls back to host `git config user.name`.

Recall blends `MEMORY.md` + `USER.md` mirrors (when present, pinned) with
committed facts under the token budget.

## Layout

- `plugin.yaml` — manifest; `__init__.py` — `register(ctx)`
- `provider.py` — `CognitionMemoryProvider` (tools: `cognition_search`,
  `cognition_status`, `cognition_dream`)
- `queue.py` — durable FIFO (retry 1m/5m/30m+jitter, 24h cap, bound 1000
  drop-oldest-with-tombstone, backlog surfacing)
- `merge.py` — fetch-first push, keep-both, per-entry locks, commit log
- `provenance.py` — mandatory provenance + eviction-preserving archive
- `budget.py` — prefetch budget + truncation order + marker
- `tests/` — one acceptance test per contract condition (`python3 -m
  unittest discover -s tests -t tests` from this directory)
