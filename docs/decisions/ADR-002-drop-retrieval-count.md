# ADR-002: Drop `facts.retrieval_count`

## Status
Accepted — 2026-05-07

## Context
The `facts` table carries a `retrieval_count` column, declared in
`store.py:31` and incremented by `MemoryStore.search_facts` (the FTS5
keyword path) on every match. It is also projected into the SELECT lists
of four retrieval query sites (`retrieval.py:163-164`, `:222-223`,
`:290-291`, plus `list_facts` in `store.py:454-455`).

Pre-flight grep confirms the column has **zero readers** in any
ranking, filtering, decay, archival, or display logic. It is read into
Python dicts via `_row_to_dict` and never consumed. The HRR retrieval
paths (`probe`, `related`, `reason`) do not write it at all — only the
FTS5 path does. So the column is both inconsistent (only one of two
retrieval paths writes it) and dead (no path reads it for a decision).

ADR-001 closed an analogous dead-write on `helpful_count`. This ADR
removes the column itself rather than the write, because there is no
future signal we plan to derive from it: recency wants a `last_seen_at`
timestamp, not a counter; archival wants age + trust, not retrieval
frequency. Keeping the column is carrying tax for a feature that does
not exist.

## Decision
- **Migration:** SQLite table-rebuild to drop `facts.retrieval_count`.
  All indexes and triggers re-created. Wrapped in transaction.
- Remove the `UPDATE facts SET retrieval_count = ...` write from
  `MemoryStore.search_facts`.
- Remove `retrieval_count` from SELECT lists in `retrieval.py:163-164`
  and the three other retrieval query sites — grep for `retrieval_count`
  to confirm full removal.
- Remove from `list_facts` projection and any Python dataclass/TypedDict
  mirroring the row shape.
- **Doctor:** extend `schema_version` check to assert `retrieval_count`
  is absent from `PRAGMA table_info(facts)`.

## Consequences

**Migration cost.** One-time table rebuild on next boot of any v1 DB.
Indexes (`idx_facts_trust`, `idx_facts_category`), the FTS5 virtual
table `facts_fts`, and its three triggers (`facts_ai`, `facts_ad`,
`facts_au`) are dropped and recreated; FTS5 content is rebuilt via the
`'rebuild'` command. Wrapped in a single `BEGIN`/`COMMIT` so a mid-flight
crash leaves the v1 DB intact.

**Schema version.** Bumps `_CURRENT_SCHEMA_VERSION` from 1 to 2 and
introduces a `_MIGRATIONS = {target: fn}` registry in `_init_db`. ADR-003
extends the same v2 migration in a follow-up commit.

**Behavior change.** None observable. No code reads the column.
`search_facts` no longer issues a redundant UPDATE per query — small
latency win on hot paths.

**`hermes doctor` verification.** The `schema_version` check now also
asserts the column is absent from the live `facts` table. A live-fire
demonstration before commit confirms the assertion fails on a synthetic
v1 DB and passes after the migration runs.

**Reversibility.** Reversible by adding the column back via
`ALTER TABLE facts ADD COLUMN retrieval_count INTEGER DEFAULT 0` and
re-introducing the UPDATE write. The dropped data (the counter values)
is unrecoverable, but no decision depended on it.
