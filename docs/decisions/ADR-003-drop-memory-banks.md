# ADR-003: Drop the `memory_banks` table

## Status
Accepted — 2026-05-07

## Context
The `memory_banks` table (`store.py:77`) stores per-category bundled HRR
vectors keyed by `cat:<category>`. It is written by
`MemoryStore._rebuild_bank` from seven call sites — `add_fact`,
`update_fact`, `remove_fact`, `rebuild_all_vectors`, `rename_entity`,
`merge_entities`, `canonicalize_existing_facts`.

Pre-flight grep confirms **zero readers**. The HRR retrieval path
(`probe`, `related`, `reason`) does per-fact unbind+similarity over the
fact-level `hrr_vector` column; the bundled bank vectors are never
consulted in any ranking, filtering, or candidate-pruning logic. The
table is a write-only side-effect of every fact mutation.

The bank machinery may have been intended as a coarse-grained candidate
pre-filter (compare the query to a category's bundled vector first, then
unbind only the matching category's facts), but that path was never
wired up. Keeping it costs:
- Storage: `dim * 8` bytes per category, rewritten on every fact write.
- Latency: every `add_fact` / `update_fact` / `remove_fact` triggers a
  full read of all that category's vectors, a bundle, and a write.
- Correctness blast radius: bank rebuilds run inside the atomic
  transactions of `rename_entity` / `merge_entities`, expanding the
  surface area for migration failures.
- Cognitive load: anyone reading the rename/merge code wonders what the
  banks are for.

## Decision
- **Migration:** `DROP TABLE memory_banks;`. Appended to the same v1→v2
  migration introduced in ADR-002 — both drops are ordered in one
  transaction, both apply on the next boot of any v1 DB.
- Remove `MemoryStore._rebuild_bank` and every call site (`add_fact`,
  `update_fact`, `delete_fact` (`remove_fact`), `merge_entities`,
  `rename_entity`, `rebuild_all_vectors`, `canonicalize_existing_facts`).
  Surrounding logic stays — only the rebuild call is excised.
- Remove the `memory_banks` schema definition from `store.py:77`.
- **Doctor:** extend `schema_version` check to assert `memory_banks` is
  absent from `sqlite_master`.

## Consequences

**Behavior change.** None observable. HRR retrieval is unaffected
(per-fact unbind path was always primary). Fact-mutation paths get
faster — no more per-write bundle/insert.

**Public API.** `rename_entity` and `merge_entities` keep their
`categories_rebuilt` return field as informational metadata describing
which categories were *affected* by the rename, even though no banks
are rebuilt. The field is now a description of the change set, not a
record of side effects performed.

**Migration cost.** Trivial — `DROP TABLE memory_banks` against the
existing rows. Wrapped in the same v1→v2 transaction as the
retrieval_count drop, so both succeed or both roll back.

**`hermes doctor` verification.** The `schema_version` check now also
asserts the table is absent from `sqlite_master`. A live-fire
demonstration before commit confirms the assertion fails on a
pre-migration DB and passes after the migration runs.

**Reversibility.** Reversible by re-creating the table from `_SCHEMA`
and re-introducing `_rebuild_bank`. The lost bundle data is
unrecoverable but no decision depended on it; recomputing all banks
from `facts.hrr_vector` is an O(n) walk if ever needed.
