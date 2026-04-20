# Hermes Mem (design stub)

This directory holds the initial design assets for the planned native `hermes_mem` memory provider.

Current contents:

- `schema.sql` — initial SQLite/FTS5 schema for the MVP operational-memory store

Status:

- design/documentation only
- provider implementation not added yet
- no `__init__.py` on purpose, so discovery does not treat this as a loadable memory provider yet

## What the initial schema covers

The first schema is designed around four core storage concerns:

1. `observations`
   - reusable memory units with stable IDs
   - session/project/profile/workspace scoping
   - kind, summary, detail, importance, confidence, and source metadata

2. `observation_tags` and `observation_links`
   - lightweight tagging
   - explicit graph relationships between observations

3. `session_summaries`
   - one summary row per session for cheap recall and viewer support

4. `recall_log`
   - audit trail of what was selected and reinjected during recall

## Design intent

This schema is for a local-first Hermes memory system that stores reusable operational observations rather than raw transcripts.

The transcript source of truth remains Hermes's existing session database. `hermes_mem` is intended to sit above that layer as a compact, searchable memory index.
