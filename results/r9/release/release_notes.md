# Holographic Release Notes (R9)

Freeze of the local-first deterministic memory stack (R1-R8). Zero LLM, offline-capable, no paid APIs, no core Hermes changes.

## Known limitations
- Runner Python 3.11 lacks numpy: 10 HRR tests skip by convention.
- 16 historical cognitive tests target the superseded pre-R1 API (preserved under results/r8/legacy/).
- Legacy DBs without HRR vectors are backfilled on first bank rebuild (derived state only).
- Backups are verbatim snapshots: protect backup files (no local encryption facility).
- Rollback = restore known-good backup (no transactional downgrade).
- Unwired `maintenance.py` carries an `enable_llm`/`max_llm_calls` hook with no backend wired by default: zero-LLM holds on every path; the hook is legacy surface, not a live dependency.
- Worktree is uncommitted by design (no commit requested): the freeze is defined by `manifest.json` (path+size+sha256) against base commit `45a6101f`, not by a release commit.
