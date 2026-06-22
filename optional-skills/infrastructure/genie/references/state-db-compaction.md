# State DB Compaction — Measured Results & Procedures

## FTS Footprint

FTS trigram indexes add significant overhead proportional to message count. Expect FTS overhead in the multi-GB range for large DBs (100K+ messages). FTS can be dropped and rebuilt on demand — it does not contain unique data, only search indexes.

FTS rebuild time scales linearly: ~8 minutes per 138K messages in a container, faster on the host.

## Decomposition of state.db Size

A large state.db (e.g. 15 GB) typically breaks down as:
- **Majority (~80%)**: Actual data — message content, session metadata, indexes. This is real data growth, not bloat.
- **Significant fraction (~15-20%)**: FTS trigram indexes. Rebuildable, not unique.
- **Small remainder (~1-3%)**: SQLite page fragmentation. Reclaimable via VACUUM.

**Key takeaway**: The DB size is mostly legitimate data growth. VACUUM saves only a small fraction. Compaction is NOT worth the risk for marginal savings. If disk becomes constrained, use message retention (pruning old messages) instead.

## What Does NOT Work

- **`VACUUM INTO` with FTS tables present** — rebuilds FTS indexes during copy, output is same size as input
- **Row-by-row INSERT to copy** — extremely slow for large tables, FTS triggers make it worse
- **gzip the raw DB** — preserves all bloat, compression ratio gains are modest for binary SQLite data
- **VACUUM inside a container** — container overlay filesystem cannot hold a multi-GB temp file; must VACUUM on host first
- **`.backup` without dropping FTS first** — copies FTS indexes into the compact output

## What Works (Copy + Drop FTS + .backup)

```bash
cp /root/.hermes/state.db /root/.hermes/backups/state-tmp.db
sqlite3 /root/.hermes/backups/state-tmp.db "DROP TABLE IF EXISTS messages_fts; DROP TABLE IF EXISTS messages_fts_trigram;"
sqlite3 /root/.hermes/backups/state-tmp.db ".backup '/root/.hermes/backups/state-nofts.db'"
pigz -6 /root/.hermes/backups/state-nofts.db
rm -f /root/.hermes/backups/state-tmp.db
```

**Space budget**: `.backup` creates a full copy — peak space ≈ 2x the DB size. For a 15 GB DB, that's ~30 GB peak. Only attempt on disks with sufficient headroom.

**IMPORTANT**: Only use this for backup creation. Do NOT run against the live DB without explicit user confirmation — always work on a copy first.

## Verified Backup + Restore Pipeline

Full pipeline tested with a copy — live DB never touched:

1. Copy DB → drop FTS tables → compress with pigz
2. On restore: decompress, create FTS5 tables via Python, repopulate from messages
3. Run completeness (orphan check, row counts) and recall (FTS search, trigram prefix) tests

**Result**: All tests pass. FTS rebuild works correctly. Search and trigram both functional after restore.

## FTS Rebuild After Restore

FTS5 tables can be dropped and rebuilt at any time. The process:
1. Drop all FTS tables and their triggers
2. Recreate FTS5 virtual tables
3. Repopulate from messages (one INSERT per FTS table)
4. Verify row counts match

This is safe — FTS tables contain no unique data, only derived search indexes.

## Recommended Approach (NOT to be run autonomously)

1. Always work on a copy, never the live DB
2. No-FTS backup before any compaction attempt
3. VACUUM savings are marginal — not worth the risk on the live DB
4. If disk is constrained, prune old messages instead of compacting
