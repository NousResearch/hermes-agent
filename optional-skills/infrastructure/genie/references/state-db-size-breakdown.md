# State DB Size Breakdown — Measured (May 2026)

**Method**: Full copy of state.db → drop FTS → VACUUM → rebuild FTS inside container. Live DB never touched.

**Note**: Specific sizes below are from May 2026 with ~139K messages. These will change as the DB grows. The *ratios* are the durable insight.

## Results

| Component | Approx Size | % of total | Notes |
|-----------|------------|-----------|-------|
| Core data + indexes | ~12 GB | ~80% | Messages content, sessions, indexes |
| FTS trigram indexes | ~2.7 GB | ~18% | Rebuildable, no unique data |
| Fragmentation/overhead | ~0.3 GB | ~2% | Reclaimable via VACUUM |

## Key Insight

The DB size is **mostly legitimate data growth**, not bloat. The earlier "11.5 GB bloat" estimate was wrong — it confused FTS index size with fragmentation. VACUUM on the live DB would only save ~0.5 GB. Compaction is not worth the risk for marginal savings.

## Procedure (Tested)

1. `cp state.db state-copy.db`
2. `sqlite3 state-copy.db "DROP TABLE IF EXISTS messages_fts; DROP TABLE IF EXISTS messages_fts_trigram;"`
3. `sqlite3 state-copy.db ".backup 'state-vacuumed.db'"`
4. Copy into container
5. Create FTS5 tables via Python (container sqlite3 CLI lacks FTS5)
6. Repopulate and verify

## FTS Rebuild

- ~8 minutes for 138K messages in a container
- Adds ~2.7 GB overhead
- All triggers and indexes recreated correctly
- Search and trigram both verified working post-rebuild
