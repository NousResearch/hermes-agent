## Summary

Adds FTS5 corruption detection and auto-recovery to `state.db`. When FTS indexes become corrupt (malformed), Hermes now self-heals on startup instead of silently breaking `session_search` and all FTS-backed features.

Closes #33865.

## Background

Related reports: #5563, #30908, #23717, #30445 — all describe SQLite corruption in state.db or kanban.db caused by interrupted WAL checkpoints, concurrent process contention, or force-kills during active transactions.

## Changes

### `hermes_state.py`

- **`_init_schema()`**: Now catches `sqlite3.DatabaseError` (corrupt FTS) in addition to `sqlite3.OperationalError` (missing FTS). On either, drops and recreates FTS tables, then backfills from messages.

- **`_drop_fts()` / `_drop_fts_trigram()`**: Static helpers that cleanly drop FTS virtual tables and their triggers. Shared by `_init_schema()` and `rebuild_fts()`.

- **`rebuild_fts()`**: Public method that drops, recreates, and backfills both FTS indexes. Returns `(fts_count, trigram_count)`. Used by `hermes sessions repair` and `hermes doctor --fix`.

- **`fts_integrity_check()`**: Returns a dict comparing FTS rowcount against messages table. Detects both corruption (`DatabaseError`) and index drift (count mismatch).

- **`integrity_check()`**: Wraps `PRAGMA integrity_check`, returns list of issues.

### `hermes_cli/doctor.py`

The state.db check now:
1. Runs `PRAGMA integrity_check` (catches B-tree corruption, page errors)
2. Validates FTS rowcount matches messages table (catches index drift)
3. Reports specific errors instead of generic "has issues"
4. `hermes doctor --fix` can auto-rebuild corrupt FTS indexes

### `hermes_cli/main.py`

New `hermes sessions repair` subcommand:
- Runs integrity check + FTS health validation
- Auto-rebuilds corrupt FTS indexes
- `--check-only` flag for read-only diagnostics
- Handles both corruption (malformed) and drift (count mismatch)

## Testing

```bash
python -c "
from hermes_state import SessionDB
import tempfile, os
os.environ['HERMES_HOME'] = tempfile.mkdtemp()
db = SessionDB()
fts_count, tri_count = db.rebuild_fts()
msg_count = db.message_count()
assert fts_count == msg_count
assert tri_count == msg_count
fts = db.fts_integrity_check()
assert fts['fts_ok'] and fts['trigram_ok']
print('All assertions passed')
db.close()
"
```

## Breaking Changes

None. All changes are additive. Existing behavior preserved for healthy databases.

## Checklist

- [x] Bug fix (crash/data loss prevention)
- [x] Cross-platform (Windows + Linux + macOS — pure sqlite3, no platform-specific code)
- [x] No new dependencies
- [x] Backward compatible
- [x] Follows existing patterns (v11 migration FTS rebuild, `_reconcile_columns()` declarative approach)
