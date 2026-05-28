# state.db FTS corruption goes undetected — no integrity check, no repair path

## Summary

The `messages_fts` and `messages_fts_trigram` FTS5 indexes in `state.db` can become corrupt ("database disk image is malformed"), silently breaking `session_search`, `/resume`, `/history`, and any feature backed by FTS. There is currently:

1. **No integrity check on startup** — `_init_schema()` creates/reconciles tables but never runs `PRAGMA integrity_check`
2. **No FTS health validation** — `hermes doctor` only checks `SELECT COUNT(*) FROM sessions`; it doesn't validate FTS indexes match the messages table
3. **No repair command** — `hermes sessions` has `list`, `prune`, `stats`, `rename`, `export`, `delete`, `browse` — but no `repair`
4. **No auto-recovery** — When FTS is corrupt, `_init_schema()` catches `sqlite3.OperationalError` (table missing) but not `sqlite3.DatabaseError` (table corrupt/malformed)

## Root Cause

FTS5 virtual tables and their triggers insert into the FTS index as part of the message INSERT transaction. If that transaction is interrupted mid-commit (force-kill, WAL checkpoint failure, power loss), the FTS and messages tables desync. The `_try_wal_checkpoint()` runs every 50 writes but is best-effort with bare `except Exception: pass` — corrupt FTS during checkpoint is silently swallowed.

## Reproduction

1. Run Hermes with heavy session activity (gateway + CLI + worktree agents sharing state.db)
2. Force-kill the process (`taskkill /F /IM hermes.exe` on Windows, or SIGKILL on Linux)
3. Restart — `session_search` returns "database disk image is malformed"

## Related Issues

- #5563 — broader state.db corruption report that includes page-level corruption
- #30908 — same root cause pattern for kanban.db (WAL checkpoint interruption)
- #23717 — documents the "hot-update death spiral" causing state.db corruption
- #30445 — multi-gateway concurrent SQLite access causing corruption

## Impact

- `session_search` (the only way for Hermes to recall cross-session context) is completely broken
- `/resume`, `/title`, `/history`, `/branch` all fail
- The only recovery path is manual: stop gateway, export JSON, rebuild from scratch
- Users lose session history if they don't know the manual recovery procedure
