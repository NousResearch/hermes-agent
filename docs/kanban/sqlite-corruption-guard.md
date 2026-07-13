# SQLite corruption guard

Hermes now quarantines a corrupt kanban SQLite database once, records a durable
incident marker next to the DB, and fails fast on repeat opens during the
cooldown window instead of repeatedly probing or generating new backups.

## What happens on first corrupt open

When `_guard_existing_db_is_healthy()` detects a real SQLite open or integrity
failure, it now:

1. Backs up the database using the existing content-addressed backup flow.
2. Writes `<db>.corrupt.marker.json` beside the database.
3. Raises `KanbanDbCorruptError` with the backup path, marker path, and
   operator guidance.
4. Logs the classified failure mode plus the next-step guidance.

The marker currently carries:

- `db_path`
- `failure_mode`
- `index_name`
- `reason`
- `backup_path`
- `first_seen_at`
- `last_seen_at`
- `suppress_until`
- `suppressed_count`

## Cooldown behavior

The marker suppresses fresh-open rechecks for 5 minutes. During that window a
new process opening the same DB:

- reuses the existing backup reference,
- increments `suppressed_count`,
- skips the repeat SQLite probe,
- skips creating another backup, and
- raises a suppressed `KanbanDbCorruptError` that points operators at the
  existing marker and backup.

If the cooldown has expired, Hermes re-probes the DB. If the bytes have changed
and the DB is still corrupt, Hermes records a fresh backup while preserving the
original `first_seen_at` lineage in the marker.

If the DB becomes healthy again, Hermes clears the marker automatically.

## Failure modes

The guard classifies corruption into two operator-facing modes:

- `sqlite_open_failure`: generic SQLite open or integrity failures. The board
  stays quarantined until it is repaired or restored.
- `index_inconsistency`: integrity output that matches `wrong # of entries in
  index <name>`. The guidance includes a copy-only `REINDEX <name>; PRAGMA
  integrity_check` workflow for reviewed, intentionally quiesced boards.

## Verification coverage

The reland packet adds regression coverage for:

- active markers suppressing repeat probes and duplicate backups,
- expired markers allowing a fresh reprobe and backup rotation, and
- index-only corruption surfacing marker-aware guidance in the raised error and
  logs.
