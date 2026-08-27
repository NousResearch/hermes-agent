# State database and FTS recovery

`state.db` stores two different data classes:

- `sessions` and `messages` are the canonical transcript.
- `messages_fts*` tables and their sync triggers are derived search indexes.

The derived indexes may be detached temporarily. They must not turn a live
message write or search into an unbounded full-transcript rebuild.

## Live behavior when FTS is corrupt

If an FTS write or search reports the corruption error class, `SessionDB`:

1. records the durable `fts_stale` marker;
2. removes the FTS sync triggers in the same transaction;
3. retries canonical writes without the derived-index sinks; and
4. serves searches from canonical rows through the `LIKE` fallback.

The failing live operation never runs `FTS5('rebuild')`. Existing recovery
ownership remains unchanged: a later `SessionDB` open may rebuild under the
cross-process admission lock and foreign-holder guard. If that guarded rebuild
cannot run, FTS remains detached, canonical writes stay available, and
`hermes doctor` reports the explicit repair command.

## Explicit repair

Stop every process that can open the profile database before repairing it.
Keep them stopped for the complete repair and verification window.

```bash
hermes gateway stop
HERMES_HOME="$HOME/.hermes" hermes sessions repair --check-only
HERMES_HOME="$HOME/.hermes" hermes sessions repair
```

`sessions repair` creates a SQLite backup by default and performs structural
work through the repository's guarded snapshot-and-promotion path. Do not copy
`state.db`, `state.db-wal`, and `state.db-shm` independently with `cp`; those
files are one live SQLite image.

After repair, verify the health probe, stale marker, trigger set, and canonical
row counts before restarting the gateway:

```bash
HERMES_HOME="$HOME/.hermes" hermes sessions repair --check-only
sqlite3 "$HOME/.hermes/state.db" \
  "SELECT key, value FROM state_meta WHERE key = 'fts_stale';"
sqlite3 "$HOME/.hermes/state.db" \
  "SELECT type, name FROM sqlite_master WHERE name IN
   ('messages_fts_insert','messages_fts_update','messages_fts_delete')
   ORDER BY name;"
sqlite3 "$HOME/.hermes/state.db" \
  "SELECT 'sessions', COUNT(*) FROM sessions
   UNION ALL SELECT 'messages', COUNT(*) FROM messages;"
```

The marker query should return no row, the expected FTS triggers should be
present, and canonical row counts must not decrease. If repair fails, preserve
both the live database and the reported backup; never delete canonical rows to
make a derived-index error disappear.
