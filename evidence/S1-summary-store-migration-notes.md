# S1 Summary Store Migration Notes

- Added a new `session_summaries` SQLite schema at version 1 for Hermes and OpenClaw source stores.
- The schema records summary JSON/text, schema version, record version, turn/hash markers, identity scope, parent summary key, status, and last error.
- Store constructors enable WAL, `busy_timeout`, and corrupt database recovery before writes.
- CAS writes require the caller's expected version to match the current record; stale writes return without overwriting.
- Repeated writes with the same `summary_key` and `last_input_hash` are idempotent and do not increment `version`.
- This stage does not connect the store to prompt building, recall, retain, or summary generation.
