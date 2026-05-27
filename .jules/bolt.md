# Performance Learnings (Bolt Workflow)

## Kanban Task Links Optimization
- **Optimization**: Batch inserting `task_links` via `executemany` instead of single looping `INSERT OR IGNORE` commands on `sqlite3.Connection`.
- **Learnings**: SQLite batch execution provides significant gains (12-36%) on bulk inserts. We must evaluate generator conversion carefully — avoiding exhausting `Iterable` objects like `parents` when they need to be reused later on in the same scope (e.g. inside `_append_event`). Converting to `list` first guarantees the optimization works correctly and prevents bugs.
