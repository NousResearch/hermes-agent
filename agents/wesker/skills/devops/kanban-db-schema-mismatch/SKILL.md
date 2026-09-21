---
name: kanban-db-schema-mismatch
description: Diagnose and fix sqlite3 OperationalError 'no such column' errors in kanban.db caused by LLM agents querying columns that don't exist in the tasks table.
version: 1.0.0
---

# Kanban DB Schema Mismatch

**Trigger:** Cron agents or workers report `sqlite3.OperationalError: no such column: <column_name>` for queries against `/home/kensei/.hermes/kanban.db`.

**Root cause:** The kanban.db `tasks` table has a specific schema defined by kernel/scheduler code. LLM-driven cron agents (especially `kensei-heartbeat-audit` in Kanban health rotation) generate inline Python queries that reference columns they *assume* exist (e.g. `updated_at`, `status_reason`, `claim_expires_at`, `tasker_kind`, `count`). These columns often don't exist — the kanban schema uses event-based tracking instead of mutable timestamp columns.

## Diagnosis

1. Check the actual schema:
   ```python
   python3 -c "import sqlite3; conn=sqlite3.connect('/home/kensei/.hermes/kanban.db'); cur=conn.execute('PRAGMA table_info(tasks)'); [print(f'{c[1]:30s} {c[2]:15s}') for c in cur.fetchall()]; conn.close()"
   ```

2. Find which columns are being queried:
   ```bash
   grep "no such column" /home/kensei/.hermes/logs/errors.log | sort -u
   grep "no such column" /home/kensei/.hermes/logs/agent.log | sort -u
   ```

## Fix

Add the missing columns as nullable INTEGER/TEXT columns via ALTER TABLE — non-destructive, zero data risk:

```python
import sqlite3
conn = sqlite3.connect('/home/kensei/.hermes/kanban.db')
cur = conn.cursor()
# Add each missing column
for col, typ in [('updated_at', 'INTEGER'), ('status_reason', 'TEXT'), ('claim_expires_at', 'INTEGER'), ('tasker_kind', 'TEXT'), ('count', 'INTEGER')]:
    try:
        cur.execute(f'ALTER TABLE tasks ADD COLUMN {col} {typ}')
        conn.commit()
    except sqlite3.OperationalError as e:
        if 'duplicate column' not in str(e):
            raise
conn.close()
```

## Verification

Re-run one of the failing query patterns:

```python
import sqlite3
conn = sqlite3.connect('/home/kensei/.hermes/kanban.db')
cur = conn.execute("SELECT id, title, status, updated_at FROM tasks LIMIT 3")
print(cur.fetchall())
conn.close()
```

## Pitfalls

- Do NOT drop/recreate tables. ALTER TABLE ADD COLUMN is safe.
- The `backlog_items` table already has `updated_at` — only `tasks` is affected.
- These are nullable columns with no default. Existing rows will return NULL for them, which is fine — agents just need the column to exist to avoid crashing.
- There's no trigger to auto-update `updated_at` on write. If kernel-level support is needed, file a separate task.
