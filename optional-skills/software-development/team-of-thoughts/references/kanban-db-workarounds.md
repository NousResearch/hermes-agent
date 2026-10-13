# Kanban DB Workarounds

> SQLite queries for when the kanban CLI's LLM-backed commands (specify, decompose) fail with `BadRequestError`.

## DB location

```bash
~/.hermes/kanban/boards/<board-slug>/kanban.db
```

Find the active board slug with:
```bash
hermes kanban boards
```

## Common workarounds

### Move triage cards to todo (when `specify` fails)

```python
python3 -c "
import sqlite3
db = sqlite3.connect('$HOME/.hermes/kanban/boards/<board-slug>/kanban.db')
db.execute(\"UPDATE tasks SET status='todo' WHERE status='triage'\")
db.commit()
db.close()
"
```

### Check all cards and their status

```python
python3 -c "
import sqlite3
db = sqlite3.connect('$HOME/.hermes/kanban/boards/<board-slug>/kanban.db')
db.row_factory = sqlite3.Row
rows = db.execute('SELECT id, title, status, assignee FROM tasks ORDER BY id').fetchall()
for r in rows:
    print(f'{r[\"id\"]} | {r[\"status\"]:8s} | {r[\"assignee\"] or \"-\":12s} | {r[\"title\"][:70]}')
db.close()
"
```

### Promote a card via DB (when promote CLI fails)

```python
python3 -c "
import sqlite3
db = sqlite3.connect('$HOME/.hermes/kanban/boards/<board-slug>/kanban.db')
db.execute(\"UPDATE tasks SET status='ready' WHERE id='<card-id>'\")
db.commit()
db.close()
"
```

## Schema notes (relevant columns)

| Column | Type | Values |
|--------|------|--------|
| id | TEXT | `t_<hex>` — the task ID used in CLI commands |
| title | TEXT | Card title |
| status | TEXT | `triage`, `todo`, `ready`, `running`, `blocked`, `done`, `archived` |
| assignee | TEXT | Profile name or null |
| parent_id | TEXT | Null for root cards, `t_<hex>` for children |
| body | TEXT | Card description / opening post |
| created_at | INTEGER | Unix timestamp |
| completed_at | INTEGER | Unix timestamp or null |
| priority | INTEGER | 0-9 for tiebreaking |
| claim_lock | TEXT | Lock owner identifier or null |
| claim_expires | INTEGER | Unix timestamp or null |
| worker_pid | INTEGER | PID of running worker process or null |
| current_run_id | INTEGER | Active run ID or null |
| consecutive_failures | INTEGER | Failure count (reset to 0 on success) |
| last_failure_error | TEXT | Last error message or null |
| started_at | INTEGER | Unix timestamp or null |

## When to use these workarounds

- **`hermes kanban specify` fails** → `BadRequestError` means the LLM behind it errored. Use the triage→todo DB update.
- **`hermes kanban decompose` fails** → Same LLM error. Manually create child cards with `hermes kanban create --parent <id>` instead.
- **`hermes kanban promote` blocks on dependency** → Use `--force` flag first (preferred). Only use DB update when `--force` also fails.
- **Need bulk promote** → DB update is faster than 9 separate CLI calls for promoting multiple cards at once.

### Reset stuck `running` cards (after killing worker processes)

When you kill a kanban worker process (e.g. because it was using the wrong executor), the card stays in `running` status with stale lock/PID. There is no `hermes kanban release` command. Use this:

```python
python3 -c "
import sqlite3
db = sqlite3.connect('$HOME/.hermes/kanban/boards/<board-slug>/kanban.db')
for card_id in ['<card-1>', '<card-2>']:
    db.execute('''UPDATE tasks SET status='ready', claim_lock=NULL, claim_expires=NULL, worker_pid=NULL, current_run_id=NULL, started_at=NULL, consecutive_failures=0, last_failure_error=NULL WHERE id=?''', (card_id,))
    print(f'Reset {card_id} to ready')
db.commit()
db.close()
"
```

**When to use:** After `kill -9 <pid>` on a kanban worker, or when a card is stuck in `running` with no active process. Always verify with `ps aux | grep <pid>` that the process is truly dead before resetting.

Do NOT use DB workarounds for anything that has a working CLI equivalent (create, archive, comment). Reserve them strictly for LLM-backed commands that fail, stuck cards, or bulk operations.
