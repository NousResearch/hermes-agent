#!/usr/bin/env python3
"""Kanban task notifier — polls all boards for ready+assigned tasks and
outputs structured Discord messages for new ones.

no_agent=True cron script: stdout is delivered verbatim to Discord.
Empty stdout = silent (no message posted).
Deduplicates via state file at ~/.hermes/data/kanban-notifier-state.json
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
# P13 isolation: HERMES_HOME parameterises the hermes root.
_HERMES_HOME = Path(os.environ.get('HERMES_HOME', str(Path.home() / '.hermes')))
KANBAN_BASE = _HERMES_HOME / 'kanban'
BOARDS_DIR = KANBAN_BASE / 'boards'
DEFAULT_DB = KANBAN_BASE / 'kanban.db'  # back-compat: default board lives here
STATE_FILE = _HERMES_HOME / 'data' / 'kanban-notifier-state.json'
WRITE_LOCK = KANBAN_BASE / 'kanban-write.lock'

# P13 dry-run: --dry-run suppresses Discord delivery (no state save, no
# stdout message). Read paths (board scan, fetch_ready_tasks) run so the
# notifier still reports what it WOULD post, but nothing is persisted or
# printed for Discord delivery.
_DRY_RUN = '--dry-run' in sys.argv

# ── Helpers ────────────────────────────────────────────────────────────────


def discover_boards() -> list[tuple[str, Path]]:
    """Return list of (slug, db_path) pairs to scan.

    Mirrors the resolution used by `hermes kanban ...` (see
    `hermes_cli/kanban.py:2556` and `hermes_cli/kanban_db.py:kanban_db_path`):

    - The ``default`` board lives at ``<root>/kanban.db`` (back-compat path).
    - All other boards live at ``<root>/kanban/boards/<slug>/kanban.db``.

    Entries whose DB does not exist on disk are skipped — the ``default``
    DB is created on demand by the kanban CLI and may not exist on a fresh
    install or after a partial migration. Entries that share the same
    underlying path (alias boards) are deduped.
    """
    seen: set[Path] = set()
    result: list[tuple[str, Path]] = []

    def _add(slug: str, path: Path) -> None:
        try:
            resolved = path.resolve()
        except OSError:
            return
        if resolved in seen:
            return
        seen.add(resolved)
        result.append((slug, path))

    # Default board first (matches `hermes kanban` board ordering).
    if DEFAULT_DB.is_file():
        _add('default', DEFAULT_DB)

    if BOARDS_DIR.is_dir():
        for entry in sorted(BOARDS_DIR.iterdir()):
            if not entry.is_dir():
                continue
            db = entry / 'kanban.db'
            if db.is_file():
                _add(entry.name, db)

    return result


def load_state() -> dict[str, list[str]]:
    """Load dedup state: {board_slug: [task_id, ...]}."""
    if STATE_FILE.is_file():
        try:
            raw = STATE_FILE.read_text(encoding='utf-8').strip()
            if raw:
                return json.loads(raw)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def save_state(state: dict[str, list[str]]) -> None:
    """Persist dedup state atomically."""
    if _DRY_RUN:
        return  # dry-run: never persist dedup state
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix('.tmp')
    tmp.write_text(json.dumps(state, sort_keys=True), encoding='utf-8')
    tmp.replace(STATE_FILE)


def is_write_locked() -> bool:
    """Check if the global kanban write lock exists. Read-only check."""
    return WRITE_LOCK.is_file()


def fetch_ready_tasks(board: str, db_path: Path) -> list[dict]:
    """Query a single board for ready+assigned tasks. Returns list of dicts.

    ``db_path`` is the resolved path to the board's kanban.db (see
    :func:`discover_boards`). Empty or zero-byte files are skipped — the
    kanban CLI creates the default board's DB on demand, and on a fresh
    install the placeholder file may be 0 bytes before the first task is
    created.
    """
    if not db_path.is_file():
        return []
    if db_path.stat().st_size == 0:
        return []

    conn = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
    conn.execute("PRAGMA wal_autocheckpoint=0")
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT id, title, body, assignee, priority
            FROM tasks
            WHERE status = 'ready'
              AND assignee IS NOT NULL
              AND assignee != ''
              AND assignee != 'default'
            ORDER BY id
            """,
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def format_message(task: dict, board: str) -> str:
    """Format a single task as a structured Discord message block."""
    tid = task['id']
    assignee = task['assignee']
    title = task['title']
    body = task.get('body') or ''
    priority = task.get('priority', 0)

    # First 200 chars of body, stripped of excess whitespace
    brief = ' '.join(body.split())[:200].rstrip()
    if len(body) > 200:
        brief += '…'

    link = f'hermes kanban --board {board} show {tid}'

    return (
        f'📋 Task {tid} assigned to @{assignee}\n'
        f'Board: {board}\n'
        f'Title: {title}\n'
        f'Brief: {brief}\n'
        f'Priority: {priority}\n'
        f'Link: {link}'
    )


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    # Check write lock — if held, skip (read-only script respects it)
    if is_write_locked():
        # Silent exit — no output means no Discord message
        sys.exit(0)

    boards = discover_boards()
    if not boards:
        sys.exit(0)

    state = load_state()
    new_messages: list[str] = []

    for board, db_path in boards:
        try:
            tasks = fetch_ready_tasks(board, db_path)
        except sqlite3.DatabaseError as exc:
            # A single corrupt/migrating board must not take down the whole
            # notifier — log to stderr and continue with the remaining boards.
            # This is the silent-failure pattern that previously surfaced as
            # `code 1` Tracebacks on cron start.
            print(
                f'[kanban-task-notifier] skipping {board}: {type(exc).__name__}: {exc}',
                file=sys.stderr,
            )
            continue

        known = set(state.get(board, []))

        for task in tasks:
            tid = task['id']
            if tid not in known:
                new_messages.append(format_message(task, board))
                known.add(tid)

        # Update state for this board
        state[board] = sorted(known)

    if new_messages:
        if _DRY_RUN:
            # dry-run: report what WOULD be posted but do not deliver.
            print(f"[kanban-task-notifier] DRY-RUN: would post {len(new_messages)} message(s); state not saved")
            return
        save_state(state)
        print('\n---\n'.join(new_messages))
    # else: silent — no output


if __name__ == '__main__':
    main()
