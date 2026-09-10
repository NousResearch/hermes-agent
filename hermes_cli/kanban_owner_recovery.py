"""Read-only canonical admission fences for dispatcher reclaim paths."""
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import psutil


def owner_reclaim_paused(conn, task_id):
    """Missing/ambiguous owner receipts never authorize another attempt."""
    marker = conn.execute(
        "SELECT e.payload FROM task_events e JOIN tasks t ON t.current_run_id=e.run_id "
        "WHERE t.id=? AND e.task_id=t.id AND e.kind='owner_admitted' ORDER BY e.id DESC LIMIT 1",
        (task_id,)).fetchone()
    if marker is None:
        return False
    try:
        binding = json.loads(marker['payload'])
        with closing(sqlite3.connect(Path(binding['db']).as_uri() + '?mode=ro', uri=True)) as owner:
            row = owner.execute('SELECT status,outcome FROM session_admissions WHERE target_session_id=? AND request_id=?',
                                (binding['session_id'], binding['request_id'])).fetchone()
        if row is not None and row[0] == 'terminal':
            # Discard acknowledges uncertainty, not permission to repeat effects.
            return row[1] == 'interrupted'
        if row is None or row[0] in {'queued', 'unknown'}:
            return True
        return psutil.Process(binding['pid']).create_time() != binding['birth']
    except (OSError, ValueError, KeyError, sqlite3.Error):
        return True
    except psutil.Error:
        return True
