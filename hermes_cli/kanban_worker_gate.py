"""Keep a newly launched Kanban worker inert until its claim is durable.

The dispatcher can start an OS process before its PID-registration transaction
commits. No worker command may execute until a fresh SQLite read sees that
registration; a rollback or dispatcher crash must not release the worker.
"""

import os
import sqlite3
import sys
import time
from pathlib import Path


_WAIT_SECONDS = 30


def wait_for_registration(db_path: str, task_id: str, run_id: int,
                          claim_lock: str, *, timeout: float = _WAIT_SECONDS) -> bool:
    deadline = time.monotonic() + timeout
    uri = Path(db_path).resolve().as_uri() + "?mode=ro"
    while time.monotonic() < deadline:
        try:
            # A new read transaction on each probe never holds a stale snapshot.
            with sqlite3.connect(uri, uri=True, timeout=0.2) as conn:
                row = conn.execute(
                    "SELECT status, current_run_id, claim_lock, worker_pid "
                    "FROM tasks WHERE id = ?", (task_id,),
                ).fetchone()
        except sqlite3.Error:
            row = None
        if row is not None:
            if row[0] != "running" or row[1] != run_id or row[2] != claim_lock:
                return False
            if row[3] is not None:
                return True
        time.sleep(0.05)
    return False


def main() -> int:
    task_id = os.environ.get("HERMES_KANBAN_TASK", "")
    claim_lock = os.environ.get("HERMES_KANBAN_CLAIM_LOCK", "")
    db_path = os.environ.get("HERMES_KANBAN_DB", "")
    try:
        run_id = int(os.environ.get("HERMES_KANBAN_RUN_ID", ""))
    except ValueError:
        run_id = 0
    if not (task_id and claim_lock and db_path and run_id and len(sys.argv) > 1):
        print("Kanban worker refused: missing claim identity", file=sys.stderr)
        return 1
    if not wait_for_registration(db_path, task_id, run_id, claim_lock):
        print("Kanban worker refused: claim not durably registered", file=sys.stderr)
        return 1
    os.execvpe(sys.argv[1], sys.argv[1:], os.environ)
    return 1


if __name__ == "__main__":
    sys.exit(main())
