"""Profile-local durable handoff for cron delivery through live gateway adapters.

A restart-safe cron worker executes outside the gateway cgroup.  It cannot own
relay/E2EE adapter objects, so it queues the final send here.  A gateway claims
each row at most once.  If that gateway dies after claiming, the outcome is
marked unknown and never retried: losing a delivery is safer than duplicating a
possibly-completed send.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

from hermes_constants import get_hermes_home
from hermes_time import now as _hermes_now

DELIVERY_DB: Optional[Path] = None
_PROCESS_ID = uuid.uuid4().hex
_lock = threading.RLock()
_ACTIVE_DELIVERIES: set[str] = set()
_TERMINAL = ("delivered", "failed", "unknown")
MAX_TERMINAL_DELIVERIES = 1000


def _prune_terminal_unlocked(conn: sqlite3.Connection) -> None:
    """Redact terminal payloads and retain only bounded outcome metadata."""
    conn.execute(
        """UPDATE deliveries SET job_json='{}', content=''
           WHERE status IN ('delivered','failed','unknown')
             AND (job_json != '{}' OR content != '')"""
    )
    keep = max(0, int(MAX_TERMINAL_DELIVERIES))
    terminal_count = int(
        conn.execute(
            "SELECT COUNT(*) FROM deliveries "
            "WHERE status IN ('delivered','failed','unknown')"
        ).fetchone()[0]
    )
    excess = terminal_count - keep
    if excess > 0:
        conn.execute(
            """DELETE FROM deliveries WHERE execution_id IN (
                 SELECT execution_id FROM deliveries
                 WHERE status IN ('delivered','failed','unknown')
                 ORDER BY finished_at, created_at, execution_id
                 LIMIT ?
               )""",
            (excess,),
        )


def _path() -> Path:
    return DELIVERY_DB or (get_hermes_home().resolve() / "cron" / "deliveries.db")


@contextmanager
def _transaction() -> Iterator[sqlite3.Connection]:
    with _lock:
        path = _path()
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path, timeout=5)
        try:
            path.chmod(0o600)
        except OSError:
            pass
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA busy_timeout=5000")
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=FULL")
            conn.execute(
                """CREATE TABLE IF NOT EXISTS deliveries (
                     execution_id TEXT PRIMARY KEY,
                     job_json TEXT NOT NULL,
                     content TEXT NOT NULL,
                     for_failure INTEGER NOT NULL DEFAULT 0,
                     status TEXT NOT NULL CHECK(status IN
                       ('pending','delivering','delivered','failed','unknown')),
                     owner_process_id TEXT,
                     owner_pid INTEGER,
                     owner_started_at INTEGER,
                     created_at TEXT NOT NULL,
                     finished_at TEXT,
                     error TEXT
                   )"""
            )
            columns = {
                str(row[1]) for row in conn.execute("PRAGMA table_info(deliveries)")
            }
            if "for_failure" not in columns:
                conn.execute(
                    "ALTER TABLE deliveries "
                    "ADD COLUMN for_failure INTEGER NOT NULL DEFAULT 0"
                )
            with conn:
                _prune_terminal_unlocked(conn)
                yield conn
        finally:
            conn.close()


def _process_start_time(pid: int) -> Optional[int]:
    try:
        from gateway.status import get_process_start_time

        return get_process_start_time(pid)
    except Exception:
        return None


def _owner_is_live(pid: int, started_at: Optional[int]) -> bool:
    try:
        from gateway.status import _pid_exists

        if not _pid_exists(pid):
            return False
    except Exception:
        return True
    if started_at is None:
        return pid == os.getpid()
    return _process_start_time(pid) == started_at


def enqueue(
    execution_id: str,
    job: dict,
    content: str,
    *,
    for_failure: bool = False,
) -> dict:
    """Persist one idempotent delivery request before the worker waits."""
    with _transaction() as conn:
        conn.execute(
            """INSERT OR IGNORE INTO deliveries
               (execution_id, job_json, content, for_failure, status, created_at)
               VALUES (?, ?, ?, ?, 'pending', ?)""",
            (
                str(execution_id),
                json.dumps(job, ensure_ascii=False, sort_keys=True),
                str(content),
                int(bool(for_failure)),
                _hermes_now().isoformat(),
            ),
        )
        row = conn.execute(
            "SELECT * FROM deliveries WHERE execution_id=?", (str(execution_id),)
        ).fetchone()
    return dict(row)


def get_status(execution_id: str) -> Optional[dict]:
    with _transaction() as conn:
        row = conn.execute(
            "SELECT * FROM deliveries WHERE execution_id=?", (str(execution_id),)
        ).fetchone()
    return dict(row) if row is not None else None


def claim_next() -> Optional[dict]:
    """Atomically claim one pending send before touching the transport."""
    pid = os.getpid()
    started = _process_start_time(pid)
    with _transaction() as conn:
        row = conn.execute(
            "SELECT execution_id FROM deliveries WHERE status='pending' "
            "ORDER BY created_at, execution_id LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        cur = conn.execute(
            """UPDATE deliveries SET status='delivering', owner_process_id=?,
               owner_pid=?, owner_started_at=?
               WHERE execution_id=? AND status='pending'""",
            (_PROCESS_ID, pid, started, row["execution_id"]),
        )
        if cur.rowcount != 1:
            return None
        claimed = conn.execute(
            "SELECT * FROM deliveries WHERE execution_id=?", (row["execution_id"],)
        ).fetchone()
        _ACTIVE_DELIVERIES.add(row["execution_id"])
    result = dict(claimed)
    result["job"] = json.loads(result.pop("job_json"))
    return result


def _finish(execution_id: str, *, error: Optional[str]) -> bool:
    status = "failed" if error else "delivered"
    with _transaction() as conn:
        cur = conn.execute(
            """UPDATE deliveries SET status=?, finished_at=?, error=?
               WHERE execution_id=? AND status='delivering'
                 AND owner_process_id=? AND owner_pid=?""",
            (
                status,
                _hermes_now().isoformat(),
                error,
                execution_id,
                _PROCESS_ID,
                os.getpid(),
            ),
        )
        _prune_terminal_unlocked(conn)
    return cur.rowcount == 1


def recover_abandoned() -> int:
    """Fence dead delivery owners as unknown; never replay uncertain sends."""
    changed = 0
    with _transaction() as conn:
        rows = conn.execute(
            "SELECT execution_id, owner_process_id, owner_pid, owner_started_at "
            "FROM deliveries WHERE status='delivering'"
        ).fetchall()
        for row in rows:
            same_process = row["owner_process_id"] == _PROCESS_ID
            if same_process:
                with _lock:
                    if row["execution_id"] in _ACTIVE_DELIVERIES:
                        continue
            elif _owner_is_live(int(row["owner_pid"]), row["owner_started_at"]):
                continue
            error = (
                "Gateway finished delivery but could not persist its outcome; "
                "send was not retried."
                if same_process
                else "Gateway exited during delivery; send outcome is unknown and was not retried."
            )
            cur = conn.execute(
                """UPDATE deliveries SET status='unknown', finished_at=?, error=?
                   WHERE execution_id=? AND status='delivering'""",
                (
                    _hermes_now().isoformat(),
                    error,
                    row["execution_id"],
                ),
            )
            changed += cur.rowcount
        _prune_terminal_unlocked(conn)
    return changed


def drain(
    send: Callable[[dict, str, bool], Optional[str]], *, limit: int = 20
) -> int:
    """Deliver pending rows through *send*, terminalizing every claimed row."""
    recover_abandoned()
    processed = 0
    for _ in range(max(0, limit)):
        row = claim_next()
        if row is None:
            break
        with _lock:
            _ACTIVE_DELIVERIES.add(row["execution_id"])
        try:
            try:
                error = send(
                    row["job"], row["content"], bool(row["for_failure"])
                )
            except BaseException as exc:
                error = f"{type(exc).__name__}: {exc}"
            _finish(row["execution_id"], error=error)
        finally:
            with _lock:
                _ACTIVE_DELIVERIES.discard(row["execution_id"])
        processed += 1
    return processed


def enqueue_and_wait(
    execution_id: str,
    job: dict,
    content: str,
    *,
    for_failure: bool = False,
    timeout: Optional[float] = None,
) -> Optional[str]:
    """Queue delivery and wait for a gateway's terminal at-most-once outcome."""
    enqueue(execution_id, job, content, for_failure=for_failure)
    deadline = None if timeout is None else time.monotonic() + timeout
    while deadline is None or time.monotonic() < deadline:
        row = get_status(execution_id)
        if row and row["status"] in _TERMINAL:
            return None if row["status"] == "delivered" else str(
                row.get("error") or f"delivery {row['status']}"
            )
        time.sleep(0.25)
    return "timed out waiting for live gateway delivery; request remains pending"
