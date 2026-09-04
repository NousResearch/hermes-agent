"""Profile-local durable handoff for cron delivery through live gateway adapters.

A restart-safe cron worker executes outside the gateway cgroup.  It cannot own
relay/E2EE adapter objects, so it queues the final send here.  A gateway claims
each row at most once.  If that gateway dies after claiming, the outcome is
marked unknown and never retried: losing a delivery is safer than duplicating a
possibly-completed send.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

from agent.redact import redact_sensitive_text
from cron.executions import _owner_is_live, _process_start_time
from hermes_cli.sqlite_util import add_column_if_missing
from hermes_constants import get_hermes_home
from hermes_time import now as _hermes_now

logger = logging.getLogger(__name__)

DELIVERY_DB: Optional[Path] = None
_PROCESS_ID = uuid.uuid4().hex
_lock = threading.RLock()
_ACTIVE_DELIVERIES: set[str] = set()
_TERMINAL = ("delivered", "failed", "unknown")
MAX_TERMINAL_DELIVERIES = 1000
DEFAULT_DELIVERY_WAIT_TIMEOUT_SECONDS = 300.0


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
            """INSERT OR IGNORE INTO delivery_tombstones
               (execution_id, terminal_status, finished_at)
               SELECT execution_id, status, finished_at FROM deliveries
               WHERE status IN ('delivered','failed','unknown')
               ORDER BY finished_at, created_at, execution_id
               LIMIT ?""",
            (excess,),
        )
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
            from hermes_state_wal import apply_wal_with_fallback

            conn.execute("PRAGMA busy_timeout=5000")
            apply_wal_with_fallback(conn, db_label="cron/deliveries.db")
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
            conn.execute(
                """CREATE TABLE IF NOT EXISTS delivery_tombstones (
                     execution_id TEXT PRIMARY KEY,
                     terminal_status TEXT NOT NULL CHECK(terminal_status IN
                       ('delivered','failed','unknown')),
                     finished_at TEXT
                   )"""
            )
            add_column_if_missing(
                conn, "deliveries", "for_failure",
                "for_failure INTEGER NOT NULL DEFAULT 0",
            )
            # Pruning is done explicitly by the paths that create terminal
            # rows (_finish / recover_abandoned / _terminalize_wait_timeout);
            # read-only polls must not pay for a full-table UPDATE + COUNT.
            with conn:
                yield conn
        finally:
            conn.close()


def enqueue(
    execution_id: str,
    job: dict,
    content: str,
    *,
    for_failure: bool = False,
) -> dict:
    """Persist one idempotent delivery request before the worker waits."""
    with _transaction() as conn:
        tombstone = conn.execute(
            "SELECT terminal_status, finished_at FROM delivery_tombstones "
            "WHERE execution_id=?",
            (str(execution_id),),
        ).fetchone()
        if tombstone is not None:
            return {
                "execution_id": str(execution_id),
                "status": tombstone["terminal_status"],
                "finished_at": tombstone["finished_at"],
            }
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
        if row is not None:
            return dict(row)
        tombstone = conn.execute(
            "SELECT execution_id, terminal_status, finished_at "
            "FROM delivery_tombstones WHERE execution_id=?",
            (str(execution_id),),
        ).fetchone()
    if tombstone is None:
        return None
    return {
        "execution_id": tombstone["execution_id"],
        "status": tombstone["terminal_status"],
        "finished_at": tombstone["finished_at"],
        "error": None,
    }


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
    safe_error = (
        redact_sensitive_text(str(error), force=True, redact_url_credentials=True)
        if error
        else None
    )
    with _transaction() as conn:
        cur = conn.execute(
            """UPDATE deliveries SET status=?, finished_at=?, error=?
               WHERE execution_id=? AND status='delivering'
                 AND owner_process_id=? AND owner_pid=?""",
            (
                status,
                _hermes_now().isoformat(),
                safe_error,
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


def _terminalize_wait_timeout(execution_id: str) -> str:
    """Fence a delivery whose worker can no longer wait for confirmation.

    A row still ``pending`` was provably never attempted, so it is left queued
    for whichever gateway comes up next (a restart that includes an update can
    easily exceed the worker's wait budget).  That is a deferral, not a
    failure: report success so the job is not recorded ``delivery_failed`` for
    a message the drain will still send.  Only a row caught mid-send is
    uncertain and gets fenced ``unknown``.
    """
    now = _hermes_now().isoformat()
    uncertain_error = (
        "timed out while gateway delivery was in progress; outcome is unknown and "
        "was not retried"
    )
    with _transaction() as conn:
        row = conn.execute(
            "SELECT status FROM deliveries WHERE execution_id=?",
            (str(execution_id),),
        ).fetchone()
        if row is not None and row["status"] == "pending":
            logger.warning(
                "Cron delivery %s: no live gateway within the wait budget; "
                "left queued for the next gateway",
                execution_id,
            )
            return ""
        conn.execute(
            """UPDATE deliveries SET status='unknown', finished_at=?, error=?
               WHERE execution_id=? AND status='delivering'""",
            (now, uncertain_error, str(execution_id)),
        )
        row = conn.execute(
            "SELECT status, error FROM deliveries WHERE execution_id=?",
            (str(execution_id),),
        ).fetchone()
        _prune_terminal_unlocked(conn)
    if row is None:
        return "timed out waiting for live gateway delivery"
    if row["status"] == "delivered":
        return ""
    return str(row["error"] or f"delivery {row['status']}")


def enqueue_and_wait(
    execution_id: str,
    job: dict,
    content: str,
    *,
    for_failure: bool = False,
    timeout: Optional[float] = None,
) -> Optional[str]:
    """Queue delivery and wait for a gateway's terminal at-most-once outcome."""
    queued = enqueue(execution_id, job, content, for_failure=for_failure)
    if queued["status"] in _TERMINAL:
        return None if queued["status"] == "delivered" else str(
            queued.get("error") or f"delivery {queued['status']}"
        )
    wait_timeout = (
        DEFAULT_DELIVERY_WAIT_TIMEOUT_SECONDS if timeout is None else max(0.0, timeout)
    )
    deadline = time.monotonic() + wait_timeout
    while time.monotonic() < deadline:
        row = get_status(execution_id)
        if row and row["status"] in _TERMINAL:
            return None if row["status"] == "delivered" else str(
                row.get("error") or f"delivery {row['status']}"
            )
        time.sleep(1.0)
    return _terminalize_wait_timeout(execution_id) or None
