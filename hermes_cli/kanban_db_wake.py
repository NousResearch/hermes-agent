"""Wake ``scheduled`` tasks whose recorded wait condition resolved.

A worker parks its own card with ``kanban_schedule(wait_for_process=..., until=...)``; the wait
condition rides in the ``wait`` key of that ``scheduled`` event's payload, so the park and its
condition commit in one transaction and a later park (or an unblock) supersedes it without extra
bookkeeping. The dispatcher's reclaim phase calls :func:`wake_scheduled_waits` every tick and
reopens a card, without any model call, when the awaited background process recorded an exit
code, vanished without one, or the deadline passed.

Split out of ``hermes_cli.kanban_db``; origin-resident helpers are reached late-bound via ``_kb``.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from datetime import UTC, datetime
from typing import Optional

logger = logging.getLogger(__name__)


def process_wait(*, session_id: str, pid: int, output_log: str, until: int) -> dict:
    """Wait condition stored on the ``scheduled`` event. Call on this host while the process lives."""
    from hermes_cli import kanban_db as _kb
    from hermes_cli.kanban_db_dispatch import _process_fingerprint

    return {
        "kind": "process", "session_id": session_id, "pid": int(pid),
        "started_at": _process_fingerprint(int(pid)), "output_log": output_log,
        "host": _kb._host_prefix(), "until": int(until),
    }


def wake_scheduled_waits(conn: sqlite3.Connection) -> list[str]:
    """Reopen ``scheduled`` tasks whose process wait resolved; returns the woken task ids.
    One task's failure is logged and skips only that task."""
    now = int(time.time())
    woken: list[str] = []
    for row in _pending_waits(conn):
        try:
            if _resolve_one(conn, row, now):
                woken.append(row["task_id"])
        except Exception:
            logger.debug("kanban wake: scheduled wait check failed for task %s",
                         row["task_id"], exc_info=True)
    return woken


def _pending_waits(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Each ``scheduled`` task with its most recent ``scheduled`` event."""
    return conn.execute(
        "SELECT t.id AS task_id, e.id AS event_id, e.run_id AS run_id, e.payload AS payload "
        "FROM tasks t JOIN task_events e ON e.id = ("
        "  SELECT id FROM task_events WHERE task_id = t.id AND kind = 'scheduled' "
        "  ORDER BY id DESC LIMIT 1) "
        "WHERE t.status = 'scheduled'"
    ).fetchall()


def _resolve_one(conn: sqlite3.Connection, row: sqlite3.Row, now: int) -> bool:
    wait = (json.loads(row["payload"]) if row["payload"] else {}).get("wait")
    if not isinstance(wait, dict) or wait.get("kind") != "process":
        return False
    outcome = _process_outcome(wait, now)
    if outcome is None or _parking_worker_alive(conn, row["task_id"], row["run_id"]):
        return False
    return _wake(conn, row, wait, *outcome)


def _process_outcome(wait: dict, now: int) -> Optional[tuple[str, Optional[int]]]:
    """``("exited", code)``, ``("lost", None)``, ``("deadline", None)`` or None (keep waiting).
    The process itself is only observable on the host that parked it."""
    from hermes_cli import kanban_db as _kb
    from hermes_cli.kanban_db_dispatch import _worker_alive
    from tools.process_registry_output_log import recorded_exit_code

    if wait.get("host") == _kb._host_prefix():
        # Decided BEFORE the read: a shell that is gone has already written its exit status.
        alive = _worker_alive(wait["pid"], wait.get("started_at"))
        code = recorded_exit_code(wait["output_log"])
        if code is not None:
            return "exited", code
        if not alive:
            return "lost", None
    if now >= int(wait["until"]):
        return "deadline", None
    return None


def _parking_worker_alive(conn: sqlite3.Connection, task_id: str, run_id: Optional[int]) -> bool:
    """True while a worker of the parking run may still be running: waking now would put a
    second worker on the card. A recorded ``worker_result`` proves that run's worker finished."""
    from hermes_cli.kanban_db_dispatch import _worker_alive

    if run_id is None:
        return False
    if conn.execute(
        "SELECT 1 FROM task_events WHERE task_id = ? AND run_id = ? AND kind = 'worker_result' LIMIT 1",
        (task_id, run_id),
    ).fetchone():
        return False
    return any(_worker_alive(pid, started_at) for pid, started_at in _run_worker_pids(conn, task_id, run_id))


def _run_worker_pids(conn: sqlite3.Connection, task_id: str, run_id: int) -> list[tuple[int, object]]:
    """Every ``(pid, fingerprint)`` recorded for one run: the run row and its spawn/bind events."""
    pids: list[tuple[int, object]] = []
    run = conn.execute(
        "SELECT worker_pid, worker_started_at FROM task_runs WHERE id = ?", (run_id,),
    ).fetchone()
    if run and run["worker_pid"]:
        pids.append((int(run["worker_pid"]), run["worker_started_at"]))
    for ev in conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND run_id = ? "
        "AND kind IN ('spawned', 'worker_bound')", (task_id, run_id),
    ):
        payload = json.loads(ev["payload"]) if ev["payload"] else {}
        if isinstance(payload, dict) and payload.get("pid"):
            pids.append((int(payload["pid"]), payload.get("started_at")))
    return pids


def _wake(conn: sqlite3.Connection, row: sqlite3.Row, wait: dict, outcome: str,
          exit_code: Optional[int]) -> bool:
    """Record why the wait ended (comment + ``wait_resolved``), then re-gate via ``unblock_task``.
    The record is written once per park, so a tick that died before the unblock just retries it."""
    from hermes_cli import kanban_db as _kb

    task_id, event_id = row["task_id"], row["event_id"]
    with _kb.write_txn(conn):
        latest = conn.execute(
            "SELECT t.status, (SELECT MAX(id) FROM task_events WHERE task_id = t.id AND kind = 'scheduled') "
            "AS event_id FROM tasks t WHERE t.id = ?", (task_id,),
        ).fetchone()
        if not latest or latest["status"] != "scheduled" or latest["event_id"] != event_id:
            return False
        recorded = conn.execute(
            "SELECT 1 FROM task_events WHERE task_id = ? AND kind = 'wait_resolved' AND id > ? LIMIT 1",
            (task_id, event_id),
        ).fetchone()
        if not recorded:
            _kb.add_comment(conn, task_id, "dispatcher", _comment(wait, outcome, exit_code))
            _kb._append_event(
                conn, task_id, "wait_resolved",
                {"kind": "process", "outcome": outcome, "exit_code": exit_code,
                 "session_id": wait["session_id"]},
                run_id=row["run_id"],
            )
    return _kb.unblock_task(conn, task_id)


def _comment(wait: dict, outcome: str, exit_code: Optional[int]) -> str:
    """Fixed English text naming the outcome and the log path; never the process output."""
    sid, log = wait["session_id"], wait["output_log"]
    if outcome == "exited":
        return f"Scheduled wait ended: process {sid} exited with code {exit_code}. Output log: {log}"
    if outcome == "lost":
        return (f"Scheduled wait ended: process {sid} is gone and recorded no exit code "
                f"(killed, or the host restarted). Output log: {log}")
    deadline = datetime.fromtimestamp(int(wait["until"]), UTC).isoformat()
    return (f"Scheduled wait ended: the deadline {deadline} passed and process {sid} "
            f"(pid {wait['pid']}) is still running. It was not stopped.")
