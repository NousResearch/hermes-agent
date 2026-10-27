"""Dispatcher-owned wake-up of scheduled cards, using the latest park event."""

from __future__ import annotations

import re
import sqlite3
import time
from datetime import datetime


CONDITION_RECHECK_SECONDS = 30 * 60
_MARKER = re.compile(r"\b(SCHEDULED_UNTIL|WAKE_ON)=")
_DEADLINE = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?"
    r"(?:Z|[+-]\d{2}:\d{2})"
)


def _wake_trigger(reason: object, parked_at: int, now: int) -> str | None:
    """A future deadline is a not-before gate; conditions are worker rechecks,
    never executable predicates or permission to proceed with external work."""
    text = reason if isinstance(reason, str) else ""
    markers = list(_MARKER.finditer(text))
    values = {}
    for index, marker in enumerate(markers):
        name = marker.group(1)
        if name in values:
            raise ValueError(f"duplicate {name}")
        end = markers[index + 1].start() if index + 1 < len(markers) else len(text)
        values[name] = text[marker.end():end].strip()
    if not values:
        raise ValueError("reason needs SCHEDULED_UNTIL=<timezone-aware ISO8601> or WAKE_ON=<condition>")
    if "WAKE_ON" in values and not values["WAKE_ON"]:
        raise ValueError("WAKE_ON needs a condition")
    if "SCHEDULED_UNTIL" in values:
        tokens = values["SCHEDULED_UNTIL"].split()
        token = tokens[0].rstrip(".,;") if tokens else ""
        if not _DEADLINE.fullmatch(token):
            raise ValueError("SCHEDULED_UNTIL needs a timezone-aware ISO8601 timestamp")
        deadline = datetime.fromisoformat(token).timestamp()
        return "deadline" if deadline <= now else None
    return "condition_recheck" if parked_at + CONDITION_RECHECK_SECONDS <= now else None


def _wake_one(conn: sqlite3.Connection, task_id: str, now: int) -> None:
    from hermes_cli import kanban_db as kb

    blocked_task = None
    with kb.write_txn(conn):
        # Read again under the writer lock: an operator may have re-scheduled
        # the card since the sweep enumerated it.
        task = conn.execute("SELECT status FROM tasks WHERE id = ?", (task_id,)).fetchone()
        if task is None or task["status"] != "scheduled":
            return
        event = conn.execute(
            "SELECT id, payload, created_at FROM task_events "
            "WHERE task_id = ? AND kind = 'scheduled' ORDER BY id DESC LIMIT 1", (task_id,),
        ).fetchone()
        payload = kb._json_dict(event["payload"]) if event else {}
        try:
            if event is None:
                raise ValueError("missing scheduled event")
            trigger = _wake_trigger(payload.get("reason"), event["created_at"], now)
        except (ValueError, OverflowError, OSError) as exc:
            error = f"Invalid scheduled wake: {exc}"
            kb._reclaim_dangling_run(conn, task_id, statuses=("scheduled",), now=now,
                                     note="invariant recovery on scheduled wake")
            conn.execute(
                "UPDATE tasks SET status = 'blocked', block_kind = 'needs_input', "
                "last_failure_error = ?, claim_lock = NULL, claim_expires = NULL, "
                "worker_pid = NULL, worker_started_at = NULL, current_run_id = NULL WHERE id = ?",
                (error, task_id),
            )
            kb._insert_comment(conn, task_id, "dispatcher", error, now)
            kb._append_event(conn, task_id, "blocked", {
                "kind": "needs_input", "reason": error, "source": "scheduled_wake",
                "resume_status": "ready",
            })
            kb._log.warning("kanban scheduled wake: %s: %s", task_id, error)
            blocked_task = kb.get_task(conn, task_id)
        else:
            if trigger is None:
                return
            kb._reclaim_dangling_run(conn, task_id, statuses=("scheduled",), now=now,
                                     note="invariant recovery on scheduled wake")
            status = kb._landing_status_after_parents(conn, task_id)
            conn.execute(
                "UPDATE tasks SET status = ?, claim_lock = NULL, claim_expires = NULL, "
                "worker_pid = NULL, worker_started_at = NULL, current_run_id = NULL WHERE id = ?",
                (status, task_id),
            )
            if trigger == "condition_recheck":
                kb._insert_comment(conn, task_id, "dispatcher",
                                   "WAKE_ON condition recheck is due. Verify the condition before proceeding; "
                                   "if it is still unmet, schedule again with WAKE_ON in the reason.", now)
            kb._append_event(conn, task_id, "scheduled_woke", {
                "status": status, "trigger": trigger, "scheduled_event_id": event["id"],
            })
    if blocked_task is not None:
        kb._fire_task_hook("kanban_task_blocked", blocked_task, task_id, None,
                           reason=blocked_task.last_failure_error)


def wake_scheduled(conn: sqlite3.Connection, *, dry_run: bool = False) -> None:
    """Sweep before spawn caps so a full queue cannot leave due cards parked.
    No schema migration or host cron: board-local events survive restarts."""
    if dry_run:
        return
    now = int(time.time())
    rows = conn.execute("SELECT id FROM tasks WHERE status = 'scheduled'").fetchall()
    for row in rows:
        _wake_one(conn, row["id"], now)
