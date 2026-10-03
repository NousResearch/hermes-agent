"""Read-only, event-ledger-authoritative review handoffs for coordinators."""

from __future__ import annotations

import json
import sqlite3
from typing import Any


def _reject_constant(token: str) -> None:
    raise ValueError(f"non-JSON constant: {token}")


def _object(raw: str | None) -> dict | None:
    if raw is None:
        return None
    try:
        value = json.loads(raw, parse_constant=_reject_constant)
    except (TypeError, ValueError, UnicodeError):
        return None
    return value if isinstance(value, dict) else None


def latest_review_handoff(
    conn: sqlite3.Connection, task_id: str, before_run_id: int | None = None,
) -> dict[str, Any] | None:
    """Read the newest review request for this task, or ``None`` if not provable.

    ``before_run_id`` is an exclusive attempt bound. The latest event is always
    authoritative: malformed or duplicated provenance must not make an older
    handoff appear current. Ordinary requests use the closed run's full summary
    and structured metadata, checked against the event's summary prefix. A
    fenced recovery request instead carries ``recovery`` (task_id, run_id,
    profile), ``handoff_summary`` and ``handoff_metadata`` in the event payload;
    those fields, not mutable task fields, are the recovered handoff's source.
    The result has task_id, run_id, event_id, summary, metadata, implementer,
    reviewer; it never changes the connection's state.
    """
    if before_run_id is not None and (type(before_run_id) is not int or before_run_id < 1):
        raise ValueError("before_run_id must be a positive integer")
    sql = "SELECT id, run_id, payload FROM task_events WHERE task_id = ? AND kind = 'review_requested'"
    params: list[Any] = [task_id]
    if before_run_id is not None:
        sql += " AND run_id < ?"
        params.append(before_run_id)
    event = conn.execute(sql + " ORDER BY id DESC LIMIT 1", params).fetchone()
    if event is None:
        return None
    payload = _object(event["payload"])
    if payload is None or not {"summary", "implementer", "reviewer"} <= payload.keys():
        return None
    implementer, reviewer = payload["implementer"], payload["reviewer"]
    if not all(value is None or isinstance(value, str) and value.strip()
               for value in (implementer, reviewer)):
        return None
    run_id = event["run_id"]
    if run_id is None:
        if payload["summary"] is not None or any(
            key in payload for key in ("recovery", "handoff_summary", "handoff_metadata", "metadata")
        ):
            return None
        if conn.execute("SELECT 1 FROM tasks WHERE id = ?", (task_id,)).fetchone() is None:
            return None
        return {
            "task_id": task_id, "run_id": None, "event_id": event["id"],
            "summary": None, "metadata": {}, "implementer": implementer, "reviewer": reviewer,
        }
    run = conn.execute(
        "SELECT profile, outcome, ended_at, summary, metadata FROM task_runs "
        "WHERE id = ? AND task_id = ?", (run_id, task_id),
    ).fetchone()
    if run is None or run["outcome"] != "review_requested" or run["ended_at"] is None:
        return None
    duplicates = conn.execute(
        "SELECT COUNT(*) FROM task_events WHERE task_id = ? AND run_id = ? "
        "AND kind = 'review_requested'", (task_id, run_id),
    ).fetchone()[0]
    if duplicates != 1:
        return None

    if "recovery" in payload:
        receipt = payload["recovery"]
        if not isinstance(receipt, dict) or set(receipt) != {"task_id", "run_id", "profile"} or (
            receipt["task_id"] != task_id or type(receipt["run_id"]) is not int or
            receipt["run_id"] != run_id or receipt["profile"] != run["profile"] or
            implementer != run["profile"] or not isinstance(run["profile"], str) or
            not run["profile"].strip()
        ):
            return None
        summary = payload.get("handoff_summary")
        metadata = payload.get("handoff_metadata")
        if ("handoff_summary" not in payload or "handoff_metadata" not in payload or
            (summary is not None and not isinstance(summary, str)) or
            (metadata is not None and not isinstance(metadata, dict))):
            return None
        metadata = metadata if metadata is not None else {}
        if run["summary"] != summary:
            return None
        stored_metadata = {} if run["metadata"] is None else _object(run["metadata"])
        if stored_metadata is None or stored_metadata != metadata:
            return None
    else:
        if "handoff_summary" in payload or "handoff_metadata" in payload:
            return None
        summary = run["summary"]
        metadata = {} if run["metadata"] is None else _object(run["metadata"])
        if metadata is None:
            return None
    from hermes_cli.kanban_db import _first_line

    if payload["summary"] != (_first_line(summary, 400) or None):
        return None
    return {
        "task_id": task_id, "run_id": run_id, "event_id": event["id"],
        "summary": summary, "metadata": metadata,
        "implementer": implementer, "reviewer": reviewer,
    }
