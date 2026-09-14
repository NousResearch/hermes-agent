"""Cooperative dispatcher shutdown, scoped to an owned worker run."""
from __future__ import annotations
import os
import secrets
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional
from utils import atomic_json_write
from hermes_cli.kanban_db import kanban_home, _retry_status_for_run, _end_run, _append_event
from hermes_cli.kanban_db_connect import write_txn
_SHUTDOWN_DRAIN_MARKER_ENV = "HERMES_KANBAN_DRAIN_MARKER"
_SHUTDOWN_DRAIN_MARKER_NAME = ".shutdown_drain.json"

def shutdown_drain_marker_path() -> Path:
    """Return this dispatcher's cooperative shutdown marker path.

    The dispatcher process chooses a unique path for its lifetime and passes
    it to worker children through an internal environment variable. A fresh
    dispatcher therefore cannot accidentally inherit an old shutdown request,
    while workers already in flight retain the exact marker they were given.
    """
    override = os.environ.get(_SHUTDOWN_DRAIN_MARKER_ENV, "").strip()
    if override:
        return Path(override).expanduser()
    return kanban_home() / "kanban" / _SHUTDOWN_DRAIN_MARKER_NAME


def prepare_shutdown_drain_marker() -> Path:
    """Allocate the marker workers spawned by this dispatcher will observe."""
    existing = os.environ.get(_SHUTDOWN_DRAIN_MARKER_ENV, "").strip()
    if existing:
        return Path(existing).expanduser()
    marker = kanban_home() / "kanban" / (
        f".shutdown_drain.{os.getpid()}.{secrets.token_hex(8)}.json"
    )
    os.environ[_SHUTDOWN_DRAIN_MARKER_ENV] = str(marker)
    return marker


def request_shutdown_drain(*, reason: str = "dispatcher shutdown") -> dict[str, Any]:
    """Ask this dispatcher's workers to pause after their current turn."""
    marker = prepare_shutdown_drain_marker()
    payload = {
        "action": "pause-at-turn-boundary",
        "requested_at": time.time(),
        "reason": str(reason or "dispatcher shutdown"),
        "marker": str(marker),
    }
    atomic_json_write(marker, payload)
    return payload


def shutdown_drain_requested() -> bool:
    """Return True when this worker's dispatcher requested cooperative pause.

    Presence is fail-safe: a partially-written or malformed marker still
    requests quiescence. The marker is process-instance scoped by
    :func:`prepare_shutdown_drain_marker`, so it is not a durable global pause.
    """
    marker = shutdown_drain_marker_path()
    try:
        return marker.exists()
    except OSError:
        return True


def pause_task(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    expected_run_id: Optional[int] = None,
    claimer: Optional[str] = None,
    reason: str = "dispatcher shutdown",
) -> bool:
    """Cooperatively pause one owned running task and release its claim.

    This is deliberately a run outcome, not a new card status. The task
    returns to the lane from which its run was claimed (``ready`` or
    ``review``), so the normal dispatcher can resume it later. The CAS checks
    bind the transition to the worker's run and claim, preventing an old
    worker from pausing a successor that already reclaimed the card.
    """
    with write_txn(conn):
        row = conn.execute(
            "SELECT status, current_run_id, claim_lock FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        if row is None or row["status"] != "running":
            return False
        run_id = row["current_run_id"]
        if run_id is None:
            return False
        if expected_run_id is not None and int(run_id) != int(expected_run_id):
            return False
        if claimer is not None and row["claim_lock"] != claimer:
            return False

        retry_status = _retry_status_for_run(conn, task_id, int(run_id))
        where = (
            "WHERE id = ? AND status = 'running' AND current_run_id = ?"
        )
        params: tuple[Any, ...] = (retry_status, task_id, int(run_id))
        if claimer is not None:
            where += " AND claim_lock = ?"
            params += (claimer,)
        cur = conn.execute(
            "UPDATE tasks SET status = ?, claim_lock = NULL, "
            "claim_expires = NULL, worker_pid = NULL " + where,
            params,
        )
        if cur.rowcount != 1:
            return False

        closed_run_id = _end_run(
            conn,
            task_id,
            outcome="paused",
            status="paused",
            error=None,
            metadata={"reason": str(reason or "dispatcher shutdown")},
        )
        _append_event(
            conn,
            task_id,
            "paused",
            {
                "reason": str(reason or "dispatcher shutdown"),
                "run_id": closed_run_id,
                "retry_status": retry_status,
            },
            run_id=closed_run_id,
        )
        return True
