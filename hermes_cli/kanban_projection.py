"""Canonical public projection of persisted Kanban task state."""

from __future__ import annotations

from dataclasses import asdict
import time
from typing import Any

from hermes_cli import kanban_db


def operational_status(task: kanban_db.Task, *, now: float | None = None) -> str:
    """Return the runtime-visible state without changing the persisted lane."""
    status = task.status
    if status == "running":
        heartbeat_fresh = bool(
            task.last_heartbeat_at
            and (time.time() if now is None else now) - int(task.last_heartbeat_at)
            <= kanban_db.DEFAULT_CLAIM_HEARTBEAT_MAX_STALE_SECONDS
        )
        from hermes_cli.kanban_db_dispatch import _worker_alive

        worker_live = bool(
            task.worker_pid
            and task.worker_started_at
            and _worker_alive(task.worker_pid, task.worker_started_at)
        )
        return "running" if heartbeat_fresh and worker_live else "recovering"
    if status == "todo":
        return "dependency-wait"
    if status in {"ready", "scheduled"}:
        return "queued"
    if status in {"blocked", "triage"}:
        return "failure"
    return status


def project_task(task: kanban_db.Task) -> dict[str, Any]:
    """Serialize a task with persisted identity plus canonical operational state."""
    projected = asdict(task)
    projected["skills"] = list(task.skills or [])
    projected["operational_status"] = operational_status(task)
    # Explicit assignments keep these contract fields present for legacy rows.
    projected["creator_task_id"] = task.creator_task_id
    projected["root_task_id"] = task.root_task_id
    return projected