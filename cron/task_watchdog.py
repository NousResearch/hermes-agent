"""Cron task heartbeat registry for long-running/background task visibility."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from hermes_time import now as _hermes_now


def _sanitize_json(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _sanitize_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_sanitize_json(v) for v in value]
    try:
        json.dumps(value)
        return value
    except Exception:
        return str(value)


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp", prefix=".task_heartbeat_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def load_tasks(path: str | Path) -> dict:
    file_path = Path(path)
    if not file_path.exists():
        return {"tasks": [], "updated_at": None}
    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
    except Exception:
        return {"tasks": [], "updated_at": None}
    tasks = data.get("tasks")
    if not isinstance(tasks, list):
        tasks = []
    return {"tasks": tasks, "updated_at": data.get("updated_at")}


def _save_tasks(path: str | Path, tasks: list[dict]) -> dict:
    payload = {
        "tasks": _sanitize_json(tasks),
        "updated_at": _hermes_now().isoformat(),
    }
    _atomic_write_json(Path(path), payload)
    return payload


def _upsert_task(path: str | Path, task_id: str, updates: dict) -> dict:
    payload = load_tasks(path)
    tasks = payload["tasks"]
    now_iso = _hermes_now().isoformat()
    sanitized_updates = _sanitize_json(updates)
    for task in tasks:
        if task.get("task_id") == task_id:
            task.update(sanitized_updates)
            task["updated_at"] = now_iso
            return _save_tasks(path, tasks)

    new_task = {"task_id": task_id, "updated_at": now_iso}
    new_task.update(sanitized_updates)
    tasks.append(new_task)
    return _save_tasks(path, tasks)


def register_task(
    *,
    path: str | Path,
    task_id: str,
    job_id: str,
    job_name: str,
    session_id: str,
    delivery_target: dict | None,
    started_at: str | None = None,
    status: str = "running",
    activity_summary: dict | None = None,
    last_progress_note: str | None = None,
    last_activity_desc: str | None = None,
) -> dict:
    started = started_at or _hermes_now().isoformat()
    return _upsert_task(
        path,
        task_id,
        {
            "job_id": job_id,
            "job_name": job_name,
            "session_id": session_id,
            "status": status,
            "started_at": started,
            "last_progress_at": started,
            "last_progress_note": last_progress_note,
            "last_activity_desc": last_activity_desc,
            "blocker_reason": None,
            "user_action_needed": None,
            "delivery_target": delivery_target,
            "activity_summary": activity_summary or {},
        },
    )


def update_task_activity(
    *,
    path: str | Path,
    task_id: str,
    activity_summary: dict | None = None,
    last_progress_note: str | None = None,
    last_activity_desc: str | None = None,
) -> dict:
    updates: dict[str, Any] = {"activity_summary": activity_summary or {}}
    if last_progress_note:
        updates["last_progress_note"] = last_progress_note
        updates["last_progress_at"] = _hermes_now().isoformat()
    if last_activity_desc is not None:
        updates["last_activity_desc"] = last_activity_desc
    return _upsert_task(path, task_id, updates)


def mark_task_completed(
    *,
    path: str | Path,
    task_id: str,
    activity_summary: dict | None = None,
    last_progress_note: str | None = None,
) -> dict:
    return _upsert_task(
        path,
        task_id,
        {
            "status": "completed",
            "activity_summary": activity_summary or {},
            "last_progress_note": last_progress_note,
            "last_progress_at": _hermes_now().isoformat(),
        },
    )


def mark_task_failed(
    *,
    path: str | Path,
    task_id: str,
    blocker_reason: str,
    user_action_needed: str | None = None,
    activity_summary: dict | None = None,
    last_progress_note: str | None = None,
) -> dict:
    return _upsert_task(
        path,
        task_id,
        {
            "status": "failed",
            "blocker_reason": blocker_reason,
            "user_action_needed": user_action_needed,
            "activity_summary": activity_summary or {},
            "last_progress_note": last_progress_note,
            "last_progress_at": _hermes_now().isoformat(),
        },
    )


def mark_task_blocked(
    *,
    path: str | Path,
    task_id: str,
    blocker_reason: str,
    user_action_needed: str | None = None,
    activity_summary: dict | None = None,
    last_progress_note: str | None = None,
) -> dict:
    return _upsert_task(
        path,
        task_id,
        {
            "status": "waiting_external",
            "blocker_reason": blocker_reason,
            "user_action_needed": user_action_needed or blocker_reason,
            "activity_summary": activity_summary or {},
            "last_progress_note": last_progress_note,
            "last_progress_at": _hermes_now().isoformat(),
        },
    )
