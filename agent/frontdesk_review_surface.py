"""Explicit review/import helpers for durable frontdesk tasks.

This module is intentionally library-level only.  It does not register gateway,
Telegram, CLI, TUI, or natural-language commands; callers must invoke these
helpers directly and pass the durable store path they want to inspect.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from agent.task_registry import FRONTDESK_DONE_PRESENTED, FRONTDESK_REVIEW_PASSED

__all__ = [
    "get_durable_frontdesk_task",
    "list_durable_frontdesk_tasks",
    "present_durable_frontdesk_task",
    "record_durable_frontdesk_discard",
    "record_durable_frontdesk_import",
]


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False))


def _record_dict(record: Any | None) -> dict[str, Any] | None:
    if record is None:
        return None
    return _json_safe(asdict(record))


def _artifact_pointer(record: Any) -> dict[str, Any]:
    payload = {
        "id": record.id,
        "job_id": record.job_id,
        "path": record.path,
        "type": record.artifact_type,
        "import_status": record.import_status,
    }
    if record.checksum is not None:
        payload["checksum"] = record.checksum
    if record.size is not None:
        payload["size"] = record.size
    return payload


def _latest(records: list[Any], *, kind: str | None = None) -> Any | None:
    filtered = [record for record in records if kind is None or record.kind == kind]
    if not filtered:
        return None
    return max(filtered, key=lambda record: (record.created_at, record.updated_at, record.id))


def _store_path(path: str | os.PathLike[str] | None) -> Path:
    if path is not None:
        return Path(path)
    from agent.frontdesk_live import frontdesk_durable_store_path

    return frontdesk_durable_store_path()


def _task_payload(store: Any, task: Any, *, include_events: bool = False) -> dict[str, Any]:
    from agent.frontdesk_store import JOB_REVIEWER, JOB_WORKER

    jobs = store.list_jobs(task_id=task.id)
    latest_worker = _latest(jobs, kind=JOB_WORKER)
    latest_reviewer = _latest(jobs, kind=JOB_REVIEWER)
    artifacts = store.list_artifacts(task_id=task.id)
    review_result = latest_reviewer.result if latest_reviewer is not None else None
    payload: dict[str, Any] = {
        "task": _record_dict(task),
        "state": task.state,
        "latest_jobs": {
            "worker": _record_dict(latest_worker),
            "reviewer": _record_dict(latest_reviewer),
        },
        "review_result": _json_safe(review_result) if review_result is not None else None,
        "artifact_pointers": [_artifact_pointer(artifact) for artifact in artifacts],
        "presentable": task.state == FRONTDESK_REVIEW_PASSED,
        "presented": task.state == FRONTDESK_DONE_PRESENTED,
        "importable": task.state in {FRONTDESK_REVIEW_PASSED, FRONTDESK_DONE_PRESENTED},
    }
    if include_events:
        payload["events"] = [_record_dict(event) for event in store.list_events(task_id=task.id)]
    return _json_safe(payload)


def list_durable_frontdesk_tasks(
    *,
    path: str | os.PathLike[str] | None = None,
    session_key: str | None = None,
) -> dict[str, Any]:
    """List durable frontdesk tasks with review/job/artifact pointers."""
    from agent.frontdesk_store import FrontdeskStore

    db_path = _store_path(path)
    store = FrontdeskStore(db_path)
    try:
        tasks = store.list_tasks(session_key=session_key)
        return {
            "path": str(db_path),
            "session_key": session_key,
            "tasks": [_task_payload(store, task) for task in tasks],
        }
    finally:
        store.close()


def get_durable_frontdesk_task(
    task_id: str,
    *,
    path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Read one durable frontdesk task detail by id."""
    from agent.frontdesk_store import FrontdeskStore

    db_path = _store_path(path)
    store = FrontdeskStore(db_path)
    try:
        task = store.get_task(task_id)
        if task is None:
            raise KeyError(f"unknown frontdesk task id: {task_id!r}")
        payload = _task_payload(store, task, include_events=True)
        payload["path"] = str(db_path)
        return payload
    finally:
        store.close()


def present_durable_frontdesk_task(
    task_id: str,
    *,
    path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Mark a review-passed durable task as presented."""
    from agent.frontdesk_store import FrontdeskStore

    db_path = _store_path(path)
    store = FrontdeskStore(db_path)
    try:
        task, already_presented = store.mark_done_presented_with_status(task_id)
        payload = _task_payload(store, task, include_events=True)
        payload.update(
            {
                "path": str(db_path),
                "presented": True,
                "already_presented": already_presented,
            }
        )
        return _json_safe(payload)
    finally:
        store.close()


def record_durable_frontdesk_import(
    task_id: str,
    *,
    path: str | os.PathLike[str] | None = None,
    artifact_ids: Iterable[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Record an explicit non-applying import decision for reviewed artifacts."""
    from agent.frontdesk_store import ARTIFACT_IMPORT_REQUESTED, FrontdeskStore

    db_path = _store_path(path)
    store = FrontdeskStore(db_path)
    try:
        artifacts = store.record_import_decision(
            task_id,
            artifact_ids=artifact_ids,
            metadata=metadata,
        )
        task = store.get_task(task_id)
        if task is None:
            raise KeyError(f"unknown frontdesk task id: {task_id!r}")
        payload = _task_payload(store, task, include_events=True)
        payload.update(
            {
                "path": str(db_path),
                "import_decision": {
                    "status": ARTIFACT_IMPORT_REQUESTED,
                    "applied": False,
                    "artifact_ids": [artifact.id for artifact in artifacts],
                },
            }
        )
        return _json_safe(payload)
    finally:
        store.close()


def record_durable_frontdesk_discard(
    task_id: str,
    *,
    path: str | os.PathLike[str] | None = None,
    artifact_ids: Iterable[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Record an explicit non-destructive discard decision for artifacts."""
    from agent.frontdesk_store import ARTIFACT_DISCARDED, FrontdeskStore

    db_path = _store_path(path)
    store = FrontdeskStore(db_path)
    try:
        artifacts = store.record_discard_decision(
            task_id,
            artifact_ids=artifact_ids,
            metadata=metadata,
        )
        task = store.get_task(task_id)
        if task is None:
            raise KeyError(f"unknown frontdesk task id: {task_id!r}")
        payload = _task_payload(store, task, include_events=True)
        payload.update(
            {
                "path": str(db_path),
                "discard_decision": {
                    "status": ARTIFACT_DISCARDED,
                    "deleted": False,
                    "artifact_ids": [artifact.id for artifact in artifacts],
                },
            }
        )
        return _json_safe(payload)
    finally:
        store.close()
