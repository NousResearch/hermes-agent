"""Shared runtime helpers for durable background job startup."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Callable


def new_background_task_id() -> str:
    """Return a new durable background job task ID."""
    return f"bg_{datetime.now().strftime('%H%M%S')}_{os.urandom(3).hex()}"


def start_background_job(
    *,
    store: Any,
    launch_worker: Callable[[str], dict[str, Any]],
    prompt: str,
    source: Any,
    conversation_history: list[dict[str, Any]] | None = None,
    context_prompt: str = "",
    session_key: str = "",
    job_kind: str = "manual",
    worker_name: str = "",
    preloaded_skills: list[str] | None = None,
    admin_user_ids: list[str] | None = None,
    is_admin_user: bool | None = None,
    task_id_factory: Callable[[], str] = new_background_task_id,
    logger=None,
) -> str:
    """Persist and launch one durable managed background job."""
    task_id = task_id_factory()
    store.create_job(
        task_id=task_id,
        prompt=prompt,
        source=source,
        session_key=session_key,
        job_kind=job_kind,
        worker_name=worker_name,
        preloaded_skills=list(preloaded_skills or []),
        conversation_history=list(conversation_history or []),
        context_prompt=context_prompt,
        admin_user_ids=list(admin_user_ids or []),
        is_admin_user=is_admin_user,
    )
    try:
        metadata = launch_worker(task_id)
        store.update_job_launcher(task_id, metadata)
    except Exception as exc:
        if logger is not None:
            logger.exception("Failed to launch background worker for %s", task_id)
        store.mark_job_failed(task_id, error=str(exc))
    return task_id
