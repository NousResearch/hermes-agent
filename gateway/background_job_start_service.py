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
    ensure_background_job_state: Callable[[], None],
    store: Any,
    refresh_cache: Callable[[dict[str, Any]], None],
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
    ensure_background_job_state()
    task_id = task_id_factory()
    record = store.create_job(
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
    refresh_cache(record)
    try:
        metadata = launch_worker(task_id)
        record = store.update_job_launcher(task_id, metadata) or record
        refresh_cache(record)
    except Exception as exc:
        if logger is not None:
            logger.exception("Failed to launch background worker for %s", task_id)
        record = store.mark_job_failed(task_id, error=str(exc)) or record
        refresh_cache(record)
    return task_id
