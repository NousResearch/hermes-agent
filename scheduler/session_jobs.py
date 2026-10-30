"""Load, claim, and fire agent-scheduled turn/step jobs.

A job never runs in its origin session: it executes in a fresh ``cron_*`` session through the
same local job body cron uses (prompt, skills, agent construction, teardown), then delivers to
the job's ``deliver`` target. The origin session's clock is never advanced by its own job, so no
self-trigger loop exists.

Design: website/docs/developer-guide/scheduled-tasks.md § Agent-scheduled turn/step jobs.
"""

from __future__ import annotations

import contextvars
import logging
import sys
import threading
from typing import Any, Optional

logger = logging.getLogger("hermes.scheduler")


def _with_db(fn):
    from hermes_state_registry import acquire, release_or_close

    db = acquire()
    try:
        return fn(db)
    finally:
        release_or_close(db)


def load_jobs(session_id: str) -> list[dict[str, Any]]:
    return _with_db(lambda db: db.list_session_jobs(session_id, enabled_only=True))


def claim_job(job_id: str, expected_due: int) -> bool:
    return _with_db(lambda db: db.claim_session_job(job_id, expected_due=int(expected_due)))


def _origin_for(db: Any, origin_session_id: str) -> dict[str, Any]:
    """Best-effort delivery origin from the origin session row (``deliver="origin"``)."""
    try:
        row = db.get_session(origin_session_id) or {}
    except Exception:
        return {}
    return {
        "platform": row.get("created_source") or row.get("source") or "",
        "chat_id": row.get("chat_id") or "",
        "chat_name": row.get("display_name") or "",
        "thread_id": row.get("thread_id") or "",
        "user_id": row.get("user_id") or "",
        "scope_id": "",
    }


def _cron_shaped_job(db: Any, job: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": job["id"],
        "name": job.get("name") or f"session job {job['id']}",
        "prompt": job["prompt"],
        "skills": job.get("skills") or [],
        "deliver": job.get("deliver"),
        "failure_deliver": job.get("failure_deliver"),
        "origin": _origin_for(db, job.get("origin_session_id") or ""),
        "no_agent": False,
        "schedule": {"kind": "once"},
        "workdir": None,
        "enabled_toolsets": None,
    }


def _delivery_handles() -> tuple[Any, Any]:
    """Live gateway adapters/loop for the current profile; ``(None, None)`` off-gateway."""
    try:
        runner_ref = getattr(sys.modules.get("gateway.run"), "_gateway_runner_ref", None)
        runner = runner_ref() if callable(runner_ref) else None
        if runner is None:
            return None, None
        adapters = getattr(runner, "adapters", None)
        if hasattr(runner, "_adapters_for_profile"):
            from hermes_constants import get_hermes_home, profile_name_for_home

            adapters = runner._adapters_for_profile(profile_name_for_home(get_hermes_home()))
        return adapters, getattr(runner, "_gateway_loop", None)
    except Exception:
        logger.debug("scheduler: gateway delivery handles unavailable", exc_info=True)
        return None, None


def run_job(job: dict[str, Any], fired_seq: int) -> None:
    """Dispatch a claimed session job on a daemon worker (never blocks the turn)."""
    context = contextvars.copy_context()
    thread = threading.Thread(
        target=lambda: context.run(_execute, job, int(fired_seq)),
        name=f"hermes-tj-{job.get('id', 'job')}"[:40], daemon=True,
    )
    thread.start()


def _execute(job: dict[str, Any], fired_seq: int) -> None:
    from cron.scheduler import _deliver_result, _run_job_local, _teardown_cron_agent

    job_id = str(job["id"])
    origin_session_id = job.get("origin_session_id")
    run_id = 0
    started = __import__("time").time()
    try:
        run_id = _with_db(lambda db: db.record_session_job_run(
            job_id, fired_at_seq=fired_seq, status="running",
            origin_session_id=origin_session_id, started_at=started))
    except Exception:
        logger.debug("scheduler: session job run row failed", exc_info=True)
    try:
        cron_job = _with_db(lambda db: _cron_shaped_job(db, job))
        agents: list[Any] = []
        ok, _output, final, error = _run_job_local(cron_job, defer_agent_teardown=agents)
        delivery_error = None
        try:
            if final:
                adapters, loop = _delivery_handles()
                delivery_error = _deliver_result(
                    cron_job, final, adapters=adapters, loop=loop, for_failure=not ok)
            elif not ok:
                adapters, loop = _delivery_handles()
                delivery_error = _deliver_result(
                    cron_job, error or "scheduled job failed", adapters=adapters, loop=loop,
                    for_failure=True)
        finally:
            for agent in agents:
                _teardown_cron_agent(agent, job_id)
        status = "ok" if ok and not delivery_error else ("delivery_failed" if ok else "error")
        detail = delivery_error if ok else (error or delivery_error)
        _finish(run_id, status=status, error=detail, job_id=job_id)
    except Exception as exc:  # background failure must be visible in history, not fatal
        logger.warning("scheduler: session job %s failed: %s", job_id, exc, exc_info=True)
        _finish(run_id, status="error", error=f"{type(exc).__name__}: {exc}", job_id=job_id)


def _finish(run_id: int, *, status: str, error: Optional[str], job_id: str) -> None:
    if run_id:
        try:
            _with_db(lambda db: db.finish_session_job_run(run_id, status=status, error=error))
        except Exception:
            logger.debug("scheduler: session job run finish failed", exc_info=True)
    try:
        _with_db(lambda db: db.update_session_job(job_id, last_status=status))
    except Exception:
        logger.debug("scheduler: session job status write failed", exc_info=True)
