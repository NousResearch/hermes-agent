"""`ctx.schedule(period="cron")`: a plugin handler dispatched by the existing cron scheduler.

The job is a normal ``jobs.json`` record with the mutually exclusive payload kind
``plugin_task`` — every wall-clock invariant (timezones, DST, catch-up windows, quota holds,
claims, ``executions.db`` history) stays with the cron scheduler. The scheduler thread calls the
registered handler in-process; a missing handler fails the fire visibly.

Design: website/docs/developer-guide/scheduled-tasks.md § Plugin wall-clock tasks.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Optional

from scheduler.runner import BoundedCallbackRunner
from scheduler.tasks import TaskContext

logger = logging.getLogger("hermes.scheduler")

_REGISTRY: dict[str, tuple[Callable[[TaskContext], Any], float, str, str]] = {}
_LOCK = threading.Lock()


def _handler_timeout() -> float:
    from scheduler.engine import _scheduler_config

    config = _scheduler_config()
    raw = config.get("turn_step")
    section = raw if isinstance(raw, dict) else {}
    return max(1, min(60_000, int(section.get("handler_timeout_ms", 50) or 50))) / 1000.0


class CronTaskHandle:
    def __init__(self, job_id: str, name: str, release: Callable[[], None]) -> None:
        self.job_id = job_id
        self.name = name
        self._release = release

    def cancel(self) -> None:
        self._release()


def _pause(job_id: str, reason: str) -> None:
    try:
        from cron.jobs import pause_job

        pause_job(job_id, reason=reason)
    except Exception:
        logger.warning("scheduler: could not pause plugin task job %s", job_id, exc_info=True)


def schedule_cron_task(
    ctx: Any, *, name: str, cron: Optional[str], handler: Callable[[TaskContext], Any],
    timeout_ms: Optional[int] = None,
) -> CronTaskHandle:
    if not cron:
        raise ValueError("period='cron' requires cron='<5-field expression>'")
    if not name.startswith(f"{ctx.plugin_id}."):
        raise ValueError("task name must be prefixed with its plugin id")
    if not callable(handler):
        raise ValueError("handler must be callable")
    from cron.jobs import list_jobs, parse_schedule, resume_job, update_job
    from cron.scheduler import create_job_with_scheduler_registration

    parsed = parse_schedule(cron)
    if parsed.get("kind") != "cron":
        raise ValueError("period='cron' requires a 5-field cron expression")
    plugin_task = {"plugin_id": ctx.plugin_id, "task_name": name}
    existing = [
        job for job in list_jobs(include_disabled=True)
        if (job.get("plugin_task") or {}).get("plugin_id") == ctx.plugin_id
        and (job.get("plugin_task") or {}).get("task_name") == name
    ]
    if existing:
        # Registration runs on every boot/hot reload: reuse the job instead of minting a
        # duplicate live schedule (N restarts must not become N fires).
        job = existing[0]
        job_id = str(job["id"])
        if job.get("paused_reason") == "plugin unloaded":
            try:
                job = resume_job(job_id) or job
            except Exception:
                logger.warning("scheduler: could not resume plugin task job %s", job_id, exc_info=True)
        if (job.get("schedule") or {}).get("expr") != parsed.get("expr"):
            try:
                job = update_job(job_id, {"schedule": cron}) or job
                from cron.scheduler import _notify_provider_jobs_changed

                _notify_provider_jobs_changed()
            except Exception:
                logger.warning("scheduler: could not update plugin task job %s", job_id, exc_info=True)
    else:
        job = create_job_with_scheduler_registration(
            prompt="", schedule=cron, name=name, deliver="local", plugin_task=plugin_task,
        )
    timeout = (max(1, min(60_000, int(timeout_ms))) / 1000.0) if timeout_ms else _handler_timeout()
    with _LOCK:
        _REGISTRY[job["id"]] = (handler, timeout, ctx.plugin_id, name)

    def _release() -> None:
        with _LOCK:
            _REGISTRY.pop(job["id"], None)
        _pause(job["id"], "plugin unloaded")

    ctx._track("scheduled_task", name, _release)
    return CronTaskHandle(job["id"], name, _release)


def run_plugin_task_job(job: dict[str, Any], execution_id: str) -> None:
    """Execute one plugin cron fire and book it in jobs.json + executions.db."""
    from cron.executions import finish_execution
    from cron.jobs import mark_job_run

    meta = job.get("plugin_task") or {}
    plugin_id = str(meta.get("plugin_id") or "")
    task_name = str(meta.get("task_name") or job.get("name") or job.get("id"))
    with _LOCK:
        entry = _REGISTRY.get(str(job.get("id")))
    ok = False
    error: Optional[str] = None
    if entry is None:
        error = (f"plugin task {plugin_id}.{task_name} is unavailable "
                 "(plugin missing, disabled, or unloaded)")
    else:
        handler, timeout, plugin_id, task_name = entry
        context = TaskContext(
            session_id="", profile_name="", agent_name="", phase="post", now=time.time(),
        )
        outcome = BoundedCallbackRunner(timeout_s=timeout).run(
            task_key=(plugin_id, task_name), session_id=None, seq=execution_id,
            fn=lambda: handler(context), timeout_s=timeout,
        )
        ok = outcome.status == "ok"
        if not ok:
            error = outcome.error or outcome.reason or outcome.status
    try:
        mark_job_run(str(job["id"]), ok, error)
    except Exception:
        logger.warning("scheduler: plugin task mark_job_run failed", exc_info=True)
    try:
        finish_execution(execution_id, success=ok, error=error)
    except Exception:
        logger.debug("scheduler: plugin task execution finish failed", exc_info=True)
