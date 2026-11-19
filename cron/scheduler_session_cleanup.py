"""Owner-handle cleanup for cron fires, including gates that never construct an agent."""
import logging

logger = logging.getLogger(__name__)


def _end_suppressed_cron_session(db, job: dict, job_id: str, job_name: str, cron_session_id: str) -> None:
    """End the admission's row without acquiring, releasing or closing its lent store.

    End BEFORE cosmetic writes: a timeout disables the shared bounded proxy, so a stalled
    title must not prevent the lifecycle write. At most one abandoned worker uses this handle.
    Receipts and terminal admissions remain available for exact retry and lazy resume.
    """
    from cron.scheduler import (
        _BoundedCronSessionDB, _set_cron_session_title, _hermes_now, safe_strftime,
    )

    if db is None:
        return
    bounded = _BoundedCronSessionDB(db.sessions, job_id)
    try:
        bounded.end(cron_session_id, "cron_suppressed")
        title = " ".join(job_name.split())[:60].strip() or f"cron {job_id}"
        _set_cron_session_title(bounded, cron_session_id,
                               f"{title} · {safe_strftime(_hermes_now(), '%b %d %H:%M')}")
        # No filesystem validation during teardown: this is the admission's configured cwd,
        # not a directory we are about to execute in. A disconnected mount must not block end.
        workdir = (job.get("workdir") or "").strip()
        if workdir:
            bounded.update_cwd(cron_session_id, workdir)
    except (Exception, KeyboardInterrupt):
        logger.debug("Job '%s': suppressed session cleanup failed", job_id, exc_info=True)


def finish_cron_run(job, job_name, session_id, owner_execution, scope, session_db,
                    agent, worker_future, defer_agent_teardown):
    """One cleanup decision: a live worker owns teardown, otherwise finalize this fire now."""
    from cron.scheduler import _finalize_cron_session, _teardown_cron_agent
    from cron.scheduler_detached_worker import defer_teardown_to_running_worker

    job_id = job["id"]
    workdir = scope.workdir if scope is not None else None
    deferred = defer_teardown_to_running_worker(
        worker_future, session_db, agent, job_id, job_name, session_id, workdir=workdir)
    # Per-occurrence containment teardown (cron/containment.py): synchronously
    # stop and verify every scope the occurrence's driver registered, BEFORE the
    # session is finalized. When the conversation thread outlives run_job
    # (deferred teardown), the conversation is still live — attach teardown to
    # the future's completion instead of killing in-flight roles now. Runs on
    # every exit path, including exceptions and cancellation.
    execution_id = getattr(scope, "_containment_execution_id", "") if scope is not None else ""
    if execution_id:
        def _containment_teardown(execution_id: str = execution_id) -> None:
            try:
                from cron import containment as _containment
                _report = _containment.end(execution_id)
                if _report["leftover"]:
                    logger.error(
                        "Job '%s': containment teardown left survivors: %s",
                        job_id, _report["leftover"])
            except Exception:
                logger.exception("Job '%s': containment teardown failed", job_id)
        if deferred and worker_future is not None:
            worker_future.add_done_callback(lambda _f: _containment_teardown())
        else:
            _containment_teardown()
    try:
        if scope is not None:
            scope.exit()
    finally:
        if not deferred:
            if session_db is not None:
                _finalize_cron_session(session_db, agent, job_id, job_name, session_id, workdir=workdir)
            elif owner_execution is not None:
                _end_suppressed_cron_session(owner_execution[0].persistence, job, job_id, job_name, session_id)
            if scope is not None:
                if defer_agent_teardown is None:
                    _teardown_cron_agent(agent, job_id)
                elif agent is not None:
                    defer_agent_teardown.append(agent)
