"""Job removal, including the subprocess boundary of run-local delivery permission."""

import logging
import os
import shutil
import sqlite3

logger = logging.getLogger(__name__)


def remove_job(job_id: str) -> bool:
    """Remove a job by ID or name without discarding a worker child's own live result."""
    from cron import jobs as store

    job = store.resolve_job_ref(job_id)
    if not job:
        return False
    canonical_id = job["id"]
    with store._jobs_lock():
        jobs = store.load_jobs()
        original_len = len(jobs)
        jobs = [j for j in jobs if j["id"] != canonical_id]
        if len(jobs) == original_len:
            return False
        marker = store._self_removal_delivery.get()
        if marker is None or marker.job_id != canonical_id:
            _check_subprocess_removal(canonical_id)
        # Resolve BEFORE saving so a legacy unsafe ID fails closed without a half-applied removal.
        job_output_dir = store._job_output_dir(canonical_id)
        store.save_jobs(jobs, removed_ids={canonical_id})
        if marker is not None and marker.job_id == canonical_id:
            marker.removed = True
        if job_output_dir.exists():
            shutil.rmtree(job_output_dir)
        try:
            from cron.notepad import clear_notepad
            clear_notepad(canonical_id)
        except Exception:
            logger.debug("Failed to clear notepad for removed job %s", canonical_id, exc_info=True)
        # Prune the fire-fence lock entry so the registry doesn't grow monotonically.
        fence_key = f"{store._current_cron_store().cron_dir.resolve()}::{canonical_id}"
        with store._fire_fence_locks_guard:
            store._fire_fence_locks.pop(fence_key, None)
        return True


def _check_subprocess_removal(job_id: str) -> None:
    """A child inherits the worker id, but cannot set its parent's delivery ContextVar."""
    from cron import executions, jobs
    from hermes_constants import get_hermes_home

    execution_id = os.environ.get("_HERMES_CRON_EXTERNAL_WORKER")
    if execution_id is None:
        return
    unverifiable = (
        f"Cannot remove job '{job_id}': cannot verify the inherited cron worker execution "
        "in this profile's ledger. Pause it instead, or remove it from an independent "
        "session after the run finishes.")
    cron_dir = get_hermes_home().resolve() / "cron"
    ledger = executions.EXECUTIONS_FILE or (cron_dir / "executions.db")
    if (not execution_id or jobs._current_cron_store().cron_dir.resolve() != cron_dir
            or ledger.resolve() != cron_dir / "executions.db" or not ledger.is_file()):
        raise ValueError(unverifiable)
    try:
        execution = executions.get_execution(execution_id)
    except (OSError, sqlite3.Error) as exc:
        raise ValueError(unverifiable) from exc
    if execution is None:
        raise ValueError(unverifiable)
    if execution["job_id"] == job_id and execution["status"] in {"claimed", "running"}:
        raise ValueError(
            f"Cannot remove job '{job_id}' from its running worker subprocess: "
            "removing it would discard this run's final delivery. "
            "Pause it to prevent future runs, then remove it after this run finishes.")
