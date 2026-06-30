"""
Cron-job registration for the session-orchestration watcher (T006).

Usage
-----
Call ``ensure_watcher_cron()`` once at Hermes startup (or from a setup
command) to idempotently register the ``--no-agent`` cron job that drives
``session-orchestration-watch.sh`` on a 1-minute cadence.  The function is
a no-op when the job already exists.

This module does NOT touch ``gateway/run.py`` or ``hermes_cli/commands.py``
(per the T006 concurrency guardrail — T011 owns those files).

The registration is isolated in this module so:
- It can be called from any context (CLI helper, gateway startup hook, tests).
- Tests can inject a fake job-list and verify registration without side-effects.
- The watcher cron is a single authoritative entry; duplicate registration
  attempts are silently ignored.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Canonical job name used for duplicate-detection.
WATCHER_JOB_NAME = "session-orchestration-watch"

#: Cron schedule: every 1 minute.
WATCHER_SCHEDULE = "every 1 minute"

#: Path to the --no-agent script, relative to the hermes-agent repo root.
_SCRIPT_RELPATH = "scripts/session-orchestration-watch.sh"


def _hermes_agent_root() -> Path:
    """Return the hermes-agent repo root (parent of this file's package)."""
    return Path(__file__).parent.parent.resolve()


def _find_script_path() -> Optional[str]:
    """Return the safe relative watcher script token, or None if missing."""
    script = _hermes_agent_root() / _SCRIPT_RELPATH
    if script.exists():
        return _SCRIPT_RELPATH
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def ensure_watcher_cron(
    *,
    job_name: str = WATCHER_JOB_NAME,
    schedule: str = WATCHER_SCHEDULE,
    _jobs_loader=None,
    _jobs_creator=None,
) -> Optional[Dict[str, Any]]:
    """Idempotently register the session-orchestration watcher cron job.

    If a job named ``job_name`` already exists (enabled or disabled), this
    function returns that job and does nothing.  Otherwise it creates a new
    ``--no-agent`` cron job that runs ``scripts/session-orchestration-watch.sh``
    on the given ``schedule``.

    Parameters
    ----------
    job_name:
        Name used for duplicate detection and the ``name`` field of the
        created job.
    schedule:
        Cron schedule string (default: ``"every 1 minute"``).
    _jobs_loader:
        Injectable callable ``() -> List[Dict]`` for tests.  Defaults to
        ``cron.jobs.load_jobs``.
    _jobs_creator:
        Injectable callable matching ``cron.jobs.create_job`` signature for
        tests.  Defaults to ``cron.jobs.create_job``.

    Returns
    -------
    dict | None
        The existing or newly-created job dict, or ``None`` if the script
        could not be located.
    """
    # Lazy imports to avoid circular deps at module-load time.
    if _jobs_loader is None:
        from cron.jobs import load_jobs as _jobs_loader  # type: ignore[assignment]
    if _jobs_creator is None:
        from cron.jobs import create_job as _jobs_creator  # type: ignore[assignment]

    # Duplicate check — linear scan (O(n) over jobs, typically < 100).
    existing: List[Dict[str, Any]] = _jobs_loader()
    for job in existing:
        if job.get("name") == job_name:
            logger.debug(
                "ensure_watcher_cron: job '%s' already registered (id=%s) — no-op",
                job_name,
                job.get("id", "?"),
            )
            return job

    # Locate the shell script
    script_path = _find_script_path()
    if script_path is None:
        logger.error(
            "ensure_watcher_cron: script not found at %s/%s — "
            "cannot register cron job",
            _hermes_agent_root(),
            _SCRIPT_RELPATH,
        )
        return None

    # Create the job
    try:
        job = _jobs_creator(
            prompt=None,
            schedule=schedule,
            name=job_name,
            script=script_path,
            no_agent=True,
            deliver="origin",  # normal cron delivery; falls back to configured home channel
        )
        logger.info(
            "ensure_watcher_cron: registered '%s' (id=%s) on schedule '%s'",
            job_name,
            job.get("id", "?"),
            schedule,
        )
        return job
    except Exception as exc:
        logger.error(
            "ensure_watcher_cron: failed to create cron job '%s': %s",
            job_name,
            exc,
        )
        return None


def remove_watcher_cron(
    *,
    job_name: str = WATCHER_JOB_NAME,
    _jobs_loader=None,
    _jobs_remover=None,
) -> bool:
    """Remove the session-orchestration watcher cron job if it exists.

    Intended for tests and uninstall flows.  Returns True if a job was
    removed, False if none was found.
    """
    if _jobs_loader is None:
        from cron.jobs import load_jobs as _jobs_loader  # type: ignore[assignment]
    if _jobs_remover is None:
        from cron.jobs import remove_job as _jobs_remover  # type: ignore[assignment]

    existing: List[Dict[str, Any]] = _jobs_loader()
    for job in existing:
        if job.get("name") == job_name:
            try:
                _jobs_remover(job["id"])
                logger.info("remove_watcher_cron: removed job '%s' (id=%s)", job_name, job["id"])
                return True
            except Exception as exc:
                logger.warning("remove_watcher_cron: failed to remove '%s': %s", job_name, exc)
                return False
    return False
