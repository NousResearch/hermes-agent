"""Grace window for cron failures caused by the gateway's own restart or update.

A job that fires while the gateway is draining, down or coming back is refused before it is
admitted (``Gateway draining: update_paused``, ``Gateway connection failed``, ``Cron admission
missing``). Every frequent job hits that window, so one restart used to post one failure notice
per job, each for a job that ran fine a few minutes later.

The notice for such a failure is withheld while the job's current run of consecutive failures is
younger than ``cron.restart_failure_grace_minutes``. A job still failing that way after the window
alerts normally, so a gateway that never comes back is still reported. Only the notice is held:
the run, the failure streak and the incident are recorded as for any other failure.

Which failures qualify is decided where the fire is refused, never from the error text: the code
that knows the fire was not admitted stamps ``job[NOT_ADMITTED]`` (see ``mark_not_admitted``).
A failure without the stamp alerts at once, whatever its message says, because a run that started
may have had side effects.

A worker that dies before taking its execution (``WORKER_EXIT``) also happens with no restart
involved, so it counts only while some job has been refused by the gateway inside the window.

State lives beside the incidents in ``cron/executions.db``: one row per job holding when its
current run of restart failures began.
"""

from __future__ import annotations

import logging
import time

from cron import incidents

logger = logging.getLogger(__name__)

DEFAULT_RESTART_FAILURE_GRACE_MINUTES = 30.0

# Job-dict stamp: this fire failed before the owner admitted it, so no agent or script ran.
NOT_ADMITTED = "_not_admitted"
GATEWAY_UNAVAILABLE = "gateway_unavailable"  # the gateway refused, was unreachable, or forgot the fire
WORKER_EXIT = "worker_exit"  # the detached worker died before adopting the execution


def mark_not_admitted(job: dict, cause: str) -> None:
    """Stamp ``job`` at a site that knows the fire was never admitted. ``cause`` is
    ``GATEWAY_UNAVAILABLE`` or ``WORKER_EXIT``."""
    job[NOT_ADMITTED] = cause


def grace_seconds() -> float:
    """``cron.restart_failure_grace_minutes`` in seconds; ``0`` (or negative) disables the hold."""
    from cron.jobs import _cron_config_number

    minutes = _cron_config_number(
        "restart_failure_grace_minutes", DEFAULT_RESTART_FAILURE_GRACE_MINUTES, float)
    return max(0.0, minutes) * 60.0


def withhold_notice(job: dict) -> bool:
    """True when this failure's notice should be held as restart noise.

    ``job["failure_streak"]`` is the stored count before this run, so ``0`` means the previous run
    succeeded and this failure starts a new run of them.
    """
    cause = job.get(NOT_ADMITTED)
    grace = grace_seconds()
    if cause not in (GATEWAY_UNAVAILABLE, WORKER_EXIT) or grace <= 0:
        return False
    job_id = str(job["id"])
    now = time.time()
    with incidents._transaction() as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS cron_restart_failures (
                 job_id       TEXT PRIMARY KEY,
                 started_at   REAL NOT NULL,
                 last_seen_at REAL NOT NULL,
                 gateway_seen INTEGER NOT NULL
               )"""
        )
        if cause == WORKER_EXIT and conn.execute(
            "SELECT 1 FROM cron_restart_failures WHERE gateway_seen=1 AND last_seen_at>=?",
            (now - grace,),
        ).fetchone() is None:
            return False
        row = conn.execute(
            "SELECT started_at, gateway_seen FROM cron_restart_failures WHERE job_id=?", (job_id,)
        ).fetchone()
        previous = row if int(job.get("failure_streak") or 0) > 0 else None
        started_at = float(previous["started_at"]) if previous is not None else now
        gateway_seen = int(
            cause == GATEWAY_UNAVAILABLE
            or (previous is not None and bool(previous["gateway_seen"])))
        conn.execute(
            "INSERT OR REPLACE INTO cron_restart_failures VALUES (?, ?, ?, ?)",
            (job_id, started_at, now, gateway_seen),
        )
    age = now - started_at
    if age < grace:
        logger.info(
            "Job '%s': holding the failure notice as gateway restart noise (%s, %.0fs into the "
            "%.0fs grace window)", job_id, cause, age, grace)
        return True
    logger.warning(
        "Job '%s': still not admitted %.0fs after the restart window opened; alerting",
        job_id, age)
    return False
