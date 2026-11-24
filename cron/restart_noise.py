"""Grace window for cron failures caused by the gateway's own restart or update.

A job that fires while the gateway is draining, down or coming back fails before anything is
submitted (``Gateway draining: update_paused``, ``Gateway connection failed``, ``Cron admission
missing``). Every frequent job hits that window, so one restart used to post one failure notice
per job, each for a job that ran fine a few minutes later.

The notice for such a failure is withheld while the job's current run of consecutive failures is
younger than ``cron.restart_failure_grace_minutes``. A job still failing that way after the window
alerts normally, so a gateway that never comes back is still reported. Only the notice is held:
the run, the failure streak and the incident are recorded as for any other failure.

A worker that dies before taking its execution (``Restart-safe cron worker dispatch failed``) also
happens with no restart involved, so it counts only while some job has reported a restart failure
inside the window. A run that started and was then interrupted is never held.

State lives beside the incidents in ``cron/executions.db``: one row per job holding when its
current run of restart failures began.
"""

from __future__ import annotations

import logging
import time
from typing import Literal, Optional

from cron import incidents

logger = logging.getLogger(__name__)

DEFAULT_RESTART_FAILURE_GRACE_MINUTES = 30.0

FailureKind = Literal["restart", "worker_exit"]

# Normalized (lowercase, single-spaced) error text -> kind. First match wins. Only failures where
# nothing was submitted belong here: an interrupted or unverified run may have had side effects,
# and that notice is the operator's only word of it.
_PATTERNS: tuple[tuple[str, FailureKind], ...] = (
    # hermes_cli.gateway_client.GatewayUnavailableError: no ready owner (draining, starting, ...).
    ("gatewayunavailableerror:", "restart"),
    # hermes_cli.gateway_client / gateway_connection: the owner's socket is gone or not yet up.
    ("gateway connection failed; no local fallback", "restart"),
    ("gateway connection timed out; no local fallback", "restart"),
    # cron.scheduler_authority.reconcile_pending: the owner holds no admission for this fire.
    ("cron admission missing:", "restart"),
    # cron.scheduler: the detached worker died before it took the execution.
    ("restart-safe cron worker dispatch failed", "worker_exit"),
)


def classify(error: str) -> Optional[FailureKind]:
    """Which restart-related kind ``error`` is, or ``None`` for an ordinary failure."""
    text = " ".join(str(error or "").split()).lower()
    return next((kind for pattern, kind in _PATTERNS if pattern in text), None)


def grace_seconds() -> float:
    """``cron.restart_failure_grace_minutes`` in seconds; ``0`` (or negative) disables the hold."""
    from cron.jobs import _cron_config_number

    minutes = _cron_config_number(
        "restart_failure_grace_minutes", DEFAULT_RESTART_FAILURE_GRACE_MINUTES, float)
    return max(0.0, minutes) * 60.0


def withhold_notice(job: dict, error: str) -> bool:
    """True when this failure's notice should be held as restart noise.

    ``job["failure_streak"]`` is the stored count before this run, so ``0`` means the previous run
    succeeded and this failure starts a new run of them.
    """
    kind = classify(error)
    grace = grace_seconds()
    if kind is None or grace <= 0:
        return False
    job_id = str(job["id"])
    now = time.time()
    with incidents._transaction() as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS cron_restart_failures (
                 job_id       TEXT PRIMARY KEY,
                 started_at   REAL NOT NULL,
                 last_seen_at REAL NOT NULL,
                 restart_seen INTEGER NOT NULL
               )"""
        )
        if kind == "worker_exit" and conn.execute(
            "SELECT 1 FROM cron_restart_failures WHERE restart_seen=1 AND last_seen_at>=?",
            (now - grace,),
        ).fetchone() is None:
            return False
        row = conn.execute(
            "SELECT started_at, restart_seen FROM cron_restart_failures WHERE job_id=?", (job_id,)
        ).fetchone()
        previous = row if int(job.get("failure_streak") or 0) > 0 else None
        started_at = float(previous["started_at"]) if previous is not None else now
        restart_seen = int(
            kind == "restart" or (previous is not None and bool(previous["restart_seen"])))
        conn.execute(
            "INSERT OR REPLACE INTO cron_restart_failures VALUES (?, ?, ?, ?)",
            (job_id, started_at, now, restart_seen),
        )
    age = now - started_at
    if age < grace:
        logger.info(
            "Job '%s': holding the failure notice as gateway restart noise (%.0fs into the %.0fs "
            "grace window): %s", job_id, age, grace, str(error)[:200])
        return True
    logger.warning(
        "Job '%s': still failing on the gateway %.0fs after the restart window opened; alerting",
        job_id, age)
    return False
