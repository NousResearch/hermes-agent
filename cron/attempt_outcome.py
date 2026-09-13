"""Occurrence-scoped terminal outcomes for cron attempts.

A job's ``fire_claim`` is transport-level: the heartbeat can lose the local fence while the run
still holds it, and a later fire can take the claim. Acting on that as "ownership lost" *after* the
run has already delivered is what wrote the fleet-wide phantom rows of 2026-09-13 — a delivered
run's execution row recorded ``failed`` ("Interrupted by shutdown before terminal completion.") one
second after its successful delivery, flipping ``last_status`` to ``error``.

The policy here keeps three things apart:

* the **attempt** — the execution-ledger row created for one fire. Its own row (``execution_id``) is
  the fence for what the attempt is recorded as, and terminal rows are immutable;
* the **occurrence** — ``scheduled_instant`` (``cron/occurrences.py``). A completed attempt for the
  occurrence owns its outcome: a sibling failure must not sit next to it as another ``error``;
* the **fire claim** — mutable, and therefore never the authority for a run that already delivered.

Nothing here schedules retries or decides due-ness: it classifies and records a terminal outcome
for an attempt that has stopped running.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

COMPLETED = "completed"
SUPERSEDED = "superseded"
INTERRUPTED = "interrupted"

INTERRUPTED_ERROR = "Interrupted by shutdown before terminal completion."
SUPERSEDED_ERROR = (
    "Superseded: a later fire for this job owns its status, so this attempt's outcome was not "
    "applied."
)


# SUPERSEDED and the JOB RECORD — the behaviour is deliberate, not an oversight
# (WH-CREATED-5A4D2A184BBA AC3):
#   * `classify_post_delivery_outcome` returns SUPERSEDED whenever a newer attempt (or a completed
#     sibling of the same `scheduled_instant`) owns the outcome.
#   * `cron/scheduler.py` passes `owns_job_record=False` on every SUPERSEDED branch, so
#     `mark_job_run` is NOT called: `last_status` keeps the value the owning attempt wrote and
#     `failure_streak` is neither advanced nor reset.
#   * Why: the job record describes ONE attempt's outcome, so letting a superseded attempt write it
#     would clobber a newer attempt's real status — the same fail-closed rule that
#     `job_status_write_blocked` encodes and `finish_kwargs` applies.
#   * Accepted consequence, written down so it is not rediscovered as a bug: consecutive superseded
#     fires cannot trip the auto-stop `failure_streak` on their own. They stay visible in the
#     executions ledger and in telemetry (agent/monitoring/cron_health.py maps status `superseded`
#     to the informational error_class `superseded`), and a stale fire claim is recovered by TTL.
def classify_post_delivery_outcome(
    *, delivered: bool, owns_job_record: bool, occurrence_completed: bool,
) -> str:
    """Classify an attempt that has stopped running.

    ``delivered`` — the run's result left the process (delivery attempted without a delivery error).
    ``owns_job_record`` — this attempt may describe the job: it is the newest attempt for the job
    and its fire claim is still ours. ``occurrence_completed`` — another attempt already completed
    this exact ``scheduled_instant``.
    """
    if delivered:
        # The work left the process: a latch that fired post hoc is not a lost run. When the job
        # record belongs to a later fire the outcome was not applied — inconclusive, never failed.
        return COMPLETED if owns_job_record else SUPERSEDED
    if occurrence_completed or not owns_job_record:
        return SUPERSEDED
    return INTERRUPTED


def attempt_is_newest(job_id: str, execution_id: str) -> bool:
    """True when no later attempt for *job_id* exists."""
    from cron.executions import newest_attempt_id

    return newest_attempt_id(job_id) == execution_id


def attempt_overtaken(job_id: str, execution_id: str) -> bool:
    """True when the ledger knows this attempt AND a later attempt for the same job exists.

    A row the ledger does not know is not overtaken — bookkeeping must not go silent because an id
    was pruned or supplied by a caller.
    """
    from cron.executions import get_execution

    if get_execution(execution_id) is None:
        return False
    return not attempt_is_newest(job_id, execution_id)


def attempt_owns_job_record(
    job_id: str, execution_id: str, fire_owner: Optional[str] = None,
) -> bool:
    """Whether this attempt may be trusted to describe the job.

    False when the ledger holds a later attempt for the same job, or when the job carries a fire
    claim that is no longer ours. An attempt row the ledger does not know does not by itself
    disprove ownership — the claim check still decides.
    """
    from cron.executions import get_execution

    if get_execution(execution_id) is not None and not attempt_is_newest(job_id, execution_id):
        return False
    if fire_owner is None:
        return True
    from cron.jobs import heartbeat_fire_claim

    try:
        return bool(heartbeat_fire_claim(job_id, expected_owner=fire_owner))
    except Exception:
        logger.debug(
            "Job '%s': fire claim could not be re-read when classifying attempt %s",
            job_id, execution_id, exc_info=True)
        return False


def job_status_write_blocked(job_id: str, execution_id: str) -> Optional[str]:
    """Why this attempt must not rewrite the job's ``last_status``, or ``None`` when it may.

    The job record describes the newest attempt of an occurrence, so an attempt a later fire — or a
    completed sibling of the same occurrence — has overtaken must record only its own ledger row.
    """
    from cron.executions import get_execution, occurrence_completed

    row = get_execution(execution_id)
    if row is None:
        return None
    if not attempt_is_newest(job_id, execution_id):
        return "a newer attempt for this job owns its status"
    if occurrence_completed(job_id, row.get("scheduled_instant")):
        return "this occurrence already completed"
    return None


def finish_kwargs(
    outcome: str, *, error: Optional[str] = None, delivery_outcome: Optional[str] = None,
) -> dict:
    """Terminal-write kwargs for a classified outcome — one place maps outcome → ledger write."""
    if outcome == COMPLETED:
        kwargs: dict = {"success": True}
    elif outcome == SUPERSEDED:
        kwargs = {"success": False, "superseded": True, "error": error or SUPERSEDED_ERROR}
    else:
        kwargs = {"success": False, "error": error or INTERRUPTED_ERROR}
    if delivery_outcome is not None:
        kwargs["delivery_outcome"] = delivery_outcome
    return kwargs


def record_post_delivery_outcome(
    job_id: str, execution_id: str, *, delivered: bool,
    owns_job_record: Optional[bool] = None, error: Optional[str] = None,
    delivery_outcome: Optional[str] = None, finish: Optional[Callable[..., Any]] = None,
) -> str:
    """Record the attempt's terminal outcome from the ledger and return the classification.

    An attempt the ledger does not know (a caller-supplied or pruned id) is classified the same way
    and its write is a no-op — bookkeeping must not go silent because a row is missing.

    ``finish`` is the caller's own ``finish_execution`` binding; without it the ledger's own is used.
    """
    from cron.executions import finish_execution, get_execution, occurrence_completed

    row = get_execution(execution_id) or {}
    if owns_job_record is None:
        owns_job_record = attempt_owns_job_record(job_id, execution_id)
    outcome = classify_post_delivery_outcome(
        delivered=delivered,
        owns_job_record=owns_job_record,
        occurrence_completed=occurrence_completed(job_id, row.get("scheduled_instant")),
    )
    finalize = finish or finish_execution
    finalize(execution_id, **finish_kwargs(
        outcome, error=error, delivery_outcome=delivery_outcome))
    return outcome


def create_attempt(job: dict, *, source: str, create: Callable[..., Any]) -> Optional[str]:
    """Create the ledger attempt for one fire, or refuse a re-fire of a completed occurrence.

    Returns the attempt id, or ``None`` when the fire was refused: this exact occurrence already
    completed, so running it again would repeat the job's side effects. ``create`` is the caller's
    own ``create_execution`` binding — the seam stays where production (and any override) reads it.
    """
    from cron.executions import DuplicateFireAttempt

    try:
        execution = create(
            job["id"], source=source, scheduled_instant=job.get("_scheduled_instant"))
    except DuplicateFireAttempt as refused:
        logger.warning(
            "Job '%s': fire refused — %s", job.get("name", job.get("id", "?")), refused)
        return None
    return str(execution["id"])
