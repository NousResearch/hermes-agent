"""Exact scheduled identities, independent of mutable jobs.json dispatch stamps."""
from datetime import datetime, timedelta, timezone
import logging

logger = logging.getLogger(__name__)


def scheduled_instant(value):
    """Canonicalize aware instants; legacy/ambiguous values carry no exact identity."""
    if not isinstance(value, str):
        return None
    try:
        instant = datetime.fromisoformat(value)
        if instant.tzinfo is None:
            return None
        return instant.astimezone(timezone.utc).isoformat()
    except ValueError:
        return None


def completed_occurrence_row(job, instant):
    """The completed execution row claiming ``instant`` for this job, or None.

    Unknown/failed/pruned attempts cannot prove completion: keep them eligible."""
    from cron.executions import _transaction
    from cron.jobs import FIRE_CLAIM_SKEW_SECONDS

    instant = scheduled_instant(instant)
    if instant is None:
        return None
    # A skewed early fire (see claim_job_for_fire) legitimately completes just before its slot.
    earliest_real = datetime.fromisoformat(instant) - timedelta(seconds=FIRE_CLAIM_SKEW_SECONDS)
    try:
        with _transaction() as conn:
            rows = conn.execute(
                "SELECT id, finished_at, claimed_at FROM executions "
                "WHERE job_id=? AND scheduled_instant=? "
                "AND status='completed' ORDER BY claimed_at DESC",
                (str(job['id']), instant),
            ).fetchall()
        for row in rows:
            completed_at = scheduled_instant(row["finished_at"] or row["claimed_at"])
            # Legacy or malformed timestamps remain proof; only positively identified poison
            # rows — completions recorded before their claimed occurrence — are ignored.
            if completed_at is None or datetime.fromisoformat(completed_at) >= earliest_real:
                return dict(row)
        return None
    except Exception:
        logger.warning("Cannot check completed occurrence for job %s", job['id'], exc_info=True)
        return None


def completed_occurrence(job, instant):
    return completed_occurrence_row(job, instant) is not None


# A legitimate execution row covering slot T is claimed at/after T — the fire happens at/after
# its scheduled instant, and only sub-minute clock skew runs the other way. A matching row
# claimed well BEFORE the instant it claims was stamped by something other than a run of that
# slot (the off-tick stamping bug class: #105704, the dashboard trigger, #111414), so the skip
# it causes is a real missed fire, not a re-delivery dedup.
COMPLETED_OCCURRENCE_BACKDATE_TOLERANCE_SECONDS = 300.0


def note_completed_occurrence_skip(job, instant, row):
    """A due slot was consumed by the completed-occurrence dedup, not dispatched: say so.

    Always logs the skip with the matching row. When the row was claimed before the instant it
    claims — an identity minted by a past run of a DIFFERENT slot (#111414) — returns an anomaly
    detail for the caller to stamp as ``last_fire_error``; None for a timely re-delivery dedup.
    """
    row_id = str((row or {}).get("id") or "?")
    claimed_at = str((row or {}).get("claimed_at") or "?")
    logger.warning(
        "Job '%s' (%s): occurrence %s was skipped, not dispatched — completed execution row %s "
        "(claimed %s) already carries this identity.",
        job.get("name", job.get("id")), job.get("id"), instant, row_id, claimed_at)
    backdate = None
    try:
        claimed = datetime.fromisoformat(claimed_at)
        due = datetime.fromisoformat(instant)
        if claimed.tzinfo is not None and due.tzinfo is not None:
            backdate = (due - claimed).total_seconds()
    except ValueError:
        pass
    if backdate is None or backdate <= COMPLETED_OCCURRENCE_BACKDATE_TOLERANCE_SECONDS:
        return None
    return (
        f"Scheduled occurrence {instant} was skipped without any run: completed execution row "
        f"{row_id} (claimed {claimed_at}, {(backdate / 86400.0):.1f} days earlier) already "
        "carries this occurrence identity, so the dedup gate consumed the slot. The identity "
        "was stamped before the occurrence existed (stale off-tick stamping, #111414 class); "
        "the slot ran nowhere."
    )


# --- Pending slot: the occurrence a tick took off the schedule but has not yet claimed ---
#
# The tick advances a recurring job's ``next_run_at`` BEFORE dispatch (at-most-once: a crash
# mid-run must not re-fire on every restart). That leaves a window — advance done, fire claim
# not yet taken (interpreter finalizing, executor refusing work, process killed) — in which the
# process exiting loses the occurrence silently: the restarted scan sees a future ``next_run_at``
# and nothing ever ran (#107485). ``pending_slot`` is the durable record of that window: the due
# scan stamps it with the exact stored instant plus the stamping owner, ``claim_job_for_fire``
# (the point after which side effects may exist) clears it, and any explicit rewrite of the
# schedule drops it. A slot still pending once its owner is provably gone (or its lease has
# expired) was never claimed, so it is restored ONCE as the due instant and then flows through
# the ordinary late / fast-forward / ``cron.catch_up_missed`` policy — never N replays.

def pending_slot_stamp(next_run, now):
    """Store value for a recurring occurrence about to be handed to the dispatcher."""
    from cron.jobs import _machine_id

    return {"scheduled_at": next_run, "at": now.isoformat(), "by": _machine_id()}


def unclaimed_pending_slot(job, now):
    """Stored instant of a slot the dispatcher never claimed, or None.

    None for non-recurring jobs and malformed stamps (never a fire) and for a job still in this
    process's running set (its queued worker will claim and clear the slot itself). A stamp by
    THIS process on a job not running here is orphaned (dispatch refused). A stamp by another
    process is honoured while that owner may still be alive within the fire-claim lease — a
    second live gateway on the same store is mid-dispatch, not dead."""
    from cron.jobs import (
        FIRE_CLAIM_TTL_SECONDS, _claim_is_live, _job_running_in_this_process, _machine_id,
    )

    pending = job.get("pending_slot")
    if not isinstance(pending, dict):
        return None
    slot = pending.get("scheduled_at")
    if job.get("schedule", {}).get("kind") not in {"cron", "interval"} or not isinstance(slot, str):
        return None
    try:
        datetime.fromisoformat(slot)
    except ValueError:
        return None
    if _job_running_in_this_process(str(job.get("id", ""))):
        return None
    if pending.get("by") != _machine_id() and _claim_is_live(pending, now, FIRE_CLAIM_TTL_SECONDS):
        return None
    return slot
