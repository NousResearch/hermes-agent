"""What happens to occurrences a job could not fire on time (machine off, gateway stopped).

Each job may carry ``catch_up``: ``true`` runs ONE catch-up for everything it missed, ``false``
skips to the next slot. Absent, recurring jobs follow ``cron.catch_up_missed`` (default true)
and one-shots keep the 120s grace rule (#93526: a stale reminder is skipped unless the job opts
in). Either way the skipped or collapsed slots are booked as one ``missed ×N`` row in the
execution ledger, so run history shows the gap instead of nothing.

Called from the due scan in ``cron/jobs.py`` under the store lock; ledger writes ride the
scan's ``on_saved`` hooks so an unwritable store cannot book the same miss every tick.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Optional

logger = logging.getLogger("cron.jobs")

# Counting walks the schedule slot by slot; a once-a-minute job off for a month stops here.
_MAX_COUNTED_SLOTS = 1000


def job_catches_up(job: dict[str, Any]) -> bool:
    """The job's own ``catch_up`` wins; otherwise one-shots skip, and recurring jobs skip past an
    explicit ``grace_minutes`` or else follow ``cron.catch_up_missed``."""
    value = job.get("catch_up")
    if isinstance(value, bool):
        return value
    if (job.get("schedule") or {}).get("kind") == "once":
        return False
    from cron import jobs_time
    from cron.jobs import _cron_config_number

    # An explicit grace_minutes is the user's catch-up limit for THIS job: past it the miss is
    # skipped even while the global cron.catch_up_missed default still runs it once.
    if jobs_time.grace_seconds_override(job.get("grace_minutes")) is not None:
        return False

    return bool(_cron_config_number("catch_up_missed", True, lambda raw: raw is not False))


def count_missed_slots(
    schedule: dict[str, Any], first: str, now: datetime, *, timezone: Any = None,
) -> int:
    """Occurrences from ``first`` up to ``now`` inclusive, capped at ``_MAX_COUNTED_SLOTS``.

    *timezone* is the job's pin (job record or zone name), as for ``compute_next_run``."""
    from cron.jobs import _instant_after, _parse_aware, compute_next_run

    count, cursor = 0, first
    while cursor and count < _MAX_COUNTED_SLOTS:
        at = _parse_aware(cursor)
        if at is None or _instant_after(at, now):
            break
        count += 1
        following = compute_next_run(schedule, cursor, timezone=timezone)
        cursor = following if following != cursor else None
    return count


def _book_missed(
    job: dict[str, Any], count: int, first: Optional[str], detail: str, *, capped: bool,
) -> None:
    """``capped``: counting hit ``_MAX_COUNTED_SLOTS``, so ``count`` is a floor, not exact. Passed
    separately because the collapse branch books one fewer than it counted."""
    from cron.executions import record_missed_occurrences

    if capped:
        detail = f"{detail}; counting stopped at {_MAX_COUNTED_SLOTS}"
    try:
        record_missed_occurrences(str(job.get("id")), count=count, first_instant=first, detail=detail)
    except Exception:
        logger.warning("Job '%s': could not book its missed occurrences", job.get("id"), exc_info=True)


def fast_forward_missed_recurring(d: Any, grace: int) -> bool:
    """Re-anchor accumulated misses; return whether catch-up is off for this job (skip the fire).

    The fast-forward is persisted immediately — NOT redundant with advance_next_run/mark_job_run:
    it protects the crash window before mark_job_run and covers the external fire_due path, which
    never calls advance_next_run. mark_job_run re-anchors on completion, so the value is
    provisional.
    """
    from cron.jobs import _elapsed_seconds, _ensure_aware, _instant_after, record_cron_missed

    if _elapsed_seconds(d.scan.now, d.next_run_dt) <= grace:
        return False
    new_next = d.recompute_next()
    if not new_next:
        return False
    d.scan.persist(d.job["id"], next_run_at=new_next)
    missed = count_missed_slots(d.schedule, d.next_run, d.scan.now, timezone=d.job)
    capped = missed >= _MAX_COUNTED_SLOTS
    if (_instant_after(_ensure_aware(datetime.fromisoformat(new_next)), d.scan.now)
            and not job_catches_up(d.job)):
        logger.info(
            "Job '%s' missed its scheduled time (%s, grace=%ds). "
            "Skipping %d missed occurrence(s) because catch-up is off for this job; next run: %s",
            d.label, d.next_run, grace, missed, new_next)
        d.scan.on_saved.append(lambda job=d.job: record_cron_missed(job))
        d.scan.on_saved.append(lambda job=d.job, first=d.next_run: _book_missed(
            job, missed, first, f"skipped, catch-up off; next run {new_next}", capped=capped))
        return True
    logger.info(
        "Job '%s' missed its scheduled time (%s, grace=%ds). "
        "Running now; next run provisionally set to: %s (re-anchored on completion)",
        d.label, d.next_run, grace, new_next)
    from cron.occurrences import record_catch_up_occurrence
    d.scan.on_saved.append(record_catch_up_occurrence)
    if missed > 1:
        d.scan.on_saved.append(lambda job=d.job, first=d.next_run: _book_missed(
            job, missed - 1, first, "collapsed into one catch-up run", capped=capped))
    return False


def retire_expired_oneshot(d: Any) -> bool:
    """One-shot grace gate; True when the job must not fire this tick.

    A one-shot beyond the grace window fires once late only when the job opted in
    (``catch_up: true``); otherwise it never fires (create/update/resume reject such schedules and
    recovery never revives them). With no claim stamped, retire it with a diagnostic and a
    ``missed ×1`` ledger row (never silently delete). A claim may mean a run is still in flight
    elsewhere — skip but keep the record so its mark_job_run can land."""
    from cron import store_health
    from cron.jobs import (
        ONESHOT_GRACE_SECONDS, _current_cron_store, _elapsed_seconds, record_cron_missed)
    from cron.jobs_oneshot_diagnostics import write_missed

    if _elapsed_seconds(d.scan.now, d.next_run_dt) <= ONESHOT_GRACE_SECONDS:
        return False
    if job_catches_up(d.job):
        logger.info("Job '%s' missed its one-shot time %s; firing it once now (catch_up on).",
                    d.label, d.next_run)
        return False
    # Due during an outage of its store: it was skipped, not missed; fire it once on recovery.
    if store_health.outage_covers(
            _current_cron_store().cron_dir, d.next_run_dt.timestamp(), ONESHOT_GRACE_SECONDS):
        return False
    claim = d.job.get("run_claim")
    # An outage-covered claim that never reached its fire claim never started: it stays covered.
    if isinstance(claim, dict) and claim.get("outage") and not d.job.get("fire_claim"):
        return False
    if not (claim or d.job.get("fire_claim")):
        write_missed(d.job, d.next_run)
        d.scan.retire(d.job["id"])
        d.scan.on_saved.append(lambda job=d.job: record_cron_missed(job))
        d.scan.on_saved.append(lambda job=d.job, first=d.next_run: _book_missed(
            job, 1, first, "one-shot past its grace window; catch-up off", capped=False))
    return True
