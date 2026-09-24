"""Hold a job's fires while a provider's usage window is known to be closed (#89376).

A quota-exhausted provider answers with an explicit ``retry after <N>s`` (Codex 429: the
``AuthError`` from ``hermes_cli.auth_codex._codex_quota_exhausted_error``). When the whole
fallback chain is unavailable, re-firing on cadence is guaranteed to fail identically until
the window reopens — every fire is a usage probe plus a delivered failure alert. The failing
run's alert says the job is held; ``mark_job_run`` then parks ``next_run_at`` at the recovery
boundary (or the first legal occurrence after it, when several fall inside the window) and
stamps ``quota_hold_until`` so the stale-error re-arm
(``cron.jobs._job_is_stale_error_recurring``) does not pull the job back early.

Complement to ``cron/unreachable_retry.py``: this one moves ``next_run_at`` out of a known
closed provider window. Any run that reaches the model clears the marker.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any, Dict, Optional

from hermes_time import now as _hermes_now, safe_strftime

logger = logging.getLogger("cron.scheduler")

# Persisted while a hold is active: ISO instant the job was parked at.
STATE_KEY = "quota_hold_until"
SCHEDULE_EXPR_KEY = "quota_hold_cron_expr"

# The provider's remaining seconds were measured when the probe ran; by the time the run is
# recorded a little wall clock has passed, so land clearly past the boundary.
HOLD_SLACK_SECONDS = 60

_RETRY_AFTER_RE = re.compile(r"retry after (\d+)s", re.IGNORECASE)


def hold_seconds_from_failure(exc: BaseException) -> Optional[float]:
    """Seconds the provider said it will stay closed, or None when *exc* (or anything in its
    cause chain) is not a rate-limited ``AuthError`` carrying a wait hint. Anchored on the
    AuthError itself, never on arbitrary text, so an unrelated "retry after" in an agent's
    output cannot park a job."""
    from hermes_cli.auth import AuthError, is_rate_limited_auth_error

    seen: set[int] = set()
    cur: Optional[BaseException] = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, AuthError) and is_rate_limited_auth_error(cur):
            hint = getattr(cur, "retry_after", None)
            if hint is None:
                m = _RETRY_AFTER_RE.search(str(cur))
                hint = float(m.group(1)) if m else None
            return float(hint) if hint is not None and float(hint) > 0 else None
        cur = cur.__cause__ or cur.__context__
    return None


def hold_active(job: Dict[str, Any], now: Optional[datetime] = None) -> bool:
    """True while the job is parked inside a provider window (an expired marker is inert)."""
    from cron.jobs import _instant_after, _parse_aware  # late: jobs imports this module's helpers

    until = _parse_aware(job.get(STATE_KEY)) if job.get(STATE_KEY) else None
    return until is not None and _instant_after(until, now or _hermes_now())


def clear_state(job: Dict[str, Any]) -> None:
    job.pop(STATE_KEY, None)
    job.pop(SCHEDULE_EXPR_KEY, None)


def is_recovery_fire(job: Dict[str, Any], next_run: str) -> bool:
    """True for the exact off-lattice cron fire parked by ``plan_hold``.

    The expression fingerprint keeps a direct ``jobs.json`` schedule edit from inheriting the
    exception: edited schedules must still re-anchor without firing.
    """
    schedule = job.get("schedule") or {}
    return (
        schedule.get("kind") == "cron"
        and job.get(STATE_KEY) == next_run
        and job.get(SCHEDULE_EXPR_KEY) == schedule.get("expr")
    )


def _window_end(hold_seconds: float) -> datetime:
    from cron.jobs import _seconds_after

    return _seconds_after(_hermes_now(), float(hold_seconds) + HOLD_SLACK_SECONDS)


def _recovery_worthwhile(
    job: Dict[str, Any], schedule: Dict[str, Any], natural_next: Optional[datetime],
    window_end: datetime, scheduled: bool,
) -> bool:
    """One off-lattice recovery fire, and only for a sparse schedule.

    Bounded: a job already carrying ``quota_hold_until`` IS the recovery fire failing again, so
    it waits for the natural schedule instead of re-parking at every hold boundary (the
    probe-per-window cost the hold exists to prevent). Sparse: the natural occurrence must be at
    least half a cadence period past the boundary — the same half-period rule as
    ``cron.jobs._compute_grace_seconds`` — otherwise the recovery fire is a near-duplicate of the
    natural one (hourly job, hold ending at :58, would fire :58 AND :00).
    """
    from cron.jobs import _schedule_cadence_seconds

    if not scheduled or job.get(STATE_KEY) or natural_next is None:
        return False
    cadence = _schedule_cadence_seconds(schedule)
    return bool(cadence) and (natural_next - window_end).total_seconds() >= cadence / 2


def plan_hold(
    job: Dict[str, Any], hold_seconds: float, *, recover_consumed_fire: bool = False,
) -> bool:
    """Called under the jobs lock AFTER ``_advance_after_run`` computed the schedule's natural
    ``next_run_at`` for a failed run. A scheduled sparse cron may retry its consumed fire at the
    recovery boundary; manual runs keep the natural schedule. Otherwise coalesce fires through
    the closed window. Returns True when parked."""
    from cron.jobs import _instant_before, _parse_aware, compute_next_run

    schedule = job.get("schedule") or {}
    kind = schedule.get("kind")
    if kind not in {"cron", "interval"} or job.get("state") == "paused":
        clear_state(job)
        return False
    window_end = _window_end(hold_seconds)
    natural_next = _parse_aware(job.get("next_run_at"))
    blocked = natural_next is None or _instant_before(natural_next, window_end)
    if kind == "interval":
        if not blocked:
            # No interval occurrence is blocked: its natural next run is already after recovery.
            clear_state(job)
            return False
        parked = window_end.isoformat()
        job.pop(SCHEDULE_EXPR_KEY, None)
    else:
        if not blocked:
            if not _recovery_worthwhile(job, schedule, natural_next, window_end, recover_consumed_fire):
                clear_state(job)
                return False
            parked = window_end.isoformat()
        else:
            # Coalesce cron occurrences inside the closed window to the first legal instant after it.
            parked = compute_next_run(schedule, window_end.isoformat()) or window_end.isoformat()
        job[SCHEDULE_EXPR_KEY] = schedule.get("expr")
    job["next_run_at"] = parked
    job[STATE_KEY] = parked
    logger.warning(
        "Job '%s': provider usage window closed for %.0fs — holding fires until %s instead of "
        "failing on every cadence tick",
        job.get("name", job.get("id", "?")), float(hold_seconds), parked)
    return True


def hold_notice(job: Dict[str, Any], hold_seconds: Optional[float]) -> str:
    """Line appended to the ONE failure alert delivered on entering the hold, else ""."""
    if not hold_seconds or (job.get("schedule") or {}).get("kind") not in {"cron", "interval"}:
        return ""
    window_end = _window_end(hold_seconds)
    hours = float(hold_seconds) / 3600.0
    return (
        f"\nThe provider's usage window is closed for about {hours:.1f}h. This job is held "
        f"through {safe_strftime(window_end, '%Y-%m-%d %H:%M %Z')} and resumes at the first safe "
        "opportunity afterwards; no further alerts are sent while the provider is unavailable."
    )
