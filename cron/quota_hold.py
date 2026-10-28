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

from cron import extra_fire
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


def hold_active(job: dict[str, Any], now: Optional[datetime] = None) -> bool:
    """True while the job is parked inside a provider window (an expired marker is inert)."""
    from cron.jobs import _instant_after, _parse_aware  # late: jobs imports this module's helpers

    until = _parse_aware(job.get(STATE_KEY)) if job.get(STATE_KEY) else None
    return until is not None and _instant_after(until, now or _hermes_now())


def clear_state(job: dict[str, Any]) -> None:
    job.pop(STATE_KEY, None)
    job.pop(SCHEDULE_EXPR_KEY, None)


def is_recovery_fire(job: dict[str, Any], next_run: str) -> bool:
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
    job: dict[str, Any], natural_next: datetime, window_end: datetime,
) -> bool:
    """One off-lattice recovery fire, and only for a sparse schedule.

    Bounded: a job already carrying ``quota_hold_until`` IS the recovery fire failing again, so
    it waits for the natural schedule instead of re-parking at every hold boundary (the
    probe-per-window cost the hold exists to prevent). Sparse: the natural occurrence must be at
    least half a cadence period past the boundary — the same half-period rule as
    ``cron.jobs._compute_grace_seconds`` — otherwise the recovery fire is a near-duplicate of the
    natural one (hourly job, hold ending at :58, would fire :58 AND :00).
    """
    from cron.jobs import _elapsed_seconds, _schedule_cadence_seconds

    if job.get(STATE_KEY):
        return False
    cadence = _schedule_cadence_seconds(job.get("schedule") or {})
    return bool(cadence) and _elapsed_seconds(natural_next, window_end) >= cadence / 2


def plan_hold(
    job: dict[str, Any], hold_seconds: float, *, recover_consumed_fire: bool = False,
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
    recover = (kind == "cron" and not blocked and recover_consumed_fire
               and _recovery_worthwhile(job, natural_next, window_end))
    if not blocked and not recover:
        clear_state(job)
        return False
    if kind == "cron" and blocked and not extra_fire.queued(job):
        # Coalesce cron occurrences inside the closed window to the first legal instant after it.
        # A queued run-now is not an occurrence: it waits for the boundary, not the next slot.
        parked = compute_next_run(schedule, window_end.isoformat()) or window_end.isoformat()
    else:
        parked = window_end.isoformat()
    if recover:
        # Only the recovery fire is off-lattice; the coalesced instant is a legal occurrence.
        job[SCHEDULE_EXPR_KEY] = schedule.get("expr")
    else:
        job.pop(SCHEDULE_EXPR_KEY, None)
    extra_fire.reschedule(job, parked)
    job[STATE_KEY] = parked
    logger.warning(
        "Job '%s': provider usage window closed for %.0fs — holding fires until %s instead of "
        "failing on every cadence tick",
        job.get("name", job.get("id", "?")), float(hold_seconds), parked)
    return True


def hold_notice(job: dict[str, Any], hold_seconds: Optional[float]) -> str:
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


# Per-job choice when its provider is already known to be at a usage limit at fire time:
# "wait" (default) runs the job once when the limit resets; "skip" drops the occurrences inside
# the window and resumes on the job's natural schedule afterwards.
ON_LIMIT_KEY = "on_limit"
ON_LIMIT_CHOICES = ("wait", "skip")


def hold_seconds_for_failed_run(exc: BaseException, agent: Any) -> Optional[float]:
    """Seconds to park a failed run: the AuthError hint, else the provider-scoped usage hold the
    agent's own terminal path just recorded (``agent.usage_hold``; any provider, any 429 shape)."""
    hint = hold_seconds_from_failure(exc)
    if hint or agent is None:
        return hint
    from agent.usage_hold import active_hold

    hold = active_hold(getattr(agent, "provider", ""), getattr(agent, "base_url", ""))
    return max(1.0, hold["until"] - _hermes_now().timestamp()) if hold else None


def _job_route(job: dict[str, Any], cfg: dict[str, Any]) -> tuple[str, str]:
    """Provider + base_url a job will call, resolved exactly as ``_resolve_job_runtime`` does
    (pin > cron default > persisted config, through ``resolve_runtime_provider``) so a bare
    ``custom`` pin or a named custom provider lands on the same ``custom:<url>`` key the failing
    turn recorded. An unresolvable route is never held (fail-open: the run reports it)."""
    from hermes_cli.runtime_provider import resolve_runtime_provider

    cron_cfg = cfg.get("cron") if isinstance(cfg.get("cron"), dict) else {}
    kwargs: dict[str, Any] = {
        "requested": job.get("provider") or str(cron_cfg.get("model_provider") or "").strip() or None,
        "target_model": job.get("model") or "",
    }
    if job.get("base_url"):
        kwargs["explicit_base_url"] = job.get("base_url")
    try:
        runtime = resolve_runtime_provider(**kwargs) or {}
    except Exception:  # health: allow BLE001 -- resolve failures are the run's to report, not the gate's
        logger.debug("usage-limit gate: route for job %s unresolvable", job.get("id"), exc_info=True)
        return "", ""
    return str(runtime.get("provider") or ""), str(runtime.get("base_url") or "")


def _defer_for_provider_hold(job_id: str, hold: dict[str, Any]) -> bool:
    """Move a due job past its provider's window under the jobs lock (wait or skip)."""
    from cron.jobs import _parse_aware, _with_job, compute_next_run, save_jobs

    def apply(jobs, _i, job):
        schedule = job.get("schedule") or {}
        if schedule.get("kind") not in {"cron", "interval"} or job.get("state") == "paused":
            return False
        # The due scan stamped this slot as handed to the dispatcher; parking is a deliberate
        # reschedule, so the stamp must go in the same save or the next scan "restores" it.
        slot_stamped = job.pop("pending_slot", None) is not None
        window_end = datetime.fromtimestamp(float(hold["until"]) + HOLD_SLACK_SECONDS).astimezone()
        natural = compute_next_run(schedule, window_end.isoformat())
        if job.get(ON_LIMIT_KEY) == "skip":
            parked = natural or window_end.isoformat()
            job.pop(SCHEDULE_EXPR_KEY, None)
            job["last_status"] = "skipped_usage_limit"
        else:
            parked = window_end.isoformat()
            # The catch-up fire is off-lattice for a cron expr; fingerprint it (is_recovery_fire).
            if schedule.get("kind") == "cron" and natural != parked:
                job[SCHEDULE_EXPR_KEY] = schedule.get("expr")
        current = _parse_aware(job.get("next_run_at"))
        if current is not None and current >= _parse_aware(parked):
            if slot_stamped:
                save_jobs(jobs)
            return False  # already parked at or past the window by another tick
        job["next_run_at"] = parked
        job[STATE_KEY] = parked
        save_jobs(jobs)
        return True

    return bool(_with_job(job_id, apply, missing=False))


def hold_due_jobs_for_provider_limits(due_jobs: list, cfg: dict[str, Any], fallback_chain) -> list:
    """Pre-fire gate: jobs whose provider is held at a usage limit are parked (wait) or skipped
    instead of firing into the wall. Jobs with a fallback chain, script-only jobs and one-shots
    run as before. Returns the jobs that may fire now."""
    from agent.usage_hold import active_hold, any_active_hold, format_until

    if not any_active_hold():
        return list(due_jobs)  # nothing held: skip per-job route resolution entirely
    runnable = []
    for job in due_jobs:
        hold = None
        schedule = job.get("schedule")
        kind = schedule.get("kind") if isinstance(schedule, dict) else None
        if not job.get("no_agent") and kind in {"cron", "interval"} and not fallback_chain(job, cfg):
            hold = active_hold(*_job_route(job, cfg))
        if hold is None:
            runnable.append(job)
            continue
        if _defer_for_provider_hold(job["id"], hold):
            logger.warning(
                "Job '%s': %s is at a usage limit until %s — %s instead of firing into it",
                job.get("name", job["id"]), hold.get("provider"), format_until(hold),
                "skipping occurrences" if job.get(ON_LIMIT_KEY) == "skip" else "waiting for the reset")
    return runnable
