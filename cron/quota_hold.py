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

# At most one early-release probe per route in this window. The tick runs every minute, and a
# provider without its own probe throttle would otherwise be asked once per tick for days.
RELEASE_PROBE_INTERVAL_SECONDS = 300
_release_probe_at: Dict[tuple, float] = {}

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
    job: Dict[str, Any], natural_next: datetime, window_end: datetime,
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
    recover = (kind == "cron" and not blocked and recover_consumed_fire
               and _recovery_worthwhile(job, natural_next, window_end))
    if not blocked and not recover:
        clear_state(job)
        return False
    if kind == "cron" and blocked:
        # Coalesce cron occurrences inside the closed window to the first legal instant after it.
        parked = compute_next_run(schedule, window_end.isoformat()) or window_end.isoformat()
    else:
        parked = window_end.isoformat()
    if recover:
        # Only the recovery fire is off-lattice; the coalesced instant is a legal occurrence.
        job[SCHEDULE_EXPR_KEY] = schedule.get("expr")
    else:
        job.pop(SCHEDULE_EXPR_KEY, None)
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


def _quota_reopened(runtime: Any) -> bool:
    """True only when the resolved route's usage window is known to be open again.

    A successful resolve proves the credentials work, not that the window reopened: a Codex
    login stays valid while its quota is exhausted (the hold's own alert says "Credentials are
    still valid"), so a singleton token resolves with the window still shut. For Codex, ask
    the usage endpoint the credential pool already trusts (``_probe_codex_quota_restored``):
    closed or unknown keeps the hold. Other providers have no window probe; for them the
    resolve itself is the signal, and a still-closed window raises the quota error there."""
    if not isinstance(runtime, dict) or runtime.get("provider") != "openai-codex":
        return True
    from hermes_cli.auth import _probe_codex_quota_restored

    return _probe_codex_quota_restored(
        runtime.get("api_key"), base_url=runtime.get("base_url")) is True


def _reopened_holds(held: list) -> Dict[str, str]:
    """``job id -> held instant`` for held jobs whose primary route has its window open again.

    Same primary resolve the run itself starts with (``_resolve_job_runtime``), then the
    route's usage check (``_quota_reopened``); one probe per route and at most one per
    ``RELEASE_PROBE_INTERVAL_SECONDS``. Any failure or unknown answer keeps the hold. Runs
    under the profile's secret scope because the tick holds none and ``get_secret`` fails
    closed under multiplex."""
    import time

    from agent.secret_scope import build_profile_secret_scope, reset_secret_scope, set_secret_scope
    from cron import scheduler as _sched
    from hermes_cli.env_loader import hydrate_profile_secret_sources
    from hermes_cli.runtime_provider import resolve_runtime_provider
    from hermes_constants import hermes_home_key

    home = _sched._get_hermes_home().resolve()
    now = time.monotonic()
    home_key = hermes_home_key(home)
    hydrate_profile_secret_sources(home)
    scope_token = set_secret_scope(build_profile_secret_scope(home), profile_home=str(home))
    verdicts: Dict[tuple, bool] = {}
    reopened: Dict[str, str] = {}
    try:
        for job in held:
            try:
                jc = _sched._load_cron_job_config(job, job["id"], job.get("name", job["id"]))
            except Exception:
                continue
            route = (job.get("provider") or jc.cron_default_provider or None, jc.model,
                     job.get("base_url") or None)
            if route not in verdicts:
                throttle_key = (home_key, *route)
                last = _release_probe_at.get(throttle_key)
                if last is not None and now - last < RELEASE_PROBE_INTERVAL_SECONDS:
                    verdicts[route] = False
                    continue
                _release_probe_at[throttle_key] = now
                kwargs = {"requested": route[0], "target_model": route[1]}
                if route[2]:
                    kwargs["explicit_base_url"] = route[2]
                try:
                    verdicts[route] = _quota_reopened(resolve_runtime_provider(**kwargs))
                except Exception:
                    verdicts[route] = False
            if verdicts[route]:
                reopened[job["id"]] = job[STATE_KEY]
    finally:
        reset_secret_scope(scope_token)
    return reopened


def release_reopened_holds() -> int:
    """Release holds whose provider's usage window is open again; returns the number released.

    The announced reset is an upper bound: Codex can reopen days earlier (a banked reset, a plan
    change, a rotated or re-added account), and nothing else lifts a hold before its instant.
    Probes only while a hold is active. A released interval job is due now; a cron job moves to
    its next legal occurrence, except a sparse cron parked on its recovery fire, which keeps
    that one retry and takes it now."""
    from cron.jobs import _jobs_lock, compute_next_run, load_jobs, save_jobs

    now = _hermes_now()
    with _jobs_lock():
        held = [dict(j) for j in load_jobs()
                if j.get("enabled", True) and j.get("state") != "paused" and j.get("id")
                and hold_active(j, now)]
    if not held:
        return 0
    reopened = _reopened_holds(held)
    if not reopened:
        return 0
    released = 0
    with _jobs_lock():
        jobs = load_jobs()
        for job in jobs:
            # A run that finished between the probe and this write already moved the job.
            if job.get("id") not in reopened or job.get(STATE_KEY) != reopened[job["id"]]:
                continue
            schedule = job.get("schedule") or {}
            if is_recovery_fire(job, job.get("next_run_at") or ""):
                # Keep the markers on the new instant so the due scan still treats it as the
                # authorized off-lattice retry (``cron.jobs._reanchor_stale_cron``).
                job["next_run_at"] = job[STATE_KEY] = now.isoformat()
            else:
                clear_state(job)
                job["next_run_at"] = (
                    now.isoformat() if schedule.get("kind") == "interval"
                    else compute_next_run(schedule, now.isoformat()) or job["next_run_at"])
            released += 1
            logger.info(
                "Job '%s': provider usage window open again before its announced reset; "
                "releasing the quota hold, next run %s", job.get("name", job.get("id")), job["next_run_at"])
        if released:
            save_jobs(jobs)
    return released
