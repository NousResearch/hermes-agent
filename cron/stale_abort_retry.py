"""One automatic re-run for cron fires aborted by the stale-call circuit breaker.

A provider that accepts the connection but goes silent (stalled stream, dead gateway
upstream) trips ``agent.chat_completion_helpers._check_stale_giveup``: after N
consecutive stale attempts (``HERMES_STREAM_STALE_GIVEUP``) it raises the RuntimeError
"Provider has been unresponsive (no response received) for N consecutive stale
attempts — aborting this call ...". ``cron/unreachable_retry.py`` only covers transient
network / DNS errors, so this abort used to sit out a full period: a weekly job that
aborted early in its window waited a week.

Deliberately narrower than the unreachable ladder: exactly ONE re-run per occurrence.
The breaker already made N attempts inside the run, so a re-run that aborts the same
way is evidence the provider is still down; it alerts normally and the job waits for
its natural occurrence (which gets a fresh single re-run). The re-run lands 30–60
minutes out, jittered so a recovering provider is not hit by every job at once.

Spend / side-effect safety follows the sibling modules: the run must have FAILED on the
exact signature AND completed zero API calls. Recurring jobs only (one-shots are
pre-claimed, at-most-times dispatch accounting, #38758). The re-run never fires past
the schedule's own next occurrence: when that lands within the window, the natural
occurrence IS the retry. Like an unreachable ladder rung, the re-run instant is off the
cron lattice, so it carries ``at``/``expr`` and ``cron.jobs._reanchor_stale_cron``
recognises it (``is_retry_fire``) instead of re-anchoring it away without firing; it
does not count toward ``repeat`` again (``is_retry_run`` -> ``ladder_rung``).
Disable with ``cron.retry_stale_abort: false`` in config.yaml.
"""

from __future__ import annotations

import logging
import random
from datetime import datetime
from typing import Any, Optional

from cron.constants import is_recurring
from hermes_time import now as _hermes_now

logger = logging.getLogger("cron.scheduler")

# The re-run lands in this jittered window.
RETRY_MIN_SECONDS = 30 * 60
RETRY_MAX_SECONDS = 60 * 60

# Persisted while the re-run is pending: {"attempt": 1, "at": <ISO instant>, "expr": <the
# cron expression it was planned under>}. Cleared by any run that reaches the model and by
# the re-run itself when it aborts again.
STATE_KEY = "stale_abort_retry"

# The exact _check_stale_giveup signature. Only matched on our own RuntimeError text,
# never on provider payloads.
_ABORT_FRAGMENTS = ("consecutive stale attempts", "aborting this call")


def retry_enabled(cfg: Optional[dict] = None) -> bool:
    """``cron.retry_stale_abort`` — default ON (spend-neutral: only fires when zero model
    calls completed). Mirrors ``cron.unreachable_retry.retry_enabled``."""
    if cfg is None:
        try:
            from hermes_cli.config import load_config

            cfg = load_config() or {}
        except Exception:  # config unreadable — keep the reliability default
            logger.debug("cron.retry_stale_abort: config unreadable, keeping default", exc_info=True)
            return True
    cron_cfg = (cfg or {}).get("cron")
    if not isinstance(cron_cfg, dict):
        return True
    return cron_cfg.get("retry_stale_abort") is not False


def is_stale_abort_failure(exc: BaseException, agent: Any = None) -> bool:
    """True when *exc* is exactly the stale-breaker give-up RuntimeError and *agent* (may be
    ``None``) never completed a model call."""
    if type(exc) is not RuntimeError:
        return False
    text = str(exc)
    if not all(fragment in text for fragment in _ABORT_FRAGMENTS):
        return False
    return int(getattr(agent, "session_api_calls", 0) or 0) <= 0


def _applies(job: dict[str, Any]) -> bool:
    return is_recurring(job) and job.get("state") != "paused" and retry_enabled()


def _retry_instant(job: dict[str, Any], natural_next: Optional[datetime],
                   now: datetime) -> Optional[datetime]:
    """The re-run instant ``plan_retry`` parks, or None when it parks nothing: not
    applicable, this occurrence already spent its re-run, or the natural occurrence lands
    within the re-run window. The single source of that decision for ``plan_retry`` and
    ``will_retry``."""
    if job.get(STATE_KEY) or not _applies(job):
        return None
    from cron.jobs import _instant_at_or_before, _seconds_after  # late: jobs imports us

    if natural_next is not None and _instant_at_or_before(
            natural_next, _seconds_after(now, RETRY_MAX_SECONDS)):
        return None  # the natural occurrence is the retry
    return _seconds_after(now, random.uniform(RETRY_MIN_SECONDS, RETRY_MAX_SECONDS))


def will_retry(job: dict[str, Any]) -> bool:
    """True iff ``plan_retry`` will park a re-run for this flagged failure, so the scheduler
    may hold the interim failure notice. Called before ``mark_job_run``."""
    repeat = job.get("repeat") or {}
    times = repeat.get("times")
    counted = 0 if is_retry_run(job) else 1
    if times is not None and times > 0 and int(repeat.get("completed") or 0) + counted >= times:
        return False  # _advance_after_run completes the job; plan_retry never runs
    from cron.jobs import _parse_aware, compute_next_run

    now = _hermes_now()
    natural_next = _parse_aware(compute_next_run(job.get("schedule") or {}, now.isoformat()))
    if natural_next is None:
        return False  # uncomputable natural occurrence leaves the record terminal
    return _retry_instant(job, natural_next, now) is not None


def clear_state(job: dict[str, Any]) -> None:
    """A run reached the model (any outcome): the pending re-run is spent."""
    job.pop(STATE_KEY, None)


def is_retry_fire(job: dict[str, Any], next_run: str) -> bool:
    """True for the exact off-lattice instant parked by ``plan_retry`` (see
    ``cron.unreachable_retry.is_retry_fire``); the expression fingerprint keeps a direct
    ``jobs.json`` schedule edit from inheriting the exception."""
    state = job.get(STATE_KEY) or {}
    return bool(state.get("at")) and state.get("at") == next_run and \
        state.get("expr") == (job.get("schedule") or {}).get("expr")


def is_retry_run(job: dict[str, Any]) -> bool:
    """True when this dispatch snapshot fired the re-run instant: a re-run of an occurrence
    that already counted toward ``repeat``. Manual runs carry no ``_scheduled_instant``."""
    from cron.occurrences import scheduled_instant

    instant = job.get("_scheduled_instant")
    at = (job.get(STATE_KEY) or {}).get("at")
    return instant is not None and at is not None and scheduled_instant(at) == instant


def plan_retry(job: dict[str, Any]) -> bool:
    """Called under the jobs lock AFTER ``_advance_after_run`` computed the natural
    ``next_run_at`` for a failed, flagged run. Pulls ``next_run_at`` earlier to the jittered
    re-run instant when that is sooner than the natural occurrence. A re-run that aborted
    again clears the state, so the next natural occurrence gets its own single re-run.
    Returns True when a re-run was scheduled."""
    from cron.jobs import _parse_aware  # late: jobs imports this module's helpers

    if job.get(STATE_KEY):
        logger.warning(
            "Job '%s': stale-provider abort again on the automatic re-run — waiting for the "
            "scheduled occurrence at %s", job.get("name", job.get("id", "?")),
            job.get("next_run_at"))
        clear_state(job)
        return False
    retry_dt = _retry_instant(job, _parse_aware(job.get("next_run_at")), _hermes_now())
    if retry_dt is None:
        return False
    retry_at = retry_dt.isoformat()
    job[STATE_KEY] = {"attempt": 1, "at": retry_at, "expr": (job.get("schedule") or {}).get("expr")}
    job["next_run_at"] = retry_at
    job["state"] = "scheduled"
    logger.info(
        "Job '%s': stale-provider abort with zero API calls — one automatic re-run at %s",
        job.get("name", job.get("id", "?")), retry_at)
    return True
