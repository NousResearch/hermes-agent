"""Hold a job while its provider refuses it for billing/credits.

A credits or spending-limit refusal (xAI's 403 ``personal-team-blocked:spending-limit``, a 402)
names no reset time, so unlike ``cron/quota_hold.py`` there is no window to park past. Re-firing
on cadence fails the same way every tick (a refused request and an alert each time), and a
background run no longer continues on a local fallback model (``agent/fallback_local_billing.py``),
so the run ends on that wall. The failing run's one alert says the job is held; ``mark_job_run``
then re-probes no more often than every ``REPROBE_SECONDS`` (the schedule's own occurrences when
they are already that sparse, else the first occurrence past the backoff) and stamps
``quota_hold_until`` with the parked instant, so the stale-error re-arm leaves the job alone, plus
``PROVIDER_KEY``. A re-probe that is refused again stays silent; any run that reaches the model
clears both markers (``quota_hold.clear_state``), as for a quota hold.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, Optional

from cron import quota_hold
from hermes_time import now as _hermes_now, safe_strftime

logger = logging.getLogger("cron.scheduler")

PROVIDER_KEY = quota_hold.BILLING_PROVIDER_KEY
# A re-probe is one refused request, but a job every few minutes would still send one per tick.
REPROBE_SECONDS = 3600


def blocked_provider(exc: BaseException) -> Optional[str]:
    """The provider's name when the agent ended the run on a verified billing/credits wall, else
    None. Read from the turn's own verdict (``_final_response_from_result`` keeps it on the
    error), never from failure text; an unverified billing verdict (the same body can be a
    content-filter rejection) does not hold."""
    verdict = getattr(exc, "turn_verdict", None) or {}
    if verdict.get("failure_reason") != "billing" or verdict.get("billing_unverified"):
        return None
    block = verdict.get("billing_block") or {}
    return str(block.get("provider_label") or block.get("provider") or "the provider")


def repeat_alert(job: Dict[str, Any]) -> bool:
    """This run was refused for billing again while the job was already held; the alert went out
    when the hold began."""
    return bool(job.get("_billing_hold_provider") and job.get(PROVIDER_KEY))


def _parked_instant(job: Dict[str, Any], natural_next: datetime, now: datetime) -> datetime:
    from cron.jobs import _instant_before, _parse_aware, _schedule_cadence_seconds, _seconds_after, compute_next_run

    schedule = job.get("schedule") or {}
    earliest = _seconds_after(now, REPROBE_SECONDS)
    cadence = _schedule_cadence_seconds(schedule)
    if (cadence and cadence >= REPROBE_SECONDS) or not _instant_before(natural_next, earliest):
        return natural_next
    if schedule.get("kind") == "cron":
        # Coalesce to a legal occurrence so the stale-expression guard does not re-anchor it.
        return _parse_aware(compute_next_run(schedule, earliest.isoformat())) or earliest
    return earliest


def plan_hold(job: Dict[str, Any], provider: str) -> bool:
    """Called under the jobs lock AFTER ``_advance_after_run`` computed the natural
    ``next_run_at`` of a run refused for billing. Returns True when the job is held."""
    from cron.jobs import _parse_aware

    schedule = job.get("schedule") or {}
    natural_next = _parse_aware(job.get("next_run_at"))
    if schedule.get("kind") not in {"cron", "interval"} or job.get("state") == "paused" or natural_next is None:
        quota_hold.clear_state(job)
        return False
    parked = _parked_instant(job, natural_next, _hermes_now())
    if parked is not natural_next:
        job["next_run_at"] = parked.isoformat()
    job[quota_hold.STATE_KEY] = job["next_run_at"]
    job.pop(quota_hold.SCHEDULE_EXPR_KEY, None)
    job[PROVIDER_KEY] = provider
    logger.warning(
        "Job '%s': %s refused it for billing/credits — held, next check at %s",
        job.get("name", job.get("id", "?")), provider, job["next_run_at"])
    return True


def hold_notice(job: Dict[str, Any]) -> str:
    """Line appended to the ONE failure alert delivered on entering the hold, else ""."""
    provider = job.get("_billing_hold_provider")
    schedule = job.get("schedule") or {}
    if not provider or schedule.get("kind") not in {"cron", "interval"}:
        return ""
    from cron.jobs import _parse_aware, compute_next_run

    now = _hermes_now()
    natural_next = _parse_aware(compute_next_run(schedule, now.isoformat()))
    if natural_next is None:
        return ""
    check_at = safe_strftime(_parked_instant(job, natural_next, now), "%Y-%m-%d %H:%M %Z")
    declined = job.get("_declined_local_fallback")
    local = (
        f" It did not continue on the local fallback {declined} (set "
        "`fallback.background_local_when_billing_blocked: true` to allow that)." if declined else ""
    )
    return (
        f"\n⏸ Held: {provider} is out of credits.{local} After a top-up Hermes re-checks around "
        f"{check_at} (then at most hourly), resumes on its own and sends no further alerts while "
        f"held; changing the job's model releases the hold at once. "
        f"`hermes cron run {job.get('id')}` retries now."
    )
