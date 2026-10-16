"""Stamping one completed run's outcome onto its job record.

Split out of ``cron/jobs.py`` — which sits at its ``FILE_LINES`` cap, so the run-outcome rules,
including the neutral case below, had no room to be stated there. The store's own read/merge/write
semantics stay in ``cron.jobs``; this module only mutates the record it is handed, under that
caller's locks.
"""

from typing import Any, Optional


def _record_run_outcome(
    job: dict[str, Any], success: bool, error: Optional[str], delivery_error: Optional[str],
    status: Optional[str], now: str, neutral_tick: bool = False,
) -> None:
    """Stamp one completed run onto *job*: status fields, failure streak, alert markers, claims.

    ``neutral_tick`` marks a fire that produced no agent work — an unchanged monitor tick, reported
    as a success only because the TICK completed (``cron.monitor``). That success says nothing about
    the job's own work, so the failure state the last real run left behind — status, error, streak,
    alert markers, sticky failure stamp — is preserved exactly. Recording it as a plain success
    instead resolved the job's open incidents and reset ``failure_streak`` on every unchanged poll,
    which defeated ``cron.failure_repeat_alert_hours`` and the streak nudge for monitor-gated jobs
    (#140178).
    """
    job["last_run_at"] = now
    job.pop("manual_run_at", None)
    # The transient manual-run context is single-fire: the run that just completed consumed it.
    job.pop("manual_run_prompt", None)
    delivery_failed = isinstance(delivery_error, str) and bool(delivery_error.strip())
    if not neutral_tick:
        job["last_status"] = status or (
            "error" if not success else ("delivery_failed" if delivery_failed else "ok"))
        job["last_error"] = None if success else error
        if success:
            # Healthy run: drop the alert-once dedup markers so a FUTURE break re-alerts, and clear
            # the forward-failure stamp so it only describes CURRENT auto-fire health.
            job.pop("preflight_alerted", None)
            job.pop("last_fire_error", None)
            job["failure_streak"] = 0
        else:
            # Consecutive agent-failure streak; delivery failures do NOT count
            # (scheduler._failure_streak_nudge).
            job["failure_streak"] = int(job.get("failure_streak") or 0) + 1
            # Sticky last-failure stamp (#118354): the next success resets last_status and
            # failure_streak, which erases the only job-level trace that a run ever failed —
            # a monitor sampling jobs.json then sees a permanently green job. last_failure
            # survives success (latest failure wins); the recency window is the consumer's
            # call. Delivery failures keep their own sticky last_delivery_error.
            job["last_failure"] = {"at": now, "detail": error or (status or "run failed")}
        job["last_delivery_error"] = delivery_error
    # Clear both claims: the run is over, so the job is claimable again.
    job["fire_claim"] = None
    job.pop("pending_slot", None)
    if job.get("run_claim") is not None:  # keep key absence for legacy records
        job["run_claim"] = None
