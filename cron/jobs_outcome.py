"""Record a completed cron run without advancing its schedule."""

from typing import Any, Dict, Optional


def record_run_outcome(
    job: Dict[str, Any], success: bool, error: Optional[str], delivery_error: Optional[str],
    status: Optional[str], now: str,
) -> None:
    """Stamp one completed run onto *job*: status fields, failure streak, alert markers, claims."""
    job["last_run_at"] = now
    job.pop("manual_run_at", None)
    # The transient manual-run context is single-fire: the run that just completed consumed it.
    job.pop("manual_run_prompt", None)
    delivery_failed = isinstance(delivery_error, str) and bool(delivery_error.strip())
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
