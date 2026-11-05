"""Headless tick gate: agent jobs need a live gateway; a headless tick never starts one.

``run_canonical_job`` reaches the owner through ``connect_gateway`` → ``ensure_gateway_runtime``,
which spawns an unmanaged daemon when nothing owns the profile. That is right for an interactive
caller (``hermes cron run``) and wrong for ``hermes cron tick`` from a system crontab: every tick
would leave a stray gateway behind. Headless surfaces refuse rather than spawn (same policy as
approvals), so a headless tick skips its agent jobs while the gateway is down and lets them fire
on the next tick once it is up — no run is recorded and the due instant does not move. The
in-process ticker never reaches this gate: it runs inside the gateway by definition.

"Down" is every outcome but ``ready``: a gateway still ``starting`` (or draining, or one whose
control socket does not answer) would otherwise send the tick into ``ensure_gateway_runtime``'s
30 s wait and book the deadline as a FAILED run, consuming the slot — the crontab minute is the
budget, and a slot lost to a restart window is a missed fire, not a job failure.
"""
import logging
import os

logger = logging.getLogger(__name__)

GATEWAY_DOWN_HINT = (
    "the gateway is not running, so agent jobs cannot fire; they run at the next tick once it is "
    "up. Start it with `hermes gateway start` (or `hermes gateway install` for a service)."
)
# A crontab tick has a minute; a gateway still starting answers within a few seconds or not at all.
_HEADLESS_PROBE_TIMEOUT_S = 5.0
# Three missed 60s gateway ticker iterations and the headless tick stops deferring to it.
_DEFER_TICKER_FRESH_S = 180.0


def _gateway_unavailable(home) -> str | None:
    """Why a headless tick must hold its agent jobs, or None when a ready owner serves *home*.
    Everything short of ``ready`` — absent (``ensure`` would spawn), starting/draining (``ensure``
    would wait a whole deadline then fail the run), inaccessible/incompatible (no owner answers) —
    is held, never booked as a run."""
    from hermes_cli.gateway_runtime import discover_gateway_endpoint

    if os.environ.get("HERMES_TUI_GATEWAY_URL", "").strip():
        return None  # explicit remote gateway: connect_gateway never spawns for it
    observed = discover_gateway_endpoint(home, timeout=_HEADLESS_PROBE_TIMEOUT_S)
    if observed.state == "ready":
        return None
    if observed.state == "absent":
        return GATEWAY_DOWN_HINT
    reason = observed.state + (f" ({observed.reason_code})" if observed.reason_code else "")
    return (f"the gateway is {reason}, so agent jobs cannot fire; they run at the next tick once it "
            "is ready. Check it with `hermes gateway status`.")


def _note_fire_refused(job_ids, detail: str | None) -> None:
    """Stamp ``last_fire_error`` (shown by ``hermes cron list``; skipped when ``detail`` is None)
    and drop the dispatch marks the due scan just wrote: the slot was never handed over, so nothing
    must "restore" or "reclaim" it."""
    from cron.jobs import _hermes_now, _jobs_lock, load_jobs, save_jobs

    ids = set(job_ids)
    with _jobs_lock():
        jobs = load_jobs()
        for job in jobs:
            if job["id"] not in ids:
                continue
            if detail is not None:
                job["last_fire_error"] = {"at": _hermes_now().isoformat(), "detail": detail[:500]}
            job.pop("pending_slot", None)
            if job.get("run_claim") is not None:
                job["run_claim"] = None
        save_jobs(jobs)


def _defer_agent_jobs_to_gateway(home) -> bool:
    """True when ``cron.headless_defer_agent_jobs`` (env HERMES_CRON_HEADLESS_DEFER_AGENT_JOBS) is
    on and another live gateway's in-process ticker serves *home*. That ticker then owns every
    agent job, so its ``max_parallel_jobs`` pool is the one host-wide agent limit: a crontab tick
    that also fired agent jobs would run them beside the gateway's, past that limit."""
    from cron.env_settings import cron_env_setting

    raw = cron_env_setting("HERMES_CRON_HEADLESS_DEFER_AGENT_JOBS").strip().lower()
    if not raw:
        from hermes_cli.config import load_config_readonly
        raw = str((load_config_readonly().get("cron") or {}).get(
            "headless_defer_agent_jobs", "")).strip().lower()
    if raw not in {"1", "true", "yes", "on"}:
        return False
    from cron.scheduler_ownership import live_gateway_ticking
    record = live_gateway_ticking(home)
    if record is None:
        return False
    # A healthy gateway is not proof its ticker fires: the loop may be dead, backing off or
    # yielding. Defer only while that process's own heartbeat and success markers are fresh.
    # (``ticker_last_success`` names no writer; the heartbeat's pid is the ownership proof.)
    from cron.jobs import _read_marker_fields, get_ticker_heartbeat_age, get_ticker_success_age
    fields = _read_marker_fields("ticker_heartbeat")
    if len(fields) < 2 or fields[1] != str(record.get("pid")):
        return False
    beat, success = get_ticker_heartbeat_age(), get_ticker_success_age()
    return (beat is not None and beat < _DEFER_TICKER_FRESH_S
            and success is not None and success < _DEFER_TICKER_FRESH_S)


def refuse_agent_jobs_without_gateway(due_jobs: list) -> list:
    """Return the due jobs a headless tick may dispatch; agent jobs are held back (skipped, not
    failed) when no ready gateway serves this home, or left silently to that gateway's ticker
    when ``_defer_agent_jobs_to_gateway``. Logs once per tick, not once per job."""
    from hermes_constants import get_hermes_home

    agent_jobs = [job for job in due_jobs if not job.get("no_agent")]
    if not agent_jobs:
        return due_jobs
    home = get_hermes_home().resolve()
    if _defer_agent_jobs_to_gateway(home):
        ids = [str(job["id"]) for job in agent_jobs]
        _note_fire_refused(ids, None)  # not a failure: the gateway fires them on its next tick
        logger.info("Cron tick for %s left %d agent job(s) to the gateway ticker (%s)",
                    home, len(ids), ", ".join(ids))
        return [job for job in due_jobs if job.get("no_agent")]
    hint = _gateway_unavailable(home)
    if hint is None:
        return due_jobs
    ids = [str(job["id"]) for job in agent_jobs]
    _note_fire_refused(ids, f"Skipped: {hint}")
    logger.warning(
        "Cron tick for %s skipped %d agent job(s) (%s): %s",
        home, len(ids), ", ".join(ids), hint)
    return [job for job in due_jobs if job.get("no_agent")]
