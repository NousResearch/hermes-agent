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
# Grace window after a supervised gateway exits with gateway_state="running" before launchd/systemd
# spawns the replacement PID and it acquires gateway.lock.
_SUPERVISOR_RESPAWN_GRACE_S = 120.0


def _supervised_gateway_respawning(home) -> bool:
    """True when ``gateway_state.json`` in *home* (or its parent multiplexer root when *home* is
    ``<root>/profiles/<name>``) records an active/restarting gateway updated within
    ``_SUPERVISOR_RESPAWN_GRACE_S`` while no live PID currently answers discovery (the
    launchd/systemd respawn throttle gap after SIGTERM)."""
    import datetime as dt
    import json
    from pathlib import Path

    home_path = Path(home).resolve()
    candidates = [home_path]
    if home_path.parent.name == "profiles":
        candidates.append(home_path.parent.parent)
    now = dt.datetime.now(dt.timezone.utc)
    for cand in candidates:
        state_file = cand / "gateway_state.json"
        try:
            if not state_file.is_file():
                continue
            data = json.loads(state_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        if data.get("gateway_state") != "running" and not data.get("restart_requested"):
            continue
        updated_raw = data.get("updated_at")
        if not isinstance(updated_raw, str) or not updated_raw.strip():
            continue
        try:
            updated_dt = dt.datetime.fromisoformat(updated_raw.replace("Z", "+00:00"))
            if updated_dt.tzinfo is None:
                updated_dt = updated_dt.astimezone(dt.timezone.utc)
            age_s = (now - updated_dt).total_seconds()
            if 0 <= age_s <= _SUPERVISOR_RESPAWN_GRACE_S:
                return True
        except Exception:
            continue
    return False


def _gateway_transitioning(home) -> str | None:
    """Why a non-owner in-process ticker (e.g. Desktop/serve fallback) must hold its agent jobs
    during a gateway restart/drain window, or None when it may proceed."""
    from gateway.session_cron import owner_for_home
    from hermes_cli.gateway_runtime import discover_gateway_endpoint

    if owner_for_home(home) is not None:
        return None
    if os.environ.get("HERMES_TUI_GATEWAY_URL", "").strip():
        return None
    observed = discover_gateway_endpoint(home, timeout=_HEADLESS_PROBE_TIMEOUT_S)
    if observed.state == "ready":
        return None
    if observed.state in ("starting", "draining"):
        reason = observed.state + (f" ({observed.reason_code})" if observed.reason_code else "")
        return (f"the gateway is {reason}, so agent jobs cannot fire; they run at the next tick once "
                "it is ready. Check it with `hermes gateway status`.")
    if observed.state == "absent" and _supervised_gateway_respawning(home):
        return ("the supervised gateway is restarting (respawn window), so agent jobs are held until "
                "it is ready. Check it with `hermes gateway status`.")
    return None


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


def _note_fire_refused(job_ids, detail: str) -> None:
    """Stamp ``last_fire_error`` (shown by ``hermes cron list``) and drop the dispatch marks the
    due scan just wrote: the slot was never handed over, so nothing must "restore" or "reclaim" it."""
    from cron.jobs import _hermes_now, _jobs_lock, load_jobs, save_jobs

    ids = set(job_ids)
    with _jobs_lock():
        jobs = load_jobs()
        for job in jobs:
            if job["id"] not in ids:
                continue
            job["last_fire_error"] = {"at": _hermes_now().isoformat(), "detail": detail[:500]}
            job.pop("pending_slot", None)
            if job.get("run_claim") is not None:
                job["run_claim"] = None
        save_jobs(jobs)


def refuse_agent_jobs_without_gateway(due_jobs: list, *, transitioning_only: bool = False) -> list:
    """Return the due jobs a tick may dispatch; agent jobs are held back (skipped, not
    failed) when no ready gateway serves this home (or, when ``transitioning_only=True``, when a
    non-owner ticker sees the gateway mid-restart/starting/draining). Logs once per tick."""
    from hermes_constants import get_hermes_home

    agent_jobs = [job for job in due_jobs if not job.get("no_agent")]
    if not agent_jobs:
        return due_jobs
    home = get_hermes_home().resolve()
    hint = _gateway_transitioning(home) if transitioning_only else _gateway_unavailable(home)
    if hint is None:
        return due_jobs
    ids = [str(job["id"]) for job in agent_jobs]
    _note_fire_refused(ids, f"Skipped: {hint}")
    logger.warning(
        "Cron tick for %s skipped %d agent job(s) (%s): %s",
        home, len(ids), ", ".join(ids), hint)
    return [job for job in due_jobs if job.get("no_agent")]
