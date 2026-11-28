"""A requested graceful restart waits for the gateway to go idle, serving normally meanwhile.

SIGUSR1, ``/restart``, ``hermes gateway restart`` and the updater's ``pause-for-update`` used to put
the gateway in ``draining`` at once: every new turn, continuation, process/watch notice and
background-delegation completion was refused ``runtime_draining`` for up to
``agent.restart_after_turn_timeout``, and at that cap ``stop()`` interrupted whatever was still live
(turns, delegations, tool subprocesses). Parents could not hear back from their children for the
whole wait, then lost them.

Now the restart is "restart when idle": nothing is refused while work is in flight, the restart
happens at the first moment no work is (background delegations, notify-watched background
processes and queued completion notices count, since each delivers a new turn later), and the cap
only means "warn and name what is still running". Only an explicit force (``pause-for-update`` with
``{"force": true}``) stops live work; SIGTERM keeps its immediate-stop meaning.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time

from gateway.shutdown_watchdog import recover_failed_restart_stop
from hermes_constants import get_hermes_home

logger = logging.getLogger("gateway.run")

# How often the wait re-checks for idleness and refreshes the published ``active_work``.
_POLL_S = 0.1
_STATUS_EVERY_S = 30.0


def delegation_units(now: float) -> list[dict]:
    """One unit per live background delegation, with the parent session it reports back to."""
    from tools.async_delegation import _LIVE_STATES, list_async_delegations
    units = []
    for record in list_async_delegations():
        if record.get("status") not in _LIVE_STATES:
            continue
        unit = {"kind": "delegation", "delegation_id": record.get("delegation_id"),
                "goal": str(record.get("goal") or "")[:80], "pid": os.getpid(),
                "session": record.get("parent_session_id") or record.get("session_key") or None}
        if record.get("dispatched_at"):
            unit["elapsed_s"] = round(now - float(record["dispatched_at"]), 1)
        units.append(unit)
    return units


def process_units(runner, now: float) -> list[dict]:
    """Background processes whose completion will start a new turn: a live notify watcher
    (``_run_process_watcher``) is still waiting on each."""
    from tools.process_registry import process_registry
    units = []
    for proc_id in sorted(getattr(runner, "_live_process_watchers", ()) or ()):
        session = process_registry.get(proc_id)
        unit = {"kind": "process", "process_id": proc_id}
        if session is not None:
            unit.update(pid=session.pid, command=str(session.command or "")[:80],
                        session=session.session_key or None, exited=bool(session.exited))
            if session.started_at:
                unit["elapsed_s"] = round(now - session.started_at, 1)
        units.append(unit)
    return units


def queued_notice_units(runner) -> list[dict]:
    """Completion notices queued for delivery, or dequeued by the watcher and not yet admitted
    (``_async_notices_in_delivery``): their turn has not started, but it will. One the watcher found
    no transport for here (``_no_transport``: a disabled or unavailable platform) is not counted: it
    would hold the restart forever, and its ledger row replays it after the restart."""
    from tools.process_registry import process_registry
    queue = process_registry.completion_queue
    with queue.mutex:
        events = list(queue.queue)
    events += list(getattr(runner, "_async_notices_in_delivery", ()) or ())
    return [{"kind": "notice", "type": "async_delegation",
             "id": evt.get("delegation_id") or evt.get("session_id"),
             "session": evt.get("parent_session_id") or evt.get("session_key") or None}
            # Only delegation completions: process completion events are re-queued by the shared
            # watcher until their own per-process watcher (counted above) consumes them.
            for evt in events if isinstance(evt, dict) and evt.get("type") == "async_delegation"
            and not evt.get("_no_transport")]


def background_units(runner) -> list[dict]:
    now = time.time()
    return delegation_units(now) + process_units(runner, now) + queued_notice_units(runner)


def _log_view(units: list[dict]) -> str:
    """Units for INFO/WARNING logs: kind, id and session only; goals and commands are user content."""
    keys = ("session", "delegation_id", "process_id", "id", "job_id", "pid")
    return "; ".join(f"{u.get('kind')}(" + ", ".join(f"{k}={u[k]}" for k in keys if u.get(k)) + ")"
                     for u in units if isinstance(u, dict)) or "nothing"


def _idle(runner) -> bool:
    """No awaitable foreground work and no background work that will deliver a turn later."""
    return runner._awaitable_work_count() <= 0 and not background_units(runner)


async def await_idle(runner) -> bool:
    """Keep serving until the gateway is idle (True) or a force arrives (False).

    ``agent.restart_after_turn_timeout`` is a warning interval: each time it elapses with work
    still live, a WARNING names every unit and the wait continues. Wedged chat turns (no activity
    past ``agent.gateway_timeout``) and restart-safe external cron runs never hold it.
    """
    force = getattr(runner, "_restart_force_event", None)
    if force is None:
        force = runner._restart_force_event = asyncio.Event()
    if _idle(runner) or force.is_set():
        return not force.is_set()
    warn_every = float(getattr(runner, "_restart_after_turn_timeout", 0.0) or 0.0)
    logger.info("Restart requested; waiting for idle while serving normally (warn after %.0fs): %s",
                warn_every, _log_view(runner._describe_active_work()))
    loop = asyncio.get_running_loop()
    started = loop.time()
    next_warn = started + warn_every
    last_status = 0.0
    while not _idle(runner):
        now = loop.time()
        if now >= next_warn:
            logger.warning(
                "Restart still waiting after %.0fs; live work is NOT interrupted (only a forced "
                "restart stops it). Waiting on: %s", now - started, _log_view(runner._describe_active_work()))
            next_warn = now + max(warn_every, _STATUS_EVERY_S)
        if now - last_status >= _STATUS_EVERY_S:
            runner._scale_to_zero_status("draining", "restart wait: status mark failed")
            last_status = now
        try:
            await asyncio.wait_for(force.wait(), timeout=_POLL_S)
        except TimeoutError:
            continue
        logger.warning("Forced restart: stopping now and interrupting: %s", _log_view(runner._describe_active_work()))
        return False
    logger.info("Restart wait complete after %.1fs: gateway idle; restarting", loop.time() - started)
    return True


def request_restart(runner, *, detached: bool, via_service: bool) -> bool:
    """Schedule the restart-when-idle task (idempotent). Admission stays open until ``stop()``."""
    if runner._restart_task_started:
        return False
    runner._restart_requested = True
    runner._restart_detached = detached
    runner._restart_via_service = via_service
    runner._restart_task_started = True
    if getattr(runner, "_restart_force_event", None) is None:
        runner._restart_force_event = asyncio.Event()

    in_stop = []

    async def _run_restart() -> None:
        try:
            await _restart_when_idle()
        except Exception:
            # Nothing else awaits this task, so its failure is logged here, except one raised inside its
            # own stop() after the stop's guard was entered: the guard logged that one and owns the exit.
            if getattr(runner, "_stop_task", None) is not None:
                if not in_stop:
                    logger.warning("Restart wait failed while the gateway was already stopping", exc_info=True)
                elif not getattr(runner, "_stop_guard_entered", False):
                    if runner._restart_via_service:
                        recover_failed_restart_stop(runner)
                    else:
                        logger.exception("Restart stop failed before its teardown began; needs an external restart")
                return
            # No stop has begun: keep serving and let the next restart request run.
            logger.exception("Restart failed before the gateway began stopping; still serving, and the "
                             "next restart request will run")
            if getattr(runner, "_detached_restart_helper_started", False):
                logger.warning("The detached restart helper launched for this restart still runs and will "
                               "restart the gateway; this process launches no second helper, so if that "
                               "restart fails, a later detached /restart stops the gateway without relaunching it")
            runner._draining = False
            runner._restart_task_started = runner._restart_requested = False
            # Drop the failed restart's plan too: a later plain stop would otherwise exit 75 and be relaunched.
            runner._restart_detached = runner._restart_via_service = False
            chat_restart = getattr(runner, "_restart_command_source", None) is not None
            runner._restart_command_source = None  # a /restart's requester: the next restart is not theirs
            runner._restart_force_event = None
            budgets = runner.__dict__.pop("_budgets_before_force", None)
            if budgets is not None:  # the force expected a stop that never came: drain as before it
                runner._restart_drain_timeout, runner._cron_drain_timeout = budgets
            if not getattr(runner, "_external_drain_active", False):  # an operator's drain still holds
                runner._scale_to_zero_status(runner._serving_state(), "restart failure: status restore failed")
            if chat_restart:  # nor is its notice; best-effort, after the recovery (a read-only home)
                try:
                    (get_hermes_home() / ".restart_notify.json").unlink(missing_ok=True)
                except OSError:
                    logger.warning("Could not remove the failed /restart's notice", exc_info=True)

    async def _restart_when_idle() -> None:
        idle = await await_idle(runner)
        # Close admission in the same loop tick the wait saw idle: a turn admitted during the helper
        # launch or the yield below would otherwise be interrupted by a drain-less stop().
        runner._draining = True
        if not idle:
            force_stop_budgets(runner)
        # Read the plan now, not at request time: a forced update pause replaces a /restart's relaunch.
        detached, via_service = runner._restart_detached, runner._restart_via_service
        # Detached helper only AFTER the wait, or its drain_timeout+5 deadline fires mid-turn.
        if detached:
            try:
                await runner._launch_detached_restart_command()
            except Exception:
                logger.error("Failed to launch detached gateway restart helper", exc_info=True)
        await asyncio.sleep(0.05)
        in_stop.append(True)
        await runner.stop(restart=True, detached_restart=detached, service_restart=via_service)

    from contextvars import Context
    # NOT in _background_tasks (``_stop_impl`` cancels those, #12875); strong ref on the runner.
    # Empty Context: a copied one would run the HOST restart as the requesting profile.
    runner._restart_task = Context().run(lambda: asyncio.create_task(_run_restart()))
    return True


def force_stop_budgets(runner) -> None:
    """A forced restart does not drain: ``stop()`` interrupts turns, delegations and processes."""
    runner.__dict__.setdefault("_budgets_before_force", (runner._restart_drain_timeout, runner._cron_drain_timeout))
    runner._restart_drain_timeout = 0.0
    runner._cron_drain_timeout = 0.0


def request_service_restart(runner) -> bool:
    """An ordinary update pause: restart when idle and exit for the service (exit 75). A restart
    already waiting (a ``/restart`` that captured a detached helper) is adopted: its plan becomes
    the pause's, so the waiting gateway does not launch that helper mid-update and the updater
    hears ``pausing`` instead of stopping a gateway that is still doing work. False only once the
    stop has begun (``_run_restart`` closes admission in the tick it reads the plan)."""
    if request_restart(runner, detached=False, via_service=True):
        return True
    if getattr(runner, "_stop_task", None) is not None or getattr(runner, "_draining", False):
        return False
    runner._restart_detached, runner._restart_via_service = False, True
    return True


def force_restart(runner, *, detached: bool = False, via_service: bool = True) -> bool:
    """Start a restart (or escalate a waiting one) that stops live work now. True when this call
    started or escalated it; False when ``stop()`` had already begun."""
    if getattr(runner, "_stop_task", None) is not None:
        return False
    if getattr(runner, "_restart_force_event", None) is None:
        runner._restart_force_event = asyncio.Event()
    runner._restart_force_event.set()
    force_stop_budgets(runner)
    if not request_restart(runner, detached=detached, via_service=via_service):
        # A restart is already waiting: the force's relaunch plan replaces the one it captured
        # (an update pause must not launch a /restart's detached helper mid-update).
        runner._restart_detached, runner._restart_via_service = detached, via_service
    return True
