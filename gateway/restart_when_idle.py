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


def queued_notice_units() -> list[dict]:
    """Completion notices already queued for delivery (a finished delegation or process whose
    notice the watcher has not injected yet): their turn has not started, but it will."""
    from tools.process_registry import process_registry
    queue = process_registry.completion_queue
    with queue.mutex:
        events = list(queue.queue)
    return [{"kind": "notice", "type": "async_delegation",
             "id": evt.get("delegation_id") or evt.get("session_id"),
             "session": evt.get("parent_session_id") or evt.get("session_key") or None}
            # Only delegation completions: process completion events are re-queued by the shared
            # watcher until their own per-process watcher (counted above) consumes them.
            for evt in events if isinstance(evt, dict) and evt.get("type") == "async_delegation"]


def background_units(runner) -> list[dict]:
    now = time.time()
    return delegation_units(now) + process_units(runner, now) + queued_notice_units()


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
                warn_every, runner._describe_active_work())
    loop = asyncio.get_running_loop()
    started = loop.time()
    next_warn = started + warn_every
    last_status = 0.0
    while not _idle(runner):
        now = loop.time()
        if now >= next_warn:
            logger.warning(
                "Restart still waiting after %.0fs; live work is NOT interrupted (only a forced "
                "restart stops it). Waiting on: %s", now - started, runner._describe_active_work())
            next_warn = now + max(warn_every, _STATUS_EVERY_S)
        if now - last_status >= _STATUS_EVERY_S:
            runner._scale_to_zero_status("draining", "restart wait: status mark failed")
            last_status = now
        try:
            await asyncio.wait_for(force.wait(), timeout=_POLL_S)
        except TimeoutError:
            continue
        logger.warning("Forced restart: stopping now and interrupting: %s", runner._describe_active_work())
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

    async def _run_restart() -> None:
        idle = await await_idle(runner)
        if not idle:
            force_stop_budgets(runner)
        # Detached helper only AFTER the wait, or its drain_timeout+5 deadline fires mid-turn.
        if detached:
            try:
                await runner._launch_detached_restart_command()
            except Exception:
                logger.error("Failed to launch detached gateway restart helper", exc_info=True)
        await asyncio.sleep(0.05)
        await runner.stop(restart=True, detached_restart=detached, service_restart=via_service)

    from contextvars import Context
    # NOT in _background_tasks (``_stop_impl`` cancels those, #12875); strong ref on the runner.
    # Empty Context: a copied one would run the HOST restart as the requesting profile.
    runner._restart_task = Context().run(lambda: asyncio.create_task(_run_restart()))
    return True


def force_stop_budgets(runner) -> None:
    """A forced restart does not drain: ``stop()`` interrupts turns, delegations and processes."""
    runner._restart_drain_timeout = 0.0
    runner._cron_drain_timeout = 0.0


def force_restart(runner, *, detached: bool = False, via_service: bool = True) -> bool:
    """Start a restart (or escalate a waiting one) that stops live work now. True when this call
    started or escalated it; False when ``stop()`` had already begun."""
    if getattr(runner, "_stop_task", None) is not None:
        return False
    if getattr(runner, "_restart_force_event", None) is None:
        runner._restart_force_event = asyncio.Event()
    runner._restart_force_event.set()
    force_stop_budgets(runner)
    request_restart(runner, detached=detached, via_service=via_service)
    return True
