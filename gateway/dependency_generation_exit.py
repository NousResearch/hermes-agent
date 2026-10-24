"""Retire a supervised gateway after dependency publication without a forced-turn deadline."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from gateway.drain_control import open_restart_wait
from pm.generation_lifecycle import GenerationObserver

log = logging.getLogger(__name__)
POLL_S = 30.0
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


async def dependency_generation_watcher(
    runner, *, observer=None, poll_s=POLL_S, max_polls=None, owned=None
):
    if owned is None:
        from gateway.restart import is_gateway_supervisor_process

        owned = is_gateway_supervisor_process()
    if not owned:
        return
    if observer is None:
        observer = await asyncio.to_thread(GenerationObserver, _PROJECT_ROOT)
    seen = polls = 0
    draining = getattr(runner, "_dependency_restart_pending", False)
    if draining:
        open_restart_wait(runner)
        runner._mark_api_runs_shutdown_requested()
    while runner._running and (max_polls is None or polls < max_polls):
        polls += 1
        reason = await asyncio.to_thread(observer.probe)
        if not runner._running or runner._restart_task_started:
            return  # an operator's stop or restart owns the handoff
        seen = seen + 1 if reason else 0
        if not draining and seen >= 2 and not runner._draining:
            runner._dependency_restart_pending = True
            runner._draining = draining = True
            open_restart_wait(runner)
            runner._mark_api_runs_shutdown_requested()
            log.warning("%s; draining before supervised restart", reason)
        if draining and reason is not None:
            try:
                pending = runner._active_work_count() > 0
            except Exception:
                log.exception(
                    "cannot prove dependency restart is idle; retaining gateway"
                )
                pending = True
            if not pending:
                # No await between the final check and request_restart: admission is
                # already closed and the loop cannot start a new turn in this gap.
                if runner.request_restart(via_service=True):
                    return
        if not draining and reason == "":
            await asyncio.to_thread(observer.collect)
        await asyncio.sleep(poll_s)
