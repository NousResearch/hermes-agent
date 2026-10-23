"""Blocking tool-process cleanup phases for gateway shutdown."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Any, Callable

logger = logging.getLogger("gateway.run")


class GatewayShutdownToolsMixin:
    """Tool sweep and its worker lifetime, shared by shutdown phases."""

    _TOOL_SUBPROCESS_SWEEP_TIMEOUT_S = 30.0

    @staticmethod
    def _stop_kill_tool_subprocesses(phase: str) -> list:
        """Kill tool subprocesses + terminal envs + browsers; returns cron job IDs marked interrupted.

        Called before adapter teardown after either a successful or timed-out drain,
        then again as a final catch-all. Best-effort; one failing subsystem cannot block the rest.
        """

        from gateway.run_shutdown import GatewayShutdownMixin

        def _step(label: str, fn: Callable[[], Any]) -> Any:
            return GatewayShutdownMixin._quiet_step(f"{label} ({phase}) error", fn)

        def _count_step(fmt: str, fn: Callable[[], int]) -> None:
            n = fn()
            if n:
                logger.info(fmt, phase, n)

        def _kill_processes() -> None:
            from tools.process_registry import process_registry
            # Host shutdown: kill even persist_on_release jobs or they become
            # PPID=1 orphans (#41225/#46778); an explicit source reaches them.
            _count_step(
                "Shutdown (%s): killed %d tool subprocess(es)",
                lambda: process_registry.kill_all(source="gateway_shutdown"))

        def _mark_cron_interrupted() -> list:
            # kill_all() is global: mark dispatched cron jobs before killing their
            # tools so truncated output cannot be recorded as success (#60432).
            # Persist first, even when kill_all later blocks or fails.
            from cron.scheduler import mark_running_jobs_interrupted
            _interrupted = mark_running_jobs_interrupted(
                f"Gateway shutdown ({phase}) interrupted the job before tool subprocess cleanup."
            )
            if _interrupted:
                logger.warning(
                    "Shutdown (%s): marked %d in-flight cron job(s) interrupted: %s",
                    phase, len(_interrupted), ", ".join(_interrupted),
                )
            return _interrupted

        def _interrupt_delegations() -> None:
            from tools.async_delegation import interrupt_all as _interrupt_async
            _count_step(
                "Shutdown (%s): interrupted %d background delegation(s)",
                lambda: _interrupt_async(reason=f"gateway shutdown ({phase})"),
            )

        # Persist interruption before a blocking kill can strand the worker.
        _marked_cron_jobs = _step("mark_running_jobs_interrupted", _mark_cron_interrupted) or []
        _step("process_registry.kill_all", _kill_processes)
        _step("async interrupt_all", _interrupt_delegations)
        def _cleanup_environments() -> None:
            from tools.terminal_tool_lifecycle import cleanup_all_environments
            cleanup_all_environments()

        def _cleanup_browsers() -> None:
            from tools.browser_tool_lifecycle import cleanup_all_browsers
            cleanup_all_browsers()

        _step("cleanup_all_environments", _cleanup_environments)
        _step("cleanup_all_browsers", _cleanup_browsers)
        return _marked_cron_jobs

    async def _stop_kill_tool_subprocesses_off_loop(self, phase: str) -> list:
        """Bound the wait, keeping the actual worker visible to the exit watchdog.

        Cancelling a to_thread await cannot stop its executor thread. A timed-out
        sweep must neither spawn duplicate cleanup nor disarm the watchdog before
        asyncio.run joins that thread. The completion Event tracks the thread, not
        the asyncio Task (which asyncio.run may cancel during its own teardown).
        """
        from gateway.run_shutdown import GatewayShutdownMixin

        pending = getattr(self, "_tool_subprocess_sweep_worker", None)
        if pending is not None and not pending[1].is_set():
            self._tool_subprocess_sweep_incomplete = True
            logger.warning("Shutdown (%s): previous tool sweep still running; not starting another", phase)
            return []
        budget = min(
            GatewayShutdownMixin._TOOL_SUBPROCESS_SWEEP_TIMEOUT_S,
            max(0.0, getattr(self, "_shutdown_tool_sweep_deadline", float("inf")) - time.monotonic()),
        )
        if budget <= 0:
            self._tool_subprocess_sweep_incomplete = True
            logger.warning("Shutdown (%s): no tool sweep budget remains; cleanup is incomplete", phase)
            return []
        done = threading.Event()

        def sweep() -> list:
            try:
                return GatewayShutdownMixin._stop_kill_tool_subprocesses(phase)
            finally:
                done.set()

        task = asyncio.create_task(asyncio.to_thread(sweep))
        def observe_result(result: asyncio.Task) -> None:
            if not result.cancelled() and (error := result.exception()) is not None:
                logger.error("Shutdown (%s): tool sweep worker failed", phase,
                             exc_info=(type(error), error, error.__traceback__))

        task.add_done_callback(observe_result)
        self._tool_subprocess_sweep_worker = (task, done)
        try:
            return await asyncio.wait_for(asyncio.shield(task), timeout=budget)
        except (TimeoutError, asyncio.CancelledError) as exc:
            self._tool_subprocess_sweep_incomplete = True
            logger.warning(
                "Shutdown (%s): tool sweep unfinished after %.2fs; cleanup is incomplete and "
                "the shutdown watchdog remains armed for any live worker", phase, budget,
            )
            if isinstance(exc, asyncio.CancelledError):
                raise
            return []
