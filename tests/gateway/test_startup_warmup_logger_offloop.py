"""Boot-time warning in ``_wait_bounded_or_release`` must not park the asyncio loop on
``logging.Handler.lock`` (P1: gateway watchdog exit-75 death loop).

Same bug class as #111091 (skill-slash fallthrough) and #110707 (Telegram command-menu scan).
The fix mirrors #111498's ``await asyncio.to_thread(...)`` shape: move the synchronous log call
off the event loop so a contended ``Handler.lock`` cannot freeze the Discord heartbeat.
"""

import asyncio
import logging
import threading
import time

import pytest

from gateway.run_startup import GatewayStartupMixin


def _runner() -> GatewayStartupMixin:
    return object.__new__(GatewayStartupMixin)


@pytest.mark.asyncio
async def test_warmup_timeout_warning_runs_off_the_event_loop(monkeypatch):
    """Heartbeat yields within a few ms of the gate-release warning entering ``emit()``.

    The bug: ``_wait_bounded_or_release`` calls ``logger.warning(warn_fmt, *args)`` directly
    on the event loop. When the logger's handler blocks on slow I/O inside ``emit()``, the
    asyncio loop is parked and the Discord heartbeat task cannot fire. After ~10 s the
    ``shutdown_watchdog`` dumps stacks and exits with code 75 — an indefinite death loop
    because the boot path auto-resumes the same ``resume_pending`` session on every restart.

    Invariant: the heartbeat task MUST be able to advance past a single ``await
    asyncio.sleep(0)`` after the warning enters ``emit()`` (proving the loop yielded
    control). Without the fix, the warning blocks the loop until ``emit()`` returns, so
    the heartbeat's first post-emit-entered sleep(0) is delayed by the entire emit duration.
    """

    emit_started = threading.Event()
    emit_duration_s = 0.4
    emit_returned_at: list[float] = []
    heartbeat_first_tick_at: list[float] = []

    class _BlockingHandler(logging.Handler):
        def emit(self, record) -> None:
            emit_started.set()
            time.sleep(emit_duration_s)
            emit_returned_at.append(time.perf_counter())

    handler = _BlockingHandler()
    base_logger = logging.getLogger(
        f"hermes.test.startup_logger_offloop.{id(handler)}"
    )
    base_logger.handlers = [handler]
    base_logger.setLevel(logging.WARNING)
    base_logger.propagate = False
    monkeypatch.setattr("gateway.run_startup.logger", base_logger)

    runner = _runner()
    pending = asyncio.create_task(asyncio.sleep(60))

    async def heartbeat() -> None:
        await asyncio.to_thread(emit_started.wait, 2.0)
        await asyncio.sleep(0)
        heartbeat_first_tick_at.append(time.perf_counter())

    hb_task = asyncio.create_task(heartbeat())

    try:
        await runner._wait_bounded_or_release(
            {pending}, 0.05,
            "warmup still running after %.0fs; pending=%d",
            "late-failure", level=logging.WARNING,
        )
    finally:
        await asyncio.gather(hb_task, return_exceptions=True)
        pending.cancel()

    assert emit_returned_at, "logger.emit was never called"
    assert heartbeat_first_tick_at, "heartbeat never ticked"
    loop_was_free_during_emit = heartbeat_first_tick_at[0] < emit_returned_at[0]
    assert loop_was_free_during_emit, (
        "logger.warning blocked the event loop: the heartbeat task could not get a "
        "tick until emit() returned. In production this is exactly the path that "
        "triggers gateway exit code 75."
    )
