from __future__ import annotations

import asyncio
import logging
import threading

import pytest

from gateway.run import GatewayRunner


def test_handoff_db_helper_times_out_without_blocking_event_loop(caplog):
    async def exercise():
        runner = object.__new__(GatewayRunner)
        runner._handoff_db_timeout_secs = 0.05
        runner._handoff_db_circuit_break_secs = 30.0
        runner._handoff_db_circuit_open_until = 0.0
        runner._handoff_db_inflight_task = None

        release = threading.Event()

        def stuck_db_call():
            release.wait(timeout=5.0)

        caplog.set_level(logging.WARNING, logger="gateway.run")
        call_task = asyncio.create_task(
            runner._run_handoff_db_call("stuck_db_call", stuck_db_call)
        )

        try:
            assert await asyncio.wait_for(asyncio.sleep(0.01, result=True), timeout=0.1)
            with pytest.raises(asyncio.TimeoutError):
                await call_task

            assert runner._handoff_db_circuit_open_until > 0.0
            assert "Handoff DB call stuck_db_call timed out" in caplog.text
        finally:
            release.set()
            inflight = runner._handoff_db_inflight_task
            if inflight is not None:
                await asyncio.wait_for(asyncio.shield(inflight), timeout=1.0)

    asyncio.run(exercise())


def test_handoff_db_helper_open_circuit_skips_without_calling_db(caplog):
    async def exercise():
        runner = object.__new__(GatewayRunner)
        runner._handoff_db_timeout_secs = 1.0
        runner._handoff_db_circuit_break_secs = 30.0
        runner._handoff_db_circuit_open_until = 9999999999.0
        runner._handoff_db_inflight_task = None
        called = False

        def db_call():
            nonlocal called
            called = True

        caplog.set_level(logging.WARNING, logger="gateway.run")
        with pytest.raises(asyncio.TimeoutError):
            await runner._run_handoff_db_call("list_pending_handoffs", db_call)

        assert called is False
        assert "circuit open" in caplog.text

    asyncio.run(exercise())
