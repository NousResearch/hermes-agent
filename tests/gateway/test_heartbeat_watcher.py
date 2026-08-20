"""Tests for the gateway heartbeat watcher (issue #32887)."""

import asyncio
from unittest.mock import MagicMock

import pytest

from tests.gateway.restart_test_helpers import make_restart_runner


@pytest.mark.asyncio
async def test_heartbeat_watcher_updates_runtime_status_periodically():
    """The heartbeat watcher must call _update_runtime_status at least twice
    within a short window, proving it ticks independently of state changes.
    """
    runner, _adapter = make_restart_runner()
    call_count = 0

    def _track_update(*, gateway_state=None, exit_reason=None):
        nonlocal call_count
        call_count += 1

    runner._update_runtime_status = _track_update

    # Run the watcher with a short interval. The initial delay is 10s in
    # production; we patch asyncio.sleep's first call to skip it.
    sleep_calls = 0
    real_sleep = asyncio.sleep

    async def _fast_first_sleep(delay, *args, **kwargs):
        nonlocal sleep_calls
        sleep_calls += 1
        if sleep_calls == 1 and delay >= 10:
            return  # skip initial 10s delay
        await real_sleep(delay, *args, **kwargs)

    import gateway.run as _gw_run
    original_sleep = _gw_run.asyncio.sleep
    _gw_run.asyncio.sleep = _fast_first_sleep
    try:
        async def _run_and_stop():
            task = asyncio.create_task(runner._heartbeat_watcher(interval=1))
            # Wait long enough for at least 2 ticks (interval=1s each)
            await real_sleep(2.5)
            runner._running = False
            await task

        await _run_and_stop()
        assert call_count >= 2, f"Expected >= 2 heartbeat ticks, got {call_count}"
    finally:
        _gw_run.asyncio.sleep = original_sleep


@pytest.mark.asyncio
async def test_heartbeat_watcher_stops_when_gateway_stops():
    """The heartbeat watcher must exit when _running is set to False."""
    runner, _adapter = make_restart_runner()
    runner._running = False  # Already stopped

    # Should return immediately without error
    await runner._heartbeat_watcher(interval=1)


@pytest.mark.asyncio
async def test_heartbeat_watcher_survives_update_failure():
    """The heartbeat watcher must not crash if _update_runtime_status raises."""
    runner, _adapter = make_restart_runner()
    runner._update_runtime_status = MagicMock(side_effect=OSError("disk full"))

    sleep_calls = 0
    real_sleep = asyncio.sleep

    async def _fast_first_sleep(delay, *args, **kwargs):
        nonlocal sleep_calls
        sleep_calls += 1
        if sleep_calls == 1 and delay >= 10:
            return
        await real_sleep(delay, *args, **kwargs)

    import gateway.run as _gw_run
    original_sleep = _gw_run.asyncio.sleep
    _gw_run.asyncio.sleep = _fast_first_sleep
    try:
        async def _run_briefly():
            task = asyncio.create_task(runner._heartbeat_watcher(interval=1))
            await real_sleep(0.2)
            runner._running = False
            await task

        # Should not raise despite OSError in _update_runtime_status
        await _run_briefly()
        assert runner._update_runtime_status.call_count >= 1
    finally:
        _gw_run.asyncio.sleep = original_sleep
