import asyncio
import concurrent.futures

import pytest

from agent.inactivity_watch import (
    InactivityDiagnostic,
    is_idle_past_limit,
    wait_for_future_or_inactivity,
    wait_for_task_or_inactivity,
    build_activity_diagnostic,
)


class FakeAgent:
    def __init__(self, seconds_since_activity=0.0):
        self.seconds_since_activity = seconds_since_activity

    def get_activity_summary(self):
        return {
            "last_activity_desc": "stream_delta",
            "seconds_since_activity": self.seconds_since_activity,
            "current_tool": "web_search",
            "api_call_count": 7,
            "max_iterations": 90,
        }


class BareAgent:
    pass


def test_idle_past_limit_fires():
    assert is_idle_past_limit(FakeAgent(seconds_since_activity=6.0), 5.0) is True


def test_active_agent_never_fires():
    assert is_idle_past_limit(FakeAgent(seconds_since_activity=4.9), 5.0) is False


def test_no_tracker_falls_back_to_not_idle_and_default_diagnostics():
    assert is_idle_past_limit(BareAgent(), 0.1) is False
    assert build_activity_diagnostic(BareAgent()) == InactivityDiagnostic(
        last_activity_desc="unknown",
        seconds_since_activity=0,
        current_tool=None,
        api_call_count=0,
        max_iterations=0,
    )


def test_sync_wait_reports_inactivity_timeout_for_pending_future():
    future = concurrent.futures.Future()

    result = wait_for_future_or_inactivity(
        future,
        agent=FakeAgent(seconds_since_activity=6.0),
        inactivity_limit=5.0,
        poll_interval=0.0,
    )

    assert result.timed_out is True
    assert result.result is None


@pytest.mark.asyncio
async def test_async_wait_reports_inactivity_timeout_for_pending_task():
    future = asyncio.Future()

    result = await wait_for_task_or_inactivity(
        future,
        get_agent=lambda: FakeAgent(seconds_since_activity=6.0),
        inactivity_limit=5.0,
        poll_interval=0.0,
    )

    assert result.timed_out is True
    assert result.result is None


def test_sync_wait_returns_result_when_future_completes():
    # Success path (Greptile): a completing future returns its value, no timeout.
    future = concurrent.futures.Future()
    future.set_result("done-value")

    result = wait_for_future_or_inactivity(
        future,
        agent=FakeAgent(seconds_since_activity=999.0),  # idle, but future already done
        inactivity_limit=5.0,
        poll_interval=0.0,
    )

    assert result.timed_out is False
    assert result.result == "done-value"


def test_sync_wait_none_limit_blocks_to_result():
    future = concurrent.futures.Future()
    future.set_result(42)
    result = wait_for_future_or_inactivity(
        future, agent=FakeAgent(), inactivity_limit=None,
    )
    assert result.result == 42 and result.timed_out is False


@pytest.mark.asyncio
async def test_async_wait_returns_result_when_task_completes():
    future = asyncio.Future()
    future.set_result("async-done")

    result = await wait_for_task_or_inactivity(
        future,
        get_agent=lambda: FakeAgent(seconds_since_activity=999.0),
        inactivity_limit=5.0,
        poll_interval=0.0,
    )

    assert result.timed_out is False
    assert result.result == "async-done"


@pytest.mark.asyncio
async def test_async_wait_invokes_callbacks_each_poll():
    # Callback path (Greptile): on_idle_check gets the idle reading, on_poll runs
    # per poll; both sync and awaitable callables are accepted (_maybe_await).
    future = asyncio.Future()
    idle_readings: list = []
    polls: list = []

    async def on_idle_check(idle_secs):
        idle_readings.append(idle_secs)
        if len(idle_readings) >= 2:
            future.set_result("finished-mid-poll")

    def on_poll():  # sync callable variant
        polls.append(1)

    result = await wait_for_task_or_inactivity(
        future,
        get_agent=lambda: FakeAgent(seconds_since_activity=1.0),  # never idle-out
        inactivity_limit=5.0,
        poll_interval=0.0,
        on_idle_check=on_idle_check,
        on_poll=on_poll,
    )

    assert result.result == "finished-mid-poll"
    assert result.timed_out is False
    assert len(idle_readings) >= 2
    assert idle_readings[0] == 1.0
    assert polls, "on_poll must run on non-idle polls"
