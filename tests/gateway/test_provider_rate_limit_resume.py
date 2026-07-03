import asyncio
import threading
import time
from typing import Any

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.run import (
    GatewayRunner,
    _coerce_provider_rate_limit_reset_at,
    _provider_rate_limit_reset_at,
)
from gateway.session import SessionSource


class _Adapter:
    def __init__(self):
        self.events = []

    async def handle_message(self, event: MessageEvent):
        self.events.append(event)


class _SessionStore:
    def __init__(self):
        self.marks = []
        self.mark_threads = []
        self.clears = []
        self.can_resume = True

    def mark_resume_pending(
        self, session_key, reason="restart_timeout", not_before=None
    ):
        self.mark_threads.append(threading.get_ident())
        self.marks.append((session_key, reason, not_before))
        return True

    def is_resume_pending(self, session_key, reason=None):
        return self.can_resume

    def clear_resume_pending(self, session_key, reason=None):
        self.clears.append((session_key, reason))
        if not self.can_resume or reason not in {None, "provider_rate_limit"}:
            return False
        self.can_resume = False
        return True


class _ConcurrencyAdapter(_Adapter):
    def __init__(self):
        super().__init__()
        self.active = 0
        self.max_active = 0
        self._session_tasks = {}
        self._both_active = asyncio.Event()

    async def _run_background(self, event: MessageEvent):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        if self.active >= 2:
            self._both_active.set()
        try:
            await asyncio.wait_for(self._both_active.wait(), timeout=1.0)
            self.events.append(event)
        finally:
            self.active -= 1

    async def handle_message(self, event: MessageEvent):
        session_key = {"C1": "s1", "C2": "s2"}[event.source.chat_id]
        self._session_tasks[session_key] = asyncio.create_task(
            self._run_background(event)
        )


def _runner(adapter: _Adapter) -> GatewayRunner:
    runner: Any = object.__new__(GatewayRunner)
    runner.adapters = {Platform.SLACK: adapter}
    runner._background_tasks = set()
    runner._provider_rate_limit_resume_tasks = {}
    runner._session_run_generation = {"s1": 7}
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._persist_active_agents = lambda: None
    runner._is_user_authorized = lambda source: True
    runner.session_store = _SessionStore()
    return runner


def test_provider_limit_reset_time_requires_future_timestamp():
    assert _coerce_provider_rate_limit_reset_at(1200, now=1000) == 1200
    assert (
        _coerce_provider_rate_limit_reset_at(
            "1970-01-01T00:20:00+00:00", now=1000
        )
        == 1200
    )
    assert _coerce_provider_rate_limit_reset_at("1970-01-01T00:20:00", now=1000) is None
    assert _coerce_provider_rate_limit_reset_at(900, now=1000) is None
    assert _coerce_provider_rate_limit_reset_at("not-a-time", now=1000) is None
    assert _coerce_provider_rate_limit_reset_at(float("nan"), now=1000) is None
    assert _coerce_provider_rate_limit_reset_at(float("inf"), now=1000) is None
    assert _coerce_provider_rate_limit_reset_at(1e100, now=1000) is None


def test_provider_limit_reset_only_uses_rate_limit_results():
    future_reset = time.time() + 120
    assert (
        _provider_rate_limit_reset_at(
            {"failure_reason": "billing", "error_context": {"reset_at": future_reset}}
        )
        is None
    )
    assert (
        _provider_rate_limit_reset_at(
            {"failure_reason": "rate_limit", "error_context": {}}
        )
        is None
    )
    assert (
        _provider_rate_limit_reset_at(
            {
                "failure_reason": "rate_limit",
                "error_context": {"reset_at": future_reset},
            }
        )
        == future_reset
    )


@pytest.mark.asyncio
async def test_provider_limit_resume_dispatches_internal_continuation():
    adapter = _Adapter()
    runner = _runner(adapter)
    source = SessionSource(platform=Platform.SLACK, chat_id="C1", user_id="U1")

    await runner._run_provider_rate_limit_resume_after_delay(
        session_key="s1",
        source=source,
        reset_at=0,
        run_generation=7,
    )

    assert len(adapter.events) == 1
    event = adapter.events[0]
    assert event.internal is True
    assert event.source is source
    assert event.text == ""


@pytest.mark.asyncio
async def test_provider_limit_resume_skips_suspended_or_cleared_session():
    adapter = _Adapter()
    runner = _runner(adapter)
    source = SessionSource(platform=Platform.SLACK, chat_id="C1", user_id="U1")
    runner.session_store.can_resume = False

    await runner._run_provider_rate_limit_resume_after_delay(
        session_key="s1",
        source=source,
        reset_at=0,
        run_generation=7,
    )

    assert adapter.events == []


@pytest.mark.asyncio
async def test_provider_limit_resume_waits_for_origin_run_to_release_slot():
    adapter = _Adapter()
    runner = _runner(adapter)
    source = SessionSource(platform=Platform.SLACK, chat_id="C1", user_id="U1")
    runner._running_agents["s1"] = object()

    continuation = asyncio.create_task(
        runner._run_provider_rate_limit_resume_after_delay(
            session_key="s1",
            source=source,
            reset_at=0,
            run_generation=7,
        )
    )
    await asyncio.sleep(0.05)
    assert adapter.events == []
    assert not continuation.done()

    runner._running_agents.pop("s1")
    await asyncio.wait_for(continuation, timeout=1.0)

    assert len(adapter.events) == 1


@pytest.mark.asyncio
async def test_provider_limit_resume_skips_when_newer_turn_exists():
    adapter = _Adapter()
    runner = _runner(adapter)
    source = SessionSource(platform=Platform.SLACK, chat_id="C1", user_id="U1")
    runner._session_run_generation["s1"] = 8

    await runner._run_provider_rate_limit_resume_after_delay(
        session_key="s1",
        source=source,
        reset_at=0,
        run_generation=7,
    )

    assert adapter.events == []


@pytest.mark.asyncio
async def test_real_user_turn_supersedes_durable_provider_continuation():
    adapter = _Adapter()
    runner = _runner(adapter)
    source = SessionSource(platform=Platform.SLACK, chat_id="C1", user_id="U1")
    reset_at = time.time() + 10_000
    assert await runner._schedule_provider_rate_limit_resume(
        session_key="s1",
        source=source,
        reset_at=reset_at,
        run_generation=7,
    )
    stale_task = runner._provider_rate_limit_resume_tasks["s1"]

    superseded = await runner._supersede_provider_rate_limit_resume_for_user_turn(
        session_key="s1"
    )

    assert superseded is True
    assert runner.session_store.clears == [("s1", "provider_rate_limit")]
    assert runner.session_store.can_resume is False
    assert "s1" not in runner._provider_rate_limit_resume_tasks
    await asyncio.gather(stale_task, return_exceptions=True)
    assert stale_task.cancelled()


@pytest.mark.asyncio
async def test_provider_limit_resume_scheduler_persists_deadline():
    adapter = _Adapter()
    runner = _runner(adapter)
    source = SessionSource(platform=Platform.SLACK, chat_id="C1", user_id="U1")
    reset_at = time.time() + 10_000
    loop_thread = threading.get_ident()

    scheduled = await runner._schedule_provider_rate_limit_resume(
        session_key="s1",
        source=source,
        reset_at=reset_at,
        run_generation=7,
    )

    assert scheduled is True
    assert runner.session_store.marks == [
        ("s1", "provider_rate_limit", reset_at)
    ]
    assert runner.session_store.mark_threads != [loop_thread]
    assert "s1" in runner._provider_rate_limit_resume_tasks
    task = runner._provider_rate_limit_resume_tasks["s1"]
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_provider_limit_scheduler_replaces_stale_wait_for_newer_turn():
    adapter = _Adapter()
    runner = _runner(adapter)
    source = SessionSource(platform=Platform.SLACK, chat_id="C1", user_id="U1")

    assert await runner._schedule_provider_rate_limit_resume(
        session_key="s1",
        source=source,
        reset_at=time.time() + 10_000,
        run_generation=7,
    )
    first = runner._provider_rate_limit_resume_tasks["s1"]
    runner._session_run_generation["s1"] = 8

    assert await runner._schedule_provider_rate_limit_resume(
        session_key="s1",
        source=source,
        reset_at=time.time() + 20_000,
        run_generation=8,
    )
    second = runner._provider_rate_limit_resume_tasks["s1"]

    assert second is not first
    assert first.cancelled() or first.cancelling()
    second.cancel()
    await asyncio.gather(first, second, return_exceptions=True)


@pytest.mark.asyncio
async def test_provider_limit_scheduler_does_not_cancel_dispatched_resume():
    adapter = _Adapter()
    runner = _runner(adapter)
    source = SessionSource(platform=Platform.SLACK, chat_id="C1", user_id="U1")

    assert await runner._schedule_provider_rate_limit_resume(
        session_key="s1",
        source=source,
        reset_at=time.time() + 10_000,
        run_generation=7,
    )
    first = runner._provider_rate_limit_resume_tasks["s1"]
    setattr(first, "_hermes_provider_resume_dispatched", True)
    runner._session_run_generation["s1"] = 8

    assert await runner._schedule_provider_rate_limit_resume(
        session_key="s1",
        source=source,
        reset_at=time.time() + 20_000,
        run_generation=8,
    )
    second = runner._provider_rate_limit_resume_tasks["s1"]

    assert not first.cancelling()
    first.cancel()
    second.cancel()
    await asyncio.gather(first, second, return_exceptions=True)


@pytest.mark.asyncio
async def test_provider_limit_resume_uses_source_profile_adapter():
    adapter = _Adapter()
    runner = _runner(adapter)
    runner.adapters = {}
    runner.__dict__["_profile_adapters"] = {"work": {Platform.SLACK: adapter}}
    source = SessionSource(
        platform=Platform.SLACK,
        chat_id="C1",
        user_id="U1",
        profile="work",
    )

    await runner._run_provider_rate_limit_resume_after_delay(
        session_key="s1", source=source, reset_at=0, run_generation=7
    )

    assert len(adapter.events) == 1


@pytest.mark.asyncio
async def test_provider_limit_resumes_run_in_parallel_across_sessions():
    adapter = _ConcurrencyAdapter()
    runner = _runner(adapter)
    runner._session_run_generation["s2"] = 7
    source1 = SessionSource(platform=Platform.SLACK, chat_id="C1", user_id="U1")
    source2 = SessionSource(platform=Platform.SLACK, chat_id="C2", user_id="U2")

    await asyncio.gather(
        runner._run_provider_rate_limit_resume_after_delay(
            session_key="s1", source=source1, reset_at=0, run_generation=7
        ),
        runner._run_provider_rate_limit_resume_after_delay(
            session_key="s2", source=source2, reset_at=0, run_generation=7
        ),
    )

    assert len(adapter.events) == 2
    assert adapter.max_active == 2
