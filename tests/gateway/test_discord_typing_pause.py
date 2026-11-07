"""Discord must honor the shared pre-delivery/approval typing suspension contract."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.discord.adapter import DiscordAdapter
import plugins.platforms.discord.adapter as discord_module


class TypingClock:
    """Park periodic sleeps without changing task scheduling or HTTP behavior."""

    def __init__(self, interval):
        self.interval = interval
        self.waiters = asyncio.Queue()

    async def sleep(self, delay):
        if delay != self.interval:
            assert delay == 0, f"unexpected clock interval: {delay}"
            await asyncio.sleep(0)
            return
        waiter = asyncio.get_running_loop().create_future()
        self.waiters.put_nowait(waiter)
        await waiter

    def __getattr__(self, name):
        return getattr(asyncio, name)

    async def parked(self):
        return await asyncio.wait_for(self.waiters.get(), 5)


@pytest.fixture
def typing(monkeypatch):
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="test-token"))
    request = AsyncMock()
    adapter._client = SimpleNamespace(
        http=SimpleNamespace(request=request),
        get_channel=lambda channel_id: SimpleNamespace(
            send=AsyncMock(return_value=SimpleNamespace(id=123))),
    )
    clock = TypingClock(12)
    monkeypatch.setattr(discord_module, "asyncio", clock)
    return adapter, request, clock


async def advance_cycle(clock, sleeper, task):
    """Wait for either the next periodic boundary or cooperative task completion."""
    next_sleep = asyncio.create_task(clock.parked())
    if not sleeper.done():
        sleeper.set_result(None)
    try:
        done, _ = await asyncio.wait({next_sleep, task}, timeout=5,
                                     return_when=asyncio.FIRST_COMPLETED)
        assert done, "typing loop neither stopped nor reached its next clock boundary"
    finally:
        next_sleep.cancel()
        await asyncio.gather(next_sleep, return_exceptions=True)


@pytest.mark.asyncio
async def test_worker_thread_pause_suppresses_http_and_can_resume(typing):
    adapter, request, clock = typing
    loop = asyncio.get_running_loop()
    debug = loop.get_debug()
    loop.set_debug(True)
    try:
        await adapter.send_typing("77")
        sleeper = await clock.parked()
        task = adapter._typing_tasks["77"]
        await asyncio.to_thread(adapter.pause_typing_for_chat, "77")
        await advance_cycle(clock, sleeper, task)
        assert request.await_count == 1
        await asyncio.to_thread(adapter.resume_typing_for_chat, "77")
        await adapter.send_typing("77")
        await clock.parked()
        assert request.await_count == 2
    finally:
        await adapter.stop_typing("77")
        loop.set_debug(debug)


@pytest.mark.asyncio
async def test_pause_before_first_run_then_resume(typing):
    adapter, request, clock = typing
    try:
        await adapter.send_typing("77")
        task = adapter._typing_tasks["77"]
        adapter.pause_typing_for_chat("77")
        # Let the newly created task run or be cancelled before its first POST.
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert request.await_count == 0
        await adapter.send_typing("77")
        await asyncio.sleep(0)
        assert request.await_count == 0
        adapter.resume_typing_for_chat("77")
        await adapter.send_typing("77")
        await clock.parked()
        assert request.await_count == 1
        assert not adapter._typing_tasks["77"].done()
        assert task.done()
    finally:
        await adapter.stop_typing("77")


@pytest.mark.asyncio
@pytest.mark.parametrize("started", [False, True])
async def test_fast_pause_resume_keeps_refresh_alive(typing, started):
    adapter, request, clock = typing
    try:
        await adapter.send_typing("77")
        sleeper = await clock.parked() if started else None
        adapter.pause_typing_for_chat("77")
        adapter.resume_typing_for_chat("77")
        await adapter.send_typing("77")
        if sleeper is not None:
            sleeper.set_result(None)
        await clock.parked()
        assert request.await_count == (2 if started else 1)
    finally:
        await adapter.stop_typing("77")


@pytest.mark.asyncio
async def test_pause_suppresses_rate_limit_retry(typing, monkeypatch):
    adapter, request, clock = typing
    request.side_effect = [RuntimeError("rate limited"), None]
    monkeypatch.setattr(adapter, "_extract_discord_retry_after", lambda exc: 12)
    try:
        await adapter.send_typing("77")
        sleeper = await clock.parked()
        task = adapter._typing_tasks["77"]
        adapter.pause_typing_for_chat("77")
        await advance_cycle(clock, sleeper, task)
        assert request.await_count == 1
        adapter.resume_typing_for_chat("77")
        await adapter.send_typing("77")
        await clock.parked()
        assert request.await_count == 2
    finally:
        await adapter.stop_typing("77")


@pytest.mark.asyncio
@pytest.mark.parametrize("started", [False, True])
async def test_fast_pause_resume_and_new_turn_survive(typing, started):
    adapter, request, clock = typing
    try:
        await adapter.send_typing("77")
        if started:
            await clock.parked()
        adapter.pause_typing_for_chat("77")
        adapter.resume_typing_for_chat("77")
        await adapter.stop_typing("77")
        before = request.await_count
        await adapter.send_typing("77")
        await clock.parked()
        assert request.await_count == before + 1
        assert not adapter._typing_tasks["77"].done()
    finally:
        await adapter.stop_typing("77")
