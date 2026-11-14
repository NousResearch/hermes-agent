"""Retry estimates are anchored when Discord surfaces the cooldown, not at delivery."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.discord.adapter import DiscordAdapter
from plugins.platforms.discord import adapter_auto_thread as auto_thread


def test_notice_uses_one_rounded_deadline_even_when_rendered_later(monkeypatch):
    clock = SimpleNamespace(epoch=1700000000.25)
    monkeypatch.setattr(auto_thread, "time", SimpleNamespace(time=lambda: clock.epoch, monotonic=lambda: 10), raising=False)
    error = auto_thread._AutoThreadRateLimited(60.01)
    clock.epoch += 45
    notice = error.notice()
    assert "<t:1700000061:R>" in notice
    assert "<t:1700000061:F>" in notice
    assert "estimate" in notice.lower()
    assert "not processed" not in notice


@pytest.mark.asyncio
async def test_known_cooldown_reuses_deadline_without_another_create(monkeypatch):
    import discord
    clock = SimpleNamespace(epoch=1700000000.25, mono=10)
    monkeypatch.setattr(auto_thread, "time", SimpleNamespace(
        time=lambda: clock.epoch, monotonic=lambda: clock.mono,
    ), raising=False)
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="test", extra={}))
    message = SimpleNamespace(
        content="hello", author=None, create_thread=AsyncMock(side_effect=discord.RateLimited(60.01)),
        channel=SimpleNamespace(id=800, send=AsyncMock()),
    )
    assert await adapter._auto_create_thread_or_notify(message) is None
    first = message.channel.send.await_args.args[0]
    clock.epoch += 20
    clock.mono += 20
    assert await adapter._auto_create_thread_or_notify(message) is None
    assert message.channel.send.await_args.args[0] == first
    message.create_thread.assert_awaited_once()
    clock.epoch += 41
    clock.mono += 41
    assert await adapter._auto_create_thread_or_notify(message) is None
    assert message.create_thread.await_count == 2
    assert message.channel.send.await_args.args[0] != first


@pytest.mark.asyncio
async def test_fallback_cleanup_and_notice_delay_do_not_restart_estimate(monkeypatch):
    import discord
    clock = SimpleNamespace(epoch=1700000000.25, mono=10)
    monkeypatch.setattr(auto_thread, "time", SimpleNamespace(
        time=lambda: clock.epoch, monotonic=lambda: clock.mono,
    ))
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="test", extra={}))
    delivered = []

    async def delayed_delete():
        clock.epoch += 20
        clock.mono += 20

    seed = SimpleNamespace(create_thread=AsyncMock(side_effect=discord.RateLimited(60.01)),
                           delete=AsyncMock(side_effect=delayed_delete))

    async def send(content):
        if not seed.create_thread.await_count:
            return seed
        clock.epoch += 25
        clock.mono += 25
        delivered.append(content)

    message = SimpleNamespace(content="hello", author=None,
        create_thread=AsyncMock(side_effect=RuntimeError("connection lost")),
        channel=SimpleNamespace(id=800, send=AsyncMock(side_effect=send)))
    assert await adapter._auto_create_thread_or_notify(message) is None
    assert "<t:1700000061:R>" in delivered[0]
    assert "<t:1700000061:F>" in delivered[0]
    seed.delete.assert_awaited_once()
    message.create_thread.assert_awaited_once()


@pytest.mark.asyncio
async def test_inflight_repeat_preserves_an_already_known_deadline(monkeypatch):
    import asyncio
    import discord
    clock = SimpleNamespace(epoch=1700000000.25, mono=10)
    monkeypatch.setattr(auto_thread, "time", SimpleNamespace(
        time=lambda: clock.epoch, monotonic=lambda: clock.mono,
    ))
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="test", extra={}))
    started = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def create_thread(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            started.set()
            await release.wait()
        raise discord.RateLimited(60)

    message = SimpleNamespace(content="hello", author=None, create_thread=create_thread,
                              channel=SimpleNamespace(id=800, send=AsyncMock()))
    pending = asyncio.create_task(adapter._auto_create_thread_or_notify(message))
    try:
        await started.wait()
        await adapter._auto_create_thread_or_notify(message)
        first = message.channel.send.await_args.args[0]
        clock.epoch += 20
        clock.mono += 20
        release.set()
        await pending
        assert message.channel.send.await_args.args[0] == first
    finally:
        release.set()
        await pending
