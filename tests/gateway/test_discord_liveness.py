"""Regression tests for the Discord REST liveness probe (#26656).

discord.py's WebSocket reconnect handles clean drops, but a wedged proxy /
NAT can leave the underlying socket dead without ever delivering a RST —
sends time out forever while ``client.start()`` happily spins and never
exits, so the bot-task done callback never fires either.  The probe in
``DiscordAdapter`` periodically hits Discord REST so we can detect the
zombie state and trip the gateway's existing reconnect path.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

# Re-use the shared discord-stub bootstrap and FakeBot from the connect
# test module so this file doesn't duplicate the (large) mock surface.
from tests.gateway.test_discord_connect import (  # noqa: E402
    FakeBot,
    _ensure_discord_mock,
)

_ensure_discord_mock()

import plugins.platforms.discord.adapter as discord_platform  # noqa: E402
from gateway.config import PlatformConfig  # noqa: E402
from plugins.platforms.discord.adapter import DiscordAdapter  # noqa: E402


class _LiveBot(FakeBot):
    """A FakeBot whose ``start()`` stays pending like a real discord.py client.

    The default ``FakeBot.start()`` returns immediately, which would let the
    bot-task done callback fire and set a spurious fatal error.  Real clients
    keep ``start()`` running for the life of the connection; this models that
    so the liveness probe is the only thing that can trip a fatal error.
    """

    def __init__(self, *, intents, proxy=None, allowed_mentions=None, **_):
        super().__init__(intents=intents, allowed_mentions=allowed_mentions)
        self._never = asyncio.Event()
        self._closed = False

    async def start(self, token):
        if "on_ready" in self._events:
            await self._events["on_ready"]()
        # Stay alive until close() is called — mirrors a real client.
        await self._never.wait()

    def is_closed(self):
        return self._closed

    async def close(self):
        self._closed = True
        self._never.set()


def _make_adapter(monkeypatch, *, interval=0.01, threshold=1) -> DiscordAdapter:
    monkeypatch.setenv("HERMES_DISCORD_LIVENESS_INTERVAL_SECONDS", str(interval))
    monkeypatch.setenv("HERMES_DISCORD_LIVENESS_FAILURE_THRESHOLD", str(threshold))
    return DiscordAdapter(PlatformConfig(enabled=True, token="test-token"))


async def _connect(adapter: DiscordAdapter, monkeypatch, bot_factory):
    monkeypatch.setattr(
        "gateway.status.acquire_scoped_lock",
        lambda scope, identity, metadata=None: (True, None),
    )
    monkeypatch.setattr("gateway.status.release_scoped_lock", lambda scope, identity: None)
    intents = SimpleNamespace(
        message_content=False, dm_messages=False, guild_messages=False,
        members=False, voice_states=False,
    )
    monkeypatch.setattr(discord_platform.Intents, "default", lambda: intents)
    monkeypatch.setattr(discord_platform.commands, "Bot", bot_factory)
    monkeypatch.setattr(adapter, "_resolve_allowed_usernames", AsyncMock())
    assert await adapter.connect() is True


@pytest.mark.asyncio
async def test_liveness_probe_disabled_when_interval_zero(monkeypatch):
    """interval<=0 must skip the probe entirely so users can opt out."""
    adapter = _make_adapter(monkeypatch, interval=0)

    bot_holder: dict = {}

    def factory(**kwargs):
        bot = _LiveBot(intents=kwargs["intents"], allowed_mentions=kwargs.get("allowed_mentions"))
        bot.fetch_user = AsyncMock()
        bot_holder["bot"] = bot
        return bot

    await _connect(adapter, monkeypatch, factory)
    assert adapter._liveness_task is None
    await asyncio.sleep(0.05)
    bot_holder["bot"].fetch_user.assert_not_called()
    await adapter.disconnect()


@pytest.mark.asyncio
async def test_liveness_probe_disabled_when_threshold_zero(monkeypatch):
    """threshold<=0 must also skip the probe."""
    adapter = _make_adapter(monkeypatch, interval=0.01, threshold=0)

    def factory(**kwargs):
        bot = _LiveBot(intents=kwargs["intents"], allowed_mentions=kwargs.get("allowed_mentions"))
        bot.fetch_user = AsyncMock()
        return bot

    await _connect(adapter, monkeypatch, factory)
    assert adapter._liveness_task is None
    await adapter.disconnect()


@pytest.mark.asyncio
async def test_liveness_probe_pings_rest_while_healthy(monkeypatch):
    """A healthy probe keeps the adapter running and never sets a fatal error."""
    adapter = _make_adapter(monkeypatch, interval=0.01, threshold=3)

    def factory(**kwargs):
        bot = _LiveBot(intents=kwargs["intents"], allowed_mentions=kwargs.get("allowed_mentions"))
        bot.fetch_user = AsyncMock(return_value=SimpleNamespace(id=999))
        return bot

    await _connect(adapter, monkeypatch, factory)
    await asyncio.sleep(0.05)
    assert adapter._client.fetch_user.await_count >= 1
    assert adapter._running is True
    assert adapter.has_fatal_error is False
    await adapter.disconnect()


@pytest.mark.asyncio
async def test_liveness_probe_forces_reconnect_after_threshold(monkeypatch):
    """Once the probe fails ``threshold`` times in a row, the adapter must
    close the wedged client and surface a retryable fatal error so the
    gateway's reconnect watcher (gateway/run.py) can rebuild it."""
    adapter = _make_adapter(monkeypatch, interval=0.005, threshold=2)

    def factory(**kwargs):
        bot = _LiveBot(intents=kwargs["intents"], allowed_mentions=kwargs.get("allowed_mentions"))
        bot.fetch_user = AsyncMock(side_effect=TimeoutError("dead proxy"))
        return bot

    handler = AsyncMock()
    adapter.set_fatal_error_handler(handler)
    await _connect(adapter, monkeypatch, factory)
    wedged = adapter._client

    # Wait for the loop to exit (it returns after threshold consecutive
    # failures).  Bounded by a generous timeout so a regression doesn't hang CI.
    for _ in range(200):
        if adapter._liveness_task and adapter._liveness_task.done():
            break
        await asyncio.sleep(0.01)
    else:
        pytest.fail("liveness loop did not terminate within 2s")

    assert wedged.is_closed() is True
    assert adapter.has_fatal_error is True
    assert adapter.fatal_error_code == "liveness_probe_failed"
    assert adapter.fatal_error_retryable is True
    handler.assert_awaited_once()

    await adapter.disconnect()


@pytest.mark.asyncio
async def test_disconnect_cancels_liveness_task(monkeypatch):
    """``disconnect()`` must cancel the probe so the gateway can shut down
    cleanly without leaking a background task."""
    adapter = _make_adapter(monkeypatch, interval=60, threshold=3)

    def factory(**kwargs):
        bot = _LiveBot(intents=kwargs["intents"], allowed_mentions=kwargs.get("allowed_mentions"))
        bot.fetch_user = AsyncMock()
        return bot

    await _connect(adapter, monkeypatch, factory)
    task = adapter._liveness_task
    assert task is not None and not task.done()

    await adapter.disconnect()
    assert task.done()
    assert adapter._liveness_task is None
