"""Dispatch-side liveness for the Discord adapter (#109521 incident 2).

An ESTAB Gateway socket can keep ACKing heartbeats while zero DISPATCH
events are parsed — every transport-side sample reads healthy while the
adapter is deaf.  The probe's ``event_silence`` dimension stamps
``on_socket_event_type`` (dispatched for every parsed DISPATCH frame,
unlike the debug-gated ``on_socket_raw_receive``) and trips after
``websocket_event_max_silence_seconds`` without one.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tests.gateway.test_discord_connect import (  # noqa: E402
    FakeBot,
    _ensure_discord_mock,
)

_ensure_discord_mock()

import plugins.platforms.discord.adapter as discord_platform  # noqa: E402
from gateway.config import PlatformConfig  # noqa: E402
from plugins.platforms.discord.adapter import DiscordAdapter  # noqa: E402

from tests.gateway.test_discord_liveness import (  # noqa: E402
    _LiveBot,
    _set_websocket_health,
)


class _DispatchingBot(_LiveBot):
    """A live bot that can deliver parsed DISPATCH events like the real gateway.

    Real discord.py ``received_message`` parses each frame, calls
    ``self._dispatch('socket_event_type', event)`` for every DISPATCH op,
    and returns early on heartbeat ACKs (op 11) — so a socket that only
    ACKs never moves the stamp. ``deliver_dispatch`` models a parsed event
    reaching ``Client.dispatch``.
    """

    async def deliver_dispatch(self, event_type: str = "MESSAGE_CREATE") -> None:
        handler = self._events.get("on_socket_event_type")
        if handler is None:
            raise AssertionError("adapter did not register on_socket_event_type")
        # Client.dispatch schedules the handler as a task; awaiting it inline
        # is equivalent for this trivially non-blocking handler and lets the
        # caller observe the stamp immediately.
        await handler(event_type)


def _make_adapter(
    monkeypatch,
    *,
    interval: float = 0.01,
    threshold: int = 1,
    max_ack_age: float = 60.0,
    max_latency: float = 30.0,
    max_event_silence: float = 14400.0,
) -> DiscordAdapter:
    monkeypatch.setenv("HERMES_DISCORD_LIVENESS_INTERVAL_SECONDS", str(interval))
    monkeypatch.setenv("HERMES_DISCORD_LIVENESS_FAILURE_THRESHOLD", str(threshold))
    return DiscordAdapter(
        PlatformConfig(
            enabled=True,
            token="test-token",
            extra={
                "websocket_heartbeat_ack_max_age_seconds": max_ack_age,
                "websocket_max_latency_seconds": max_latency,
                "websocket_event_max_silence_seconds": max_event_silence,
            },
        )
    )


async def _connect(adapter: DiscordAdapter, monkeypatch, bot_factory) -> FakeBot:
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
    return adapter._client


def _transport_healthy(bot: _DispatchingBot) -> None:
    """Make every transport-side sample read healthy (incident 2's fingerprint)."""
    _set_websocket_health(bot, ready=True, socket_open=True, latency=0.05, ack_age=0.0)


@pytest.mark.asyncio
async def test_deaf_socket_trips_event_silence_dimension(monkeypatch):
    """Incident 2 e2e: transport-green + event-starved must trip the probe.

    The probe samples through ``_liveness_loop``, the real dispatch surface
    (no direct ``_read_websocket_health`` call), so this also proves the
    dimension is gated inside the health check and not in the startup guard.
    """
    adapter = _make_adapter(monkeypatch, interval=0.01, threshold=2, max_event_silence=0.05)
    handler = AsyncMock()
    adapter.set_fatal_error_handler(handler)

    def factory(**kwargs):
        bot = _DispatchingBot(intents=kwargs["intents"], allowed_mentions=kwargs.get("allowed_mentions"))
        bot.fetch_user = AsyncMock()
        return bot

    bot = await _connect(adapter, monkeypatch, factory)
    _transport_healthy(bot)

    # One early event arms the stamp; then total DISPATCH silence while every
    # transport sample stays green.
    await bot.deliver_dispatch("READY")

    async def handler_awaited() -> None:
        # _liveness_loop sets the fatal code, then hands off to
        # _notify_liveness_fatal_error (a separate task that closes the client
        # — up to a 1s budget — before notifying the runner), so wait for the
        # handler itself, not just the code.
        while True:
            if handler.await_count:
                return
            code = getattr(adapter, "_fatal_error_code", None)
            if code and adapter._liveness_notification_task is not None:
                with_context = adapter._liveness_notification_task
                try:
                    await asyncio.wait_for(asyncio.shield(with_context), timeout=3.0)
                except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
                    pass
                if handler.await_count:
                    return
            await asyncio.sleep(0.01)

    await asyncio.wait_for(handler_awaited(), timeout=8.0)
    assert adapter._fatal_error_code == "discord_websocket_health_stale"
    assert "event_silence" in (adapter._fatal_error_message or "")
    assert adapter._fatal_error_retryable is True
    handler.assert_awaited_once()




@pytest.mark.asyncio
async def test_missing_stamp_before_first_event_is_not_silence(monkeypatch):
    """A connected client with no DISPATCH event yet must not read as deaf.

    ``None`` means "nothing parsed on this connection" — the pre-READY
    window; ``not_ready`` owns that failure shape. Treating ``None`` as
    silence would false-trip fresh reconnects on quiet guilds.
    """
    adapter = _make_adapter(monkeypatch, interval=0.01, threshold=1, max_event_silence=0.05)

    def factory(**kwargs):
        bot = _DispatchingBot(intents=kwargs["intents"], allowed_mentions=kwargs.get("allowed_mentions"))
        bot.fetch_user = AsyncMock()
        return bot

    bot = await _connect(adapter, monkeypatch, factory)
    _transport_healthy(bot)
    assert adapter._last_dispatched_event_monotonic is None

    deadline = asyncio.get_running_loop().time() + 0.4
    while asyncio.get_running_loop().time() < deadline:
        healthy, reason = adapter._read_websocket_health(bot)
        assert healthy is True, f"None-stamp window must read healthy, got {reason}"
        await asyncio.sleep(0.05)

    await adapter.disconnect()
