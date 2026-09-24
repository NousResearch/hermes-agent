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
import time
from unittest.mock import AsyncMock

import pytest

from tests.gateway.test_discord_connect import _ensure_discord_mock  # noqa: E402

_ensure_discord_mock()

from tests.gateway.test_discord_liveness import (  # noqa: E402
    _LiveBot,
    _connect,
    _make_adapter,
    _set_websocket_health,
    _wait_until,
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


def _dispatching_bot(**kwargs) -> _DispatchingBot:
    bot = _DispatchingBot(intents=kwargs["intents"], allowed_mentions=kwargs.get("allowed_mentions"))
    bot.fetch_user = AsyncMock()
    return bot


def _transport_healthy(bot: _DispatchingBot, *, ack_age: float = 0.0) -> None:
    """Make every transport-side sample read healthy (incident 2's fingerprint)."""
    _set_websocket_health(bot, ready=True, socket_open=True, latency=0.05, ack_age=ack_age)


@pytest.mark.asyncio
async def test_deaf_socket_trips_event_silence_dimension(monkeypatch):
    """Incident 2 e2e: transport-green + event-starved must trip the probe.

    Sampled through ``_liveness_loop`` (the real dispatch surface). Before the
    first DISPATCH the stamp is ``None`` — "nothing parsed yet", owned by
    ``not_ready`` — and must not read as silence, or fresh reconnects on quiet
    guilds would false-trip.
    """
    adapter = _make_adapter(monkeypatch, interval=0.01, threshold=2, max_event_silence=0.05)
    handler = AsyncMock()
    adapter.set_fatal_error_handler(handler)

    await _connect(adapter, monkeypatch, _dispatching_bot)
    bot = adapter._client
    _transport_healthy(bot)
    assert adapter._last_dispatched_event_monotonic is None

    # Several probe intervals past the silence bound with no stamp: still healthy.
    await asyncio.sleep(0.2)
    assert adapter._fatal_error_code is None
    assert adapter._read_websocket_health(bot) == (True, "healthy")

    # One early event arms the stamp; then total DISPATCH silence while every
    # transport sample stays green.
    await bot.deliver_dispatch("READY")

    # _liveness_loop sets the code, then a separate task closes the client (1s
    # budget) before notifying the runner — so wait for the handler itself.
    await _wait_until(lambda: handler.await_count, "fatal handler never awaited", timeout=8.0)
    assert adapter._fatal_error_code == "discord_websocket_health_stale"
    assert "event_silence" in (adapter._fatal_error_message or "")
    assert adapter._fatal_error_retryable is True
    handler.assert_awaited_once()


@pytest.mark.asyncio
async def test_zero_silence_bound_disables_only_that_dimension(monkeypatch):
    """``websocket_event_max_silence_seconds: 0`` must not switch the watchdog off.

    The #109782 regression put the knob in ``_start_liveness_probe``'s
    all-or-nothing guard, so opting out of event-silence also dropped the
    ack-age/latency guards. With a stale stamp AND a stale heartbeat ACK the
    probe must still run and trip on ``ack_stale``.
    """
    adapter = _make_adapter(monkeypatch, interval=0.01, threshold=1, max_ack_age=60.0, max_event_silence=0)
    handler = AsyncMock()
    adapter.set_fatal_error_handler(handler)

    await _connect(adapter, monkeypatch, _dispatching_bot)
    bot = adapter._client
    adapter._last_dispatched_event_monotonic = time.perf_counter() - 1000.0

    # Knob at 0: a stale stamp alone must read healthy (the dimension is off),
    # with every transport-side sample green.
    _transport_healthy(bot)
    assert adapter._read_websocket_health(bot) == (True, "healthy")

    # Same stale stamp, now with a stale heartbeat ACK: the transport guards
    # still trip, and only the ack-age dimension is named.
    _transport_healthy(bot, ack_age=120.0)

    await _wait_until(lambda: handler.await_count, "probe did not run with the knob at 0", timeout=8.0)
    assert adapter._fatal_error_code == "discord_websocket_health_stale"
    assert "ack_stale" in (adapter._fatal_error_message or "")
    assert "event_silence" not in (adapter._fatal_error_message or "")
