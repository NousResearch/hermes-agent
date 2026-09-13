"""Regression tests for Discord Gateway WebSocket liveness.

A Discord REST response and the Gateway WebSocket are independent transports.
A half-closed Gateway socket can leave ``Bot.start()`` alive while REST still
returns 200, so health must come from the active WebSocket's ready/open/ACK and
heartbeat-latency state rather than ``fetch_user()``.
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

# Re-use the shared discord-stub bootstrap and FakeBot from the connect
# test module so this file doesn't duplicate the (large) mock surface.
from tests.gateway.test_discord_connect import (  # noqa: E402
    FakeBot,
    _ensure_discord_mock,
)

_ensure_discord_mock()

import plugins.platforms.discord.adapter as discord_platform  # noqa: E402
from gateway.config import Platform, PlatformConfig  # noqa: E402
from gateway.run import GatewayRunner  # noqa: E402
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
        self._gateway_ready = True
        self.latency = 0.05
        self.ws = _FakeWebSocket()

    def is_ready(self):
        return self._gateway_ready

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


class _FakeKeepAlive:
    def __init__(self, *, ack_age: float = 0.0):
        self._last_ack = time.perf_counter() - ack_age


class _FakeWebSocket:
    def __init__(self, *, open: bool = True, ack_age: float = 0.0):
        self.open = open
        self._keep_alive = _FakeKeepAlive(ack_age=ack_age)


def _set_websocket_health(
    bot: _LiveBot,
    *,
    ready: bool = True,
    socket_open: bool = True,
    latency: float = 0.05,
    ack_age: float = 0.0,
) -> None:
    bot._gateway_ready = ready
    bot.latency = latency
    bot.ws = _FakeWebSocket(open=socket_open, ack_age=ack_age)


def _make_adapter(
    monkeypatch,
    *,
    interval=0.01,
    threshold=1,
    max_ack_age=1.0,
    max_latency=1.0,
    max_silence=300.0,
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
                "websocket_event_max_silence_seconds": max_silence,
            },
        )
    )


class _BrokenWebSocket:
    @property
    def open(self):
        raise RuntimeError("socket state unavailable")


@pytest.mark.parametrize(
    ("key", "attribute", "raw"),
    [
        ("websocket_liveness_interval_seconds", "_liveness_interval_seconds", "nan"),
        ("websocket_heartbeat_ack_max_age_seconds", "_heartbeat_ack_max_age_seconds", "inf"),
        ("websocket_max_latency_seconds", "_max_latency_seconds", "-inf"),
        ("websocket_event_max_silence_seconds", "_event_max_silence_seconds", "15s"),
    ],
)
def test_nonfinite_liveness_config_disables_that_probe_dimension(monkeypatch, key, attribute, raw):
    adapter = DiscordAdapter(
        PlatformConfig(enabled=True, token="test-token", extra={key: raw})
    )

    assert getattr(adapter, attribute) == 0.0


def test_unparsable_liveness_config_warns_instead_of_disabling_silently(monkeypatch, caplog):
    """A knob value that can't parse must not disable the probe without a trace (#109521).

    Pre-fix, ``websocket_liveness_interval_seconds: 15s`` mapped to 0.0 with no log line —
    the watchdog was off and the only visible symptom was hours of Discord silence.
    """
    with caplog.at_level("WARNING", logger="plugins.platforms.discord.adapter"):
        adapter = DiscordAdapter(
            PlatformConfig(
                enabled=True,
                token="test-token",
                extra={"websocket_liveness_interval_seconds": "15s"},
            )
        )

    assert adapter._liveness_interval_seconds == 0.0
    assert "websocket_liveness_interval_seconds" in caplog.text
    assert "15s" in caplog.text


def test_default_liveness_bounds_trigger_timed_recovery(monkeypatch):
    for key in (
        "HERMES_DISCORD_LIVENESS_INTERVAL_SECONDS",
        "HERMES_DISCORD_LIVENESS_FAILURE_THRESHOLD",
    ):
        monkeypatch.delenv(key, raising=False)

    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="test-token"))

    assert adapter._liveness_interval_seconds == 15.0
    assert adapter._liveness_failure_threshold == 2
    assert adapter._heartbeat_ack_max_age_seconds == 60.0
    assert adapter._max_latency_seconds == 30.0
    assert adapter._event_max_silence_seconds == 300.0


def test_platform_config_extra_overrides_process_liveness_bridge(monkeypatch):
    monkeypatch.setenv("HERMES_DISCORD_LIVENESS_INTERVAL_SECONDS", "99")
    monkeypatch.setenv("HERMES_DISCORD_LIVENESS_FAILURE_THRESHOLD", "9")

    adapter = DiscordAdapter(
        PlatformConfig(
            enabled=True,
            token="test-token",
            extra={
                "websocket_liveness_interval_seconds": 7,
                "websocket_liveness_failure_threshold": 2,
                "websocket_heartbeat_ack_max_age_seconds": 45,
                "websocket_max_latency_seconds": 12,
            },
        )
    )

    assert adapter._liveness_interval_seconds == 7
    assert adapter._liveness_failure_threshold == 2
    assert adapter._heartbeat_ack_max_age_seconds == 45
    assert adapter._max_latency_seconds == 12


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


async def _wait_until(predicate, message: str, timeout: float = 2.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() >= deadline:
            pytest.fail(message)
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_liveness_close_timeout_aborts_aiohttp_transport_before_fatal_notification(
    monkeypatch,
):
    """A close handshake timeout must abort the stale socket before reconnect."""
    adapter = _make_adapter(monkeypatch, interval=60, threshold=1, max_ack_age=1.0)
    handler = AsyncMock()
    adapter.set_fatal_error_handler(handler)

    close_started = asyncio.Event()
    release_close = asyncio.Event()

    async def hanging_close():
        close_started.set()
        while not release_close.is_set():
            try:
                await release_close.wait()
            except asyncio.CancelledError:
                # Model a close path that catches cancellation while unwinding.
                continue

    transport = Mock()
    replacement_transport = Mock()
    aiohttp_socket = SimpleNamespace(
        close=hanging_close,
        # aiohttp clears response.connection while cancellation unwinds close(),
        # but its WebSocket writer still owns the underlying transport.
        _response=SimpleNamespace(connection=None),
        _conn=None,
        _writer=SimpleNamespace(transport=transport),
    )
    gateway_websocket = SimpleNamespace(socket=aiohttp_socket)
    replacement_websocket = SimpleNamespace(
        socket=SimpleNamespace(
            _response=SimpleNamespace(connection=None),
            _conn=None,
            _writer=SimpleNamespace(transport=replacement_transport),
        )
    )

    class _StickyCloseClient:
        def __init__(self):
            self.ws = gateway_websocket
            self._closing_task = None
            self.close_attempts = 0

        async def close(self):
            if self._closing_task is not None:
                return await self._closing_task

            async def _close():
                self.close_attempts += 1
                if self.close_attempts == 1:
                    # The library may publish a replacement WebSocket while the
                    # old close handshake is still stuck. Recovery must never
                    # abort the replacement transport.
                    self.ws = replacement_websocket
                    await hanging_close()

            self._closing_task = asyncio.create_task(_close())
            return await self._closing_task

    client = _StickyCloseClient()

    adapter._set_fatal_error(
        "discord_websocket_health_stale",
        "Discord Gateway WebSocket health check failed: socket_closed",
        retryable=True,
    )
    notify_task = asyncio.create_task(adapter._notify_liveness_fatal_error(client))

    await asyncio.wait_for(close_started.wait(), timeout=0.5)
    done, _pending = await asyncio.wait({notify_task}, timeout=1.5)
    finished_within_bound = notify_task in done
    release_close.set()
    if not notify_task.done():
        await asyncio.wait_for(notify_task, timeout=0.5)

    assert finished_within_bound is True
    transport.abort.assert_called_once_with()
    replacement_transport.abort.assert_not_called()
    handler.assert_awaited_once()
    assert client._closing_task is None
    await client.close()
    assert client.close_attempts == 2


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


class _DeafHealthBot(_LiveBot):
    """Incident-2 fingerprint from #109521: ESTAB socket, keep-alive ACKs, zero frames."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.raw_receive_events = []

    async def deliver_raw_frame(self, payload: str = "{}"):
        """Feed a raw gateway frame through the adapter's registered hook."""
        await self._events["on_socket_raw_receive"](payload)


def _connect_deaf_bot(adapter, monkeypatch, bot_box):
    def factory(**kwargs):
        bot = _DeafHealthBot(
            intents=kwargs["intents"],
            allowed_mentions=kwargs.get("allowed_mentions"),
        )
        bot.fetch_user = AsyncMock()
        bot_box.append(bot)
        return bot

    return _connect(adapter, monkeypatch, factory)


@pytest.mark.asyncio
async def test_connected_but_deaf_socket_is_unhealthy(monkeypatch):
    """#109521 incident 2: a socket that stays open, ready, low-latency, and ACKing —
    but through which no gateway frame has arrived since before the silence bound —
    must read unhealthy, not healthy."""
    adapter = _make_adapter(
        monkeypatch, interval=60, threshold=1, max_ack_age=60.0, max_latency=30.0,
        max_silence=10.0,
    )
    box = []
    await _connect_deaf_bot(adapter, monkeypatch, box)
    bot = box[0]

    _set_websocket_health(bot, ready=True, socket_open=True, latency=0.05, ack_age=0.0)
    # on_ready reset the stamp at connect; age it past the bound so the adapter is
    # "connected, all transport dimensions green, and silent" — incident 2's fingerprint.
    adapter._last_gateway_frame_at = time.perf_counter() - 60.0

    healthy, reason = adapter._read_websocket_health(bot)
    assert healthy is False
    assert reason == "event_silence"

    await adapter.disconnect()


@pytest.mark.asyncio
async def test_recent_raw_frame_keeps_deaf_fingerprint_socket_healthy(monkeypatch):
    """Any raw gateway frame (heartbeat, ACK — not just messages) refreshes the dispatch
    clock: a legitimately quiet server must not be flagged by the event-age bound."""
    adapter = _make_adapter(
        monkeypatch, interval=60, threshold=1, max_ack_age=60.0, max_latency=30.0,
    )
    box = []
    await _connect_deaf_bot(adapter, monkeypatch, box)
    bot = box[0]

    _set_websocket_health(bot, ready=True, socket_open=True, latency=0.05, ack_age=0.0)
    # on_ready resets the stamp (fresh connection); a non-message frame then advances it.
    assert adapter._last_gateway_frame_at > 0
    await bot.deliver_raw_frame('{"t":null,"op":11}')
    frame_at = adapter._last_gateway_frame_at
    assert frame_at > 0

    healthy, reason = adapter._read_websocket_health(bot)
    assert (healthy, reason) == (True, "healthy")

    await adapter.disconnect()


@pytest.mark.asyncio
async def test_stale_frame_over_silence_bound_reads_unhealthy(monkeypatch):
    """A frame that arrived, then a silence longer than the bound, while transport
    dimensions still look healthy — the exact connected-but-deaf progression."""
    adapter = _make_adapter(
        monkeypatch, interval=60, threshold=1, max_ack_age=60.0, max_latency=30.0,
        max_silence=10.0,
    )
    box = []
    await _connect_deaf_bot(adapter, monkeypatch, box)
    bot = box[0]

    _set_websocket_health(bot, ready=True, socket_open=True, latency=0.05, ack_age=0.0)
    await bot.deliver_raw_frame('{"t":null,"op":11}')
    # Simulate the silence: age the last frame past every bound, keeping the transport green.
    adapter._last_gateway_frame_at = time.perf_counter() - 60.0

    healthy, reason = adapter._read_websocket_health(bot)
    assert (healthy, reason) == (False, "event_silence")

    await adapter.disconnect()


@pytest.mark.asyncio
async def test_event_silence_drives_liveness_loop_to_retryable_fatal(monkeypatch, caplog):
    """End to end: a persistent deaf socket trips the probe's failure threshold and
    surfaces as the retryable fatal the reconnect watcher already handles."""
    import logging

    caplog.at_level(logging.WARNING, logger="plugins.platforms.discord.adapter")
    adapter = _make_adapter(
        monkeypatch, interval=0.01, threshold=2, max_ack_age=60.0, max_latency=30.0,
        max_silence=5.0,
    )
    handler = AsyncMock()
    adapter.set_fatal_error_handler(handler)
    box = []
    await _connect_deaf_bot(adapter, monkeypatch, box)
    bot = box[0]

    # Transport dimensions stay green the whole time; frames never arrive.
    _set_websocket_health(bot, ready=True, socket_open=True, latency=0.05, ack_age=0.0)
    adapter._last_gateway_frame_at = time.perf_counter() - 60.0

    await _wait_until(
        lambda: adapter.fatal_error_code == "discord_websocket_health_stale",
        "deaf socket never tripped the liveness probe",
    )
    assert "event_silence" in caplog.text
    # The runner-facing handler fires from the notification task (post-close); give it a beat.
    await _wait_until(
        lambda: handler.await_count >= 1,
        "fatal notification never reached the runner handler",
    )

    await adapter.disconnect()


@pytest.mark.asyncio
async def test_silence_knob_zero_disables_event_dimension_only(monkeypatch):
    """Setting ``websocket_event_max_silence_seconds: 0`` must opt out of the dispatch
    check alone; the transport dimensions (ack age, latency) still work."""
    adapter = DiscordAdapter(
        PlatformConfig(
            enabled=True,
            token="test-token",
            extra={
                "websocket_event_max_silence_seconds": 0,
                "websocket_heartbeat_ack_max_age_seconds": 60.0,
                "websocket_max_latency_seconds": 30.0,
            },
        )
    )
    assert adapter._event_max_silence_seconds == 0.0

    box = []
    await _connect_deaf_bot(adapter, monkeypatch, box)
    bot = box[0]
    # Never delivered a frame and the stamp is at its connect-time value.
    adapter._last_gateway_frame_at = 0.0

    _set_websocket_health(bot, ready=True, socket_open=True, latency=0.05, ack_age=0.0)
    healthy, reason = adapter._read_websocket_health(bot)
    assert (healthy, reason) == (True, "healthy")

    # While the transport dimensions still catch their own failure shape:
    _set_websocket_health(bot, ready=True, socket_open=True, latency=0.05, ack_age=999.0)
    healthy, reason = adapter._read_websocket_health(bot)
    assert (healthy, reason) == (False, "ack_stale")

    await adapter.disconnect()
