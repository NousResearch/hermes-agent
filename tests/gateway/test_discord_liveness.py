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
) -> DiscordAdapter:
    monkeypatch.setenv("HERMES_DISCORD_LIVENESS_INTERVAL_SECONDS", str(interval))
    monkeypatch.setenv("HERMES_DISCORD_LIVENESS_FAILURE_THRESHOLD", str(threshold))
    monkeypatch.setenv("HERMES_DISCORD_HEARTBEAT_ACK_MAX_AGE_SECONDS", str(max_ack_age))
    monkeypatch.setenv("HERMES_DISCORD_MAX_LATENCY_SECONDS", str(max_latency))
    return DiscordAdapter(PlatformConfig(enabled=True, token="test-token"))


class _BrokenWebSocket:
    @property
    def open(self):
        raise RuntimeError("socket state unavailable")


@pytest.mark.parametrize(
    ("key", "attribute", "raw"),
    [
        ("HERMES_DISCORD_LIVENESS_INTERVAL_SECONDS", "_liveness_interval_seconds", "nan"),
        ("HERMES_DISCORD_HEARTBEAT_ACK_MAX_AGE_SECONDS", "_heartbeat_ack_max_age_seconds", "inf"),
        ("HERMES_DISCORD_MAX_LATENCY_SECONDS", "_max_latency_seconds", "-inf"),
    ],
)
def test_nonfinite_liveness_config_disables_that_probe_dimension(monkeypatch, key, attribute, raw):
    monkeypatch.setenv(key, raw)

    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="test-token"))

    assert getattr(adapter, attribute) == 0.0


def test_default_liveness_bounds_trigger_timed_recovery(monkeypatch):
    for key in (
        "HERMES_DISCORD_LIVENESS_INTERVAL_SECONDS",
        "HERMES_DISCORD_LIVENESS_FAILURE_THRESHOLD",
        "HERMES_DISCORD_HEARTBEAT_ACK_MAX_AGE_SECONDS",
        "HERMES_DISCORD_MAX_LATENCY_SECONDS",
    ):
        monkeypatch.delenv(key, raising=False)

    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="test-token"))

    assert adapter._liveness_interval_seconds == 20.0
    assert adapter._liveness_failure_threshold == 3
    assert adapter._heartbeat_ack_max_age_seconds == 75.0
    assert adapter._max_latency_seconds == 30.0


def test_platform_config_extra_overrides_process_liveness_bridge(monkeypatch):
    monkeypatch.setenv("HERMES_DISCORD_LIVENESS_INTERVAL_SECONDS", "99")
    monkeypatch.setenv("HERMES_DISCORD_LIVENESS_FAILURE_THRESHOLD", "9")
    monkeypatch.setenv("HERMES_DISCORD_HEARTBEAT_ACK_MAX_AGE_SECONDS", "999")
    monkeypatch.setenv("HERMES_DISCORD_MAX_LATENCY_SECONDS", "99")

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
async def test_successful_connection_clears_stale_websocket_health(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = _make_adapter(monkeypatch, interval=60)

    from gateway import status

    status.write_runtime_status(
        platform="discord",
        platform_state="degraded",
        health={"transport": "websocket", "last_health_reason": "ack_stale"},
    )

    def factory(**kwargs):
        return _LiveBot(
            intents=kwargs["intents"],
            allowed_mentions=kwargs.get("allowed_mentions"),
        )

    await _connect(adapter, monkeypatch, factory)

    payload = status.read_runtime_status()
    assert payload["platforms"]["discord"]["state"] == "connected"
    assert payload["platforms"]["discord"]["health"] is None
    await adapter.disconnect()


@pytest.mark.asyncio
async def test_liveness_probe_does_not_call_rest_while_websocket_is_healthy(monkeypatch):
    """A fresh Gateway ACK is sufficient; REST is not a transport health probe."""
    adapter = _make_adapter(monkeypatch, interval=0.01, threshold=3)

    def factory(**kwargs):
        bot = _LiveBot(intents=kwargs["intents"], allowed_mentions=kwargs.get("allowed_mentions"))
        _set_websocket_health(bot)
        bot.fetch_user = AsyncMock(return_value=SimpleNamespace(id=999))
        return bot

    await _connect(adapter, monkeypatch, factory)
    await asyncio.sleep(0.05)
    adapter._client.fetch_user.assert_not_awaited()
    assert adapter._running is True
    assert adapter.has_fatal_error is False
    await adapter.disconnect()


@pytest.mark.asyncio
async def test_liveness_probe_forces_reconnect_when_rest_succeeds_but_gateway_ack_is_stale(monkeypatch):
    """A REST 200 must not hide the CLOSE_WAIT / stale-ACK Gateway failure mode."""
    adapter = _make_adapter(monkeypatch, interval=0.005, threshold=2, max_ack_age=0.01)

    def factory(**kwargs):
        bot = _LiveBot(intents=kwargs["intents"], allowed_mentions=kwargs.get("allowed_mentions"))
        _set_websocket_health(bot, ack_age=3600)
        bot.fetch_user = AsyncMock(return_value=SimpleNamespace(id=999))
        return bot

    handler = AsyncMock()
    adapter.set_fatal_error_handler(handler)
    await _connect(adapter, monkeypatch, factory)
    wedged = adapter._client

    for _ in range(200):
        if adapter._liveness_task and adapter._liveness_task.done():
            break
        await asyncio.sleep(0.01)
    else:
        pytest.fail("liveness loop did not terminate within 2s")

    # The sampler schedules the close + supervisor callback in a sibling task
    # so the fatal path cannot cancel/await itself through disconnect().
    for _ in range(200):
        notification = adapter._liveness_notification_task
        if notification and notification.done():
            break
        await asyncio.sleep(0.01)
    else:
        pytest.fail("liveness recovery notification did not complete within 2s")

    assert wedged.is_closed() is True
    assert adapter.has_fatal_error is True
    assert adapter.fatal_error_code == "discord_websocket_health_stale"
    assert adapter.fatal_error_retryable is True
    wedged.fetch_user.assert_not_awaited()
    handler.assert_awaited_once()

    await adapter.disconnect()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("health", "expected_reason"),
    [
        ({"ready": False}, "not_ready"),
        ({"socket_open": False}, "socket_closed"),
        ({"latency": float("inf")}, "latency_non_finite"),
    ],
)
async def test_liveness_probe_reports_gateway_health_failure_reason(monkeypatch, health, expected_reason):
    adapter = _make_adapter(monkeypatch, interval=0.005, threshold=1)

    def factory(**kwargs):
        bot = _LiveBot(intents=kwargs["intents"], allowed_mentions=kwargs.get("allowed_mentions"))
        _set_websocket_health(bot, **health)
        bot.fetch_user = AsyncMock(return_value=SimpleNamespace(id=999))
        return bot

    handler = AsyncMock()
    adapter.set_fatal_error_handler(handler)
    await _connect(adapter, monkeypatch, factory)

    for _ in range(200):
        if adapter.has_fatal_error:
            break
        await asyncio.sleep(0.01)
    else:
        pytest.fail("liveness loop did not surface a websocket health failure")

    assert expected_reason in (adapter.fatal_error_message or "")
    adapter._client.fetch_user.assert_not_awaited()
    for _ in range(200):
        if handler.await_count:
            break
        await asyncio.sleep(0.01)
    handler.assert_awaited_once()
    await adapter.disconnect()




@pytest.mark.asyncio
async def test_liveness_probe_treats_websocket_state_read_error_as_unhealthy(monkeypatch):
    adapter = _make_adapter(monkeypatch, interval=0.005, threshold=1)

    def factory(**kwargs):
        bot = _LiveBot(intents=kwargs["intents"], allowed_mentions=kwargs.get("allowed_mentions"))
        bot.ws = _BrokenWebSocket()
        return bot

    handler = AsyncMock()
    adapter.set_fatal_error_handler(handler)
    await _connect(adapter, monkeypatch, factory)

    for _ in range(200):
        if adapter.has_fatal_error:
            break
        await asyncio.sleep(0.01)
    else:
        pytest.fail("liveness loop did not surface a WebSocket state read error")

    assert "socket_state_unavailable" in (adapter.fatal_error_message or "")
    for _ in range(200):
        if handler.await_count:
            break
        await asyncio.sleep(0.01)
    handler.assert_awaited_once()
    await adapter.disconnect()


@pytest.mark.asyncio
async def test_liveness_probe_recovers_when_health_reader_raises(monkeypatch):
    adapter = _make_adapter(monkeypatch, interval=0.005, threshold=1)

    def factory(**kwargs):
        return _LiveBot(
            intents=kwargs["intents"],
            allowed_mentions=kwargs.get("allowed_mentions"),
        )

    handler = AsyncMock()
    adapter.set_fatal_error_handler(handler)
    await _connect(adapter, monkeypatch, factory)
    monkeypatch.setattr(
        adapter,
        "_read_websocket_health",
        lambda _client: (_ for _ in ()).throw(RuntimeError("unexpected state")),
    )

    for _ in range(200):
        if adapter.has_fatal_error:
            break
        await asyncio.sleep(0.01)
    else:
        pytest.fail("liveness loop did not recover from health-reader failure")

    assert "health_check_error" in (adapter.fatal_error_message or "")
    for _ in range(200):
        if handler.await_count:
            break
        await asyncio.sleep(0.01)
    handler.assert_awaited_once()
    await adapter.disconnect()


@pytest.mark.asyncio
async def test_liveness_recovery_keeps_websocket_fatal_when_client_task_exits(monkeypatch):
    """The close callback must not replace stale-ACK recovery with task-exited."""
    adapter = _make_adapter(monkeypatch, interval=0.005, threshold=1, max_ack_age=0.01)

    def factory(**kwargs):
        bot = _LiveBot(intents=kwargs["intents"], allowed_mentions=kwargs.get("allowed_mentions"))
        _set_websocket_health(bot, ack_age=3600)
        return bot

    handler = AsyncMock()
    adapter.set_fatal_error_handler(handler)
    await _connect(adapter, monkeypatch, factory)

    for _ in range(200):
        if adapter._bot_task and adapter._bot_task.done():
            break
        await asyncio.sleep(0.01)
    else:
        pytest.fail("closed client task did not finish within 2s")

    assert adapter.fatal_error_code == "discord_websocket_health_stale"
    assert handler.await_count == 1
    await adapter.disconnect()


@pytest.mark.asyncio
async def test_liveness_recovery_not_blocked_by_hanging_client_close(monkeypatch):
    """A wedged close must not prevent fatal notification/reconnect queueing."""
    adapter = _make_adapter(monkeypatch, interval=60, threshold=1, max_ack_age=1.0)
    monkeypatch.setenv("HERMES_GATEWAY_ADAPTER_DISCONNECT_TIMEOUT", "0.02")

    def factory(**kwargs):
        bot = _LiveBot(intents=kwargs["intents"], allowed_mentions=kwargs.get("allowed_mentions"))
        _set_websocket_health(bot, ack_age=3600)
        bot.fetch_user = AsyncMock(return_value=SimpleNamespace(id=999))
        return bot

    handler = AsyncMock()
    adapter.set_fatal_error_handler(handler)
    await _connect(adapter, monkeypatch, factory)
    wedged = adapter._client
    close_started = asyncio.Event()

    async def hanging_close():
        close_started.set()
        await asyncio.Event().wait()

    wedged.close = hanging_close
    adapter._set_fatal_error(
        "discord_websocket_health_stale",
        "Discord Gateway WebSocket health check failed: ack_stale",
        retryable=True,
    )
    notify_task = asyncio.create_task(adapter._notify_liveness_fatal_error(wedged))
    await asyncio.wait_for(close_started.wait(), timeout=0.5)
    await asyncio.wait_for(notify_task, timeout=2.0)
    assert close_started.is_set() is True
    assert handler.await_count == 1
    assert adapter.fatal_error_code == "discord_websocket_health_stale"

    # Restore a cooperative fake close so the test can release the bot task.
    wedged.close = _LiveBot.close.__get__(wedged, _LiveBot)
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
