"""Telegram send() must wait for reconnect instead of dropping the final reply.

A network blip can disconnect Telegram while the agent is sending its final
response. send() used to return "Not connected" immediately (retryable=False),
so the delivery ledger held the answer until the next gateway boot — hours
later. QQBot already waits; these tests pin the same contract on Telegram,
including the reconnect-watcher case that *replaces* the adapter object.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter


def _make_adapter() -> TelegramAdapter:
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    adapter._rich_send_disabled = True
    adapter.send_typing = AsyncMock()
    adapter._RECONNECT_WAIT_SECONDS = 0.6
    adapter._RECONNECT_POLL_INTERVAL = 0.05
    return adapter


def _connected_bot() -> MagicMock:
    bot = MagicMock()
    bot.send_message = AsyncMock(return_value=MagicMock(message_id=42))
    return bot


@pytest.mark.asyncio
async def test_send_waits_and_succeeds_when_bot_returns():
    adapter = _make_adapter()
    adapter._bot = None

    async def restore_bot() -> None:
        await asyncio.sleep(0.12)
        adapter._bot = _connected_bot()

    asyncio.get_running_loop().create_task(restore_bot())
    result = await adapter.send("123", "hello")

    assert result.success is True
    assert result.message_id == "42"
    adapter._bot.send_message.assert_awaited()


@pytest.mark.asyncio
async def test_send_delegates_to_replacement_adapter_installed_mid_wait():
    old = _make_adapter()
    old._bot = None

    live = _make_adapter()
    live._bot = _connected_bot()
    live._RECONNECT_WAIT_SECONDS = 0.01

    runner = MagicMock()
    runner.adapters = {}
    runner._adapters_for_profile = lambda _profile: runner.adapters
    old.gateway_runner = runner

    async def install_replacement() -> None:
        await asyncio.sleep(0.12)
        runner.adapters[old.platform] = live

    asyncio.get_running_loop().create_task(install_replacement())
    result = await old.send("123", "hello")

    assert result.success is True
    assert result.message_id == "42"
    live._bot.send_message.assert_awaited()


@pytest.mark.asyncio
async def test_send_delegates_immediately_when_replacement_already_live():
    old = _make_adapter()
    old._bot = None
    live = _make_adapter()
    live._bot = _connected_bot()
    runner = MagicMock()
    runner.adapters = {old.platform: live}
    runner._adapters_for_profile = lambda _profile: runner.adapters
    old.gateway_runner = runner
    old._wait_for_reconnection = AsyncMock(
        side_effect=AssertionError("must not wait when replacement is already live")
    )

    result = await old.send("123", "hello")

    assert result.success is True
    live._bot.send_message.assert_awaited()


@pytest.mark.asyncio
async def test_send_timeout_is_retryable_not_connected():
    adapter = _make_adapter()
    adapter._bot = None
    adapter._RECONNECT_WAIT_SECONDS = 0.15
    adapter._RECONNECT_POLL_INTERVAL = 0.05

    result = await adapter.send("123", "hello")

    assert result.success is False
    assert result.error == "Not connected"
    assert result.retryable is True


@pytest.mark.asyncio
async def test_send_permanent_fatal_fails_immediately_without_wait():
    adapter = _make_adapter()
    adapter._bot = None
    adapter._set_fatal_error("telegram_auth_error", "invalid token", retryable=False)
    adapter._wait_for_reconnection = AsyncMock(
        side_effect=AssertionError("must not wait on permanent fatal")
    )

    result = await adapter.send("123", "hello")

    assert result.success is False
    assert result.error == "Not connected"
    assert result.retryable is False


@pytest.mark.asyncio
async def test_reset_after_the_request_was_written_is_never_resent():
    """A connection reset after sendMessage was written may already be in the chat (#64238)."""
    import httpx
    from telegram.error import NetworkError

    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    adapter._bot = MagicMock()
    lost = NetworkError("httpx.RemoteProtocolError: Server disconnected without sending a response.")
    lost.__cause__ = httpx.RemoteProtocolError("Server disconnected without sending a response.")
    adapter._bot.send_message = AsyncMock(side_effect=lost)
    adapter._bot.send_chat_action = AsyncMock()

    result = await adapter._send_with_retry("4242", "hello", max_retries=2, base_delay=0)

    assert result.success is False and result.retryable is False
    assert adapter._bot.send_message.await_count == 1  # no in-loop resend, no gateway retry, no plain fallback

    refused = NetworkError("httpx.ConnectError: All connection attempts failed")
    refused.__cause__ = httpx.ConnectError("All connection attempts failed")
    adapter._bot.send_message = AsyncMock(side_effect=[refused, MagicMock(message_id=9)])
    with patch("plugins.platforms.telegram.adapter.asyncio.sleep", AsyncMock()):
        assert (await adapter.send("4242", "hello")).success  # never left: retried and delivered


def test_replacement_lookup_stays_inside_the_adapters_own_profile():
    """#143364: a torn-down secondary must never borrow the primary's bot.

    ``runner.adapters`` holds only the primary map; the replacement lookup
    goes through ``_adapters_for_profile(owner)`` so a secondary with no
    live replacement fails closed instead of sending under the default
    profile's identity."""
    secondary = _make_adapter()
    secondary._bot = None
    secondary._owner_profile = "yori"

    primary_live = _make_adapter()
    primary_live._bot = _connected_bot()

    asked = []

    def _for_profile(profile):
        asked.append(profile)
        # The gate answers {} for a named profile with no live adapter.
        return {} if profile == "yori" else {secondary.platform: primary_live}

    runner = MagicMock()
    runner.adapters = {secondary.platform: primary_live}
    runner._adapters_for_profile = _for_profile
    secondary.gateway_runner = runner

    assert secondary._replacement_telegram_adapter() is None
    assert asked == ["yori"]


def test_replacement_lookup_returns_same_profile_replacement():
    secondary = _make_adapter()
    secondary._bot = None
    secondary._owner_profile = "yori"

    replacement = _make_adapter()
    replacement._bot = _connected_bot()

    runner = MagicMock()
    runner._adapters_for_profile = lambda _profile: {secondary.platform: replacement}
    secondary.gateway_runner = runner

    assert secondary._replacement_telegram_adapter() is replacement
