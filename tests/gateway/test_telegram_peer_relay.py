import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from telegram.error import TimedOut
from telegram.constants import ParseMode

from gateway.config import PlatformConfig
from gateway.platforms.telegram import TelegramAdapter


def _make_adapter(peer_relays=None):
    extra = {}
    if peer_relays is not None:
        extra["bot_peer_relays"] = peer_relays
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***", extra=extra))
    adapter._bot = SimpleNamespace(
        send_message=AsyncMock(return_value=SimpleNamespace(message_id=321)),
        username="superwing_bot",
    )
    return adapter


@pytest.mark.asyncio
async def test_send_shadows_configured_peer_mentions_to_api_server():
    adapter = _make_adapter(
        [
            {
                "username": "Draco_hermes_bot",
                "endpoint": "http://peer.example/api/telegram/bot-relay",
                "api_key": "***",
            }
        ]
    )
    adapter._shadow_peer_relays = AsyncMock()

    result = await adapter.send(
        "-1001",
        "@Draco_hermes_bot 你看得到我吗？",
        metadata={"thread_id": "17"},
    )

    assert result.success is True
    await asyncio.sleep(0)
    adapter._shadow_peer_relays.assert_awaited_once_with(
        chat_id="-1001",
        content="@Draco_hermes_bot 你看得到我吗？",
        message_id="321",
        metadata={"thread_id": "17"},
    )


@pytest.mark.asyncio
async def test_first_peer_relay_payload_uses_hop_zero():
    adapter = _make_adapter(
        [
            {
                "username": "Draco_hermes_bot",
                "endpoint": "http://peer.example/api/telegram/bot-relay",
                "api_key": "***",
            }
        ]
    )
    adapter._bot = SimpleNamespace(
        send_message=AsyncMock(return_value=SimpleNamespace(message_id=321)),
        username="Superwing_draco_hermes_bot",
        id=8657839499,
    )
    adapter._post_peer_relay = AsyncMock()

    result = await adapter.send(
        "-1001",
        "@Draco_hermes_bot 你看得到我吗？",
        metadata={"thread_id": "17"},
    )

    assert result.success is True
    if adapter._background_tasks:
        await asyncio.gather(*list(adapter._background_tasks), return_exceptions=True)
    adapter._post_peer_relay.assert_awaited_once()
    _, payload = adapter._post_peer_relay.await_args.args
    assert payload["peer_relay_hop"] == 0


@pytest.mark.asyncio
async def test_send_skips_peer_relay_when_suppressed():
    adapter = _make_adapter(
        [
            {
                "username": "Draco_hermes_bot",
                "endpoint": "http://peer.example/api/telegram/bot-relay",
            }
        ]
    )
    adapter._shadow_peer_relays = AsyncMock()

    with patch.object(adapter, "_peer_relays_suppressed", return_value=True):
        result = await adapter.send(
            "-1001",
            "@Draco_hermes_bot 这条不该被转发",
            metadata={"thread_id": "17"},
        )

    assert result.success is True
    await asyncio.sleep(0)
    adapter._shadow_peer_relays.assert_not_awaited()


@pytest.mark.asyncio
async def test_send_timeout_still_schedules_peer_relay_fallback():
    adapter = _make_adapter(
        [
            {
                "username": "Draco_hermes_bot",
                "endpoint": "http://peer.example/api/telegram/bot-relay",
            }
        ]
    )
    adapter._bot = SimpleNamespace(
        send_message=AsyncMock(side_effect=TimedOut("timed out")),
        username="superwing_bot",
    )
    adapter._shadow_peer_relays = AsyncMock()

    result = await adapter.send(
        "-1001",
        "@Draco_hermes_bot 这条超时后也该被转发",
        metadata={"thread_id": "17"},
    )

    assert result.success is False
    assert result.retryable is False
    await asyncio.sleep(0)
    adapter._shadow_peer_relays.assert_awaited_once_with(
        chat_id="-1001",
        content="@Draco_hermes_bot 这条超时后也该被转发",
        message_id=None,
        metadata={"thread_id": "17"},
    )


@pytest.mark.asyncio
async def test_send_defers_peer_relay_when_requested():
    adapter = _make_adapter(
        [
            {
                "username": "Draco_hermes_bot",
                "endpoint": "http://peer.example/api/telegram/bot-relay",
            }
        ]
    )
    adapter._shadow_peer_relays = AsyncMock()

    result = await adapter.send(
        "-1001",
        "@Draco_hermes_bot partial ▉",
        metadata={"thread_id": "17", "defer_peer_relay": True},
    )

    assert result.success is True
    await asyncio.sleep(0)
    adapter._shadow_peer_relays.assert_not_awaited()


@pytest.mark.asyncio
async def test_schedule_deferred_peer_relay_reuses_normal_shadow_logic():
    adapter = _make_adapter(
        [
            {
                "username": "Draco_hermes_bot",
                "endpoint": "http://peer.example/api/telegram/bot-relay",
            }
        ]
    )
    adapter._shadow_peer_relays = AsyncMock()

    scheduled = adapter.schedule_deferred_peer_relay(
        chat_id="-1001",
        content="@Draco_hermes_bot 最终成句",
        message_id="321",
        metadata={"thread_id": "17", "defer_peer_relay": True},
    )

    assert scheduled is True
    await asyncio.sleep(0)
    adapter._shadow_peer_relays.assert_awaited_once_with(
        chat_id="-1001",
        content="@Draco_hermes_bot 最终成句",
        message_id="321",
        metadata={"thread_id": "17"},
    )


@pytest.mark.asyncio
async def test_send_allows_single_return_relay_under_suppression():
    adapter = _make_adapter(
        [
            {
                "username": "Superwing_draco_hermes_bot",
                "endpoint": "http://peer.example/api/telegram/bot-relay",
            }
        ]
    )
    adapter._bot = SimpleNamespace(
        send_message=AsyncMock(return_value=SimpleNamespace(message_id=654)),
        username="Draco_hermes_bot",
        id=42,
    )
    adapter._post_peer_relay = AsyncMock()

    with adapter._suppress_peer_relays():
        result = await adapter.send(
            "-1001",
            "@Superwing_draco_hermes_bot 桥通-RETURN1",
            metadata={
                "thread_id": "17",
                "peer_relay_sender_username": "Superwing_draco_hermes_bot",
                "peer_relay_hop": 0,
            },
        )

    assert result.success is True
    if adapter._background_tasks:
        await asyncio.gather(*list(adapter._background_tasks), return_exceptions=True)
    adapter._post_peer_relay.assert_awaited_once()
    relay, payload = adapter._post_peer_relay.await_args.args
    assert relay["username"] == "superwing_draco_hermes_bot"
    assert payload["peer_relay_hop"] == 1
    assert payload["text"] == "桥通-RETURN1"


@pytest.mark.asyncio
async def test_send_skips_second_return_hop_under_suppression():
    adapter = _make_adapter(
        [
            {
                "username": "Superwing_draco_hermes_bot",
                "endpoint": "http://peer.example/api/telegram/bot-relay",
            }
        ]
    )
    adapter._bot = SimpleNamespace(
        send_message=AsyncMock(return_value=SimpleNamespace(message_id=654)),
        username="Draco_hermes_bot",
        id=42,
    )
    adapter._post_peer_relay = AsyncMock()

    with adapter._suppress_peer_relays():
        result = await adapter.send(
            "-1001",
            "@Superwing_draco_hermes_bot 桥通-RETURN2",
            metadata={
                "thread_id": "17",
                "peer_relay_sender_username": "Superwing_draco_hermes_bot",
                "peer_relay_hop": 1,
            },
        )

    assert result.success is True
    await asyncio.sleep(0)
    adapter._post_peer_relay.assert_not_awaited()


@pytest.mark.asyncio
async def test_send_ignores_non_numeric_reply_to_tokens():
    adapter = _make_adapter()

    result = await adapter.send(
        "-1001",
        "桥接验证回复",
        reply_to="verify-r4m6",
        metadata={"thread_id": "17"},
    )

    assert result.success is True
    adapter._bot.send_message.assert_awaited_once_with(
        chat_id=-1001,
        text="桥接验证回复",
        parse_mode=ParseMode.MARKDOWN_V2,
        reply_to_message_id=None,
        message_thread_id=17,
        disable_notification=True,
    )
