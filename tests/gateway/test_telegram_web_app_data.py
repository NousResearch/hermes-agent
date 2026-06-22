"""Tests for Telegram Mini App WebApp data handling."""

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageType


def _make_adapter():
    from plugins.platforms.telegram.adapter import TelegramAdapter

    adapter = cast(Any, object.__new__(TelegramAdapter))
    adapter.platform = Platform.TELEGRAM
    adapter.config = PlatformConfig(enabled=True, token="test-token")
    adapter._dm_topic_chat_ids = set()
    adapter._forum_lock = asyncio.Lock()
    adapter._forum_command_registered = set()
    adapter._bot = None
    adapter.handle_message = AsyncMock()
    adapter._ensure_forum_commands = AsyncMock()
    adapter._apply_telegram_group_observe_attribution = lambda event: event
    return adapter


def _make_web_app_update(data: str, button_text: str = "Open Mini App"):
    chat = SimpleNamespace(
        id=6168627430,
        type="private",
        title=None,
        full_name="Filip Václavík",
        is_forum=False,
    )
    user = SimpleNamespace(id=6168627430, full_name="Filip Václavík")
    message = SimpleNamespace(
        chat=chat,
        from_user=user,
        text=None,
        message_id=123,
        message_thread_id=None,
        is_topic_message=False,
        reply_to_message=None,
        date=datetime.now(timezone.utc),
        web_app_data=SimpleNamespace(data=data, button_text=button_text),
    )
    return SimpleNamespace(update_id=456, effective_message=message, message=message)


@pytest.mark.asyncio
async def test_web_app_data_dispatches_as_text_event_with_pretty_json_payload():
    adapter = _make_adapter()
    update = _make_web_app_update('{"type":"budget_summary","spent":2677}')

    await adapter._handle_web_app_data_message(update, SimpleNamespace())

    adapter.handle_message.assert_called_once()
    event = adapter.handle_message.call_args.args[0]
    assert event.message_type is MessageType.TEXT
    assert event.source.chat_id == "6168627430"
    assert event.source.user_id == "6168627430"
    assert event.message_id == "123"
    assert event.platform_update_id == 456
    assert "[Telegram Mini App data]" in event.text
    assert "Button: Open Mini App" in event.text
    assert '"type": "budget_summary"' in event.text
    assert '"spent": 2677' in event.text


@pytest.mark.asyncio
async def test_web_app_data_uses_raw_payload_when_not_json():
    adapter = _make_adapter()
    update = _make_web_app_update("plain payload")

    await adapter._handle_web_app_data_message(update, SimpleNamespace())

    event = adapter.handle_message.call_args.args[0]
    assert "plain payload" in event.text
