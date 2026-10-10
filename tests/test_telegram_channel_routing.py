"""Offline channel-post integration with real PTB updates and the Telegram adapter.

Lives outside tests/gateway so its conftest cannot replace PTB with stubs.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gateway.config import PlatformConfig

Update = pytest.importorskip("telegram").Update


@pytest.mark.asyncio
@pytest.mark.parametrize("guest_mode", [False, True])
async def test_channel_post_handler_enforces_allowlist_and_mentions(guest_mode):
    from plugins.platforms.telegram.adapter import TelegramAdapter

    channel_id = -1003950368353
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***", extra={
        "allowed_chats": ["-100"], "allowed_topics": [],
        "group_allow_from": [str(channel_id)], "guest_mode": guest_mode,
        "require_mention": True, "observe_unmentioned_group_messages": False,
    }))
    adapter._bot = SimpleNamespace(id=999, username="hermes_bot")
    adapter._enqueue_text_event = MagicMock()
    channel = {"id": channel_id, "type": "channel", "title": "Broadcast"}
    post = {"message_id": 11, "date": 1710000000, "chat": channel, "sender_chat": channel,
            "text": "hi @hermes_bot", "entities": [{"type": "mention", "offset": 3, "length": 11}]}
    update = Update.de_json({"update_id": 12345, "channel_post": post}, bot=None)

    await adapter._handle_text_message(update, MagicMock())
    adapter._enqueue_text_event.assert_not_called()
    adapter.config.extra["allowed_chats"] = [str(channel_id)]
    await adapter._handle_text_message(update, MagicMock())
    adapter._enqueue_text_event.assert_called_once()
    event = adapter._enqueue_text_event.call_args.args[0]
    assert event.source.chat_id == str(channel_id)
    assert event.source.chat_type == "channel"
    assert event.platform_update_id == 12345

    adapter._enqueue_text_event.reset_mock()
    post = {**post, "text": "ordinary post", "entities": []}
    update = Update.de_json({"update_id": 12346, "channel_post": post}, bot=None)
    await adapter._handle_text_message(update, MagicMock())
    adapter._enqueue_text_event.assert_not_called()
