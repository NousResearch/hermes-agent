"""Distinct slash-command messages remain repeatable (#29779)."""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import MessageType
from gateway.platforms.weixin import WeixinAdapter


@pytest.mark.parametrize("text", ["/approve", "  /approve", "/deny", "/sms", "/retry"])
@pytest.mark.parametrize("with_media", [False, True])
def test_distinct_commands_dispatch_but_transport_replays_do_not(text, with_media):
    adapter = WeixinAdapter(PlatformConfig(enabled=True, extra={"account_id": "test-account"}))
    adapter._poll_session = object()
    adapter._token = ""  # No typing-ticket network request in this ingress fixture.
    events = []

    async def handler(event):
        events.append(event)

    adapter._message_handler = handler
    if with_media:
        adapter._download_media = AsyncMock(return_value=("fixture-image.png", "image/png"))
    source = adapter.build_source(chat_id="sender", chat_type="dm", user_id="sender", user_name="sender")
    adapter._active_sessions[adapter._source_session_key(source)] = object()
    items = [{"type": 1, "text_item": {"text": text}}]
    if with_media:
        items.append({"type": 2, "image_item": {}})
    message = {"from_user_id": "sender", "item_list": items}

    async def drive():
        for message_id in ("first", "second", "second", "third"):
            await adapter._process_message({**message, "message_id": message_id})

    # Unknown/custom commands need the normal idle path, rather than the busy
    # bypass reserved for control commands such as /approve and /deny.
    if text.strip() in {"/sms", "/retry"}:
        adapter.handle_message = handler
    asyncio.run(drive())
    assert [event.message_id for event in events] == ["first", "second", "third"]
    assert all(event.get_command() == text.strip()[1:] for event in events)
    assert all(event.message_type == (MessageType.PHOTO if with_media else MessageType.COMMAND) for event in events)
    assert all(event.text == text for event in events)
    assert not adapter._pending_messages


def test_plain_text_dedup_and_empty_ingress_are_unchanged():
    adapter = WeixinAdapter(PlatformConfig(enabled=True, extra={"account_id": "test-account"}))
    adapter._poll_session = object()
    adapter._token = ""
    adapter._enqueue_text_event = Mock()
    adapter.handle_message = AsyncMock()

    async def drive():
        for message_id, text in (("one", "hello"), ("two", "hello"), ("three", "")):
            await adapter._process_message({
                "from_user_id": "sender", "message_id": message_id,
                "item_list": [{"type": 1, "text_item": {"text": text}}],
            })

    asyncio.run(drive())
    adapter._enqueue_text_event.assert_called_once()
    assert adapter._enqueue_text_event.call_args.args[0].text == "hello"
    adapter.handle_message.assert_not_awaited()
