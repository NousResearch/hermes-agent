"""Distinct slash-command messages remain repeatable (#29779)."""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import MessageType
from gateway.platforms.weixin import WeixinAdapter


@pytest.mark.parametrize("text", ["/approve", "  /approve", "/deny", "/sms", "/retry"])
def test_distinct_commands_dispatch_but_transport_replays_do_not(text):
    adapter = WeixinAdapter(PlatformConfig(enabled=True, extra={"account_id": "test-account"}))
    adapter._poll_session = object()
    adapter._token = ""  # No typing-ticket network request in this ingress fixture.
    adapter.handle_message = AsyncMock()
    message = {"from_user_id": "sender", "item_list": [{"type": 1, "text_item": {"text": text}}]}

    async def drive():
        for message_id in ("first", "second", "second", "third"):
            await adapter._process_message({**message, "message_id": message_id})

    asyncio.run(drive())
    events = [call.args[0] for call in adapter.handle_message.await_args_list]
    assert [event.message_id for event in events] == ["first", "second", "third"]
    assert all(event.message_type == MessageType.COMMAND for event in events)
    assert all(event.text == text for event in events)


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
