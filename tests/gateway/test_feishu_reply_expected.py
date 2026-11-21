"""Free-response Feishu groups may stay silent without swallowing addressed requests."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.event import MessageType
from plugins.platforms.feishu.adapter import FeishuAdapter
from tests.gateway.feishu_helpers import make_message, make_sender
from tests.gateway.test_gateway_silence_tokens import _runner


def _adapter():
    adapter = FeishuAdapter(PlatformConfig(extra={
        "default_group_policy": "open", "require_mention": False, "group_sessions_per_user": False,
    }))
    adapter._bot_open_id = "ou_bot"
    adapter._bot_name = "Hermes"
    adapter.get_chat_info = AsyncMock(return_value={"name": "Test Chat"})
    adapter._resolve_sender_profile = AsyncMock(return_value={
        "user_id": "ou_human", "user_name": "Alice", "user_id_alt": None,
    })
    adapter._fetch_message_text = AsyncMock(return_value="earlier message")
    adapter._download_feishu_message_resources = AsyncMock(
        side_effect=lambda *, normalized, **_: (["/cache/photo.png"], ["image/png"])
        if normalized.raw_type == "image" else ([], [])
    )
    adapter._schedule_text_batch_flush = Mock()
    adapter._schedule_media_batch_flush = Mock()
    adapter.handle_message = AsyncMock()
    return adapter


async def _receive(adapter, *, text="side chatter", chat_type="group", mention=None,
                   message_id="om_message", message_type="text", reply=False, thread=False):
    mentions = [] if mention is None else [SimpleNamespace(
        key="@_user_1", id=SimpleNamespace(open_id=mention, user_id=None),
        name="Hermes" if mention == "ou_bot" else "Bob",
    )]
    message = make_message(message_id=message_id, chat_type=chat_type, mentions=mentions)
    message.message_type = message_type
    message.content = json.dumps({"text": text} if message_type == "text" else {"image_key": message_id})
    message.parent_id = "om_parent" if reply else None
    message.thread_id = "omt_topic" if thread else None
    sender = make_sender()
    assert adapter._admit(sender, message) is None
    await adapter._process_inbound_message(
        data=message, message=message, sender_id=sender.sender_id,
        chat_type=chat_type, message_id=message_id,
    )


async def _flush(adapter):
    for key in list(adapter._pending_text_batches):
        await adapter._flush_text_batch_now(key)
    for key in list(adapter._pending_media_batches):
        await adapter._flush_media_batch_now(key)
    adapter.handle_message.assert_awaited_once()
    return adapter.handle_message.await_args.args[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("incoming, expected", [
    ({}, False),
    ({"mention": "ou_peer", "text": "@_user_1 can you check?"}, False),
    ({"chat_type": "p2p"}, True),
    ({"mention": "ou_bot", "text": "@_user_1 can you check?"}, True),
    ({"text": "@_all can you check?"}, True),
    ({"text": "/ask can you check?"}, True),
    ({"reply": True}, None),
    ({"thread": True}, None),
], ids=["group-chatter", "peer-mention", "dm", "bot-mention", "all-mention", "command", "quote", "thread"])
async def test_inbound_silence_delivery(monkeypatch, tmp_path, incoming, expected):
    adapter = _adapter()
    await _receive(adapter, **incoming)
    event = await _flush(adapter)
    runner = _runner(monkeypatch, tmp_path)
    runner._session_db.get_model_override.return_value = None
    runner._run_agent = AsyncMock(return_value={"final_response": "NO_REPLY", "messages": [], "failed": False})
    key = adapter._event_session_key(event)

    response = await runner._handle_message_with_agent(event, event.source, key, 1)

    assert event.reply_expected is expected
    if expected is False:
        assert response == ""
    else:
        assert "silence marker" in response


@pytest.mark.asyncio
@pytest.mark.parametrize("message_type", ["text", "image"])
@pytest.mark.parametrize("addressed_first", [False, True])
async def test_batch_keeps_an_addressed_request(message_type, addressed_first):
    adapter = _adapter()
    for index, addressed in enumerate([addressed_first, not addressed_first]):
        await _receive(
            adapter, text="@_user_1 please check" if addressed else "more context",
            mention="ou_bot" if addressed else None,
            message_id=f"om_{index}", message_type=message_type,
        )
    event = await _flush(adapter)

    assert event.reply_expected is True
    assert event.message_id == "om_1"
    if message_type == "image":
        assert event.message_type == MessageType.PHOTO
        assert len(event.media_urls) == 2
    else:
        assert "please check" in event.text and "more context" in event.text
