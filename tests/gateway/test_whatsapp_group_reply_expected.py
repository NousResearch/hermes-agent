"""WhatsApp group chatter read without a mention may go unanswered (reply_expected=False).

With ``require_mention: false`` (or a free-response chat) every group message reaches the agent.
Like Slack and Telegram, a message nobody addressed to the bot lets a bare silence marker stand;
anything addressed to it, and every DM, keeps the default (None: the visible fallback).
"""
import pytest

from tests.gateway.test_whatsapp_formatting import _make_adapter

BOT = "15551230000@s.whatsapp.net"


def _group(body="dinner at 7?", **overrides):
    data = {
        "messageId": "m1",
        "chatId": "120363001234567890@g.us",
        "senderId": "15557654321@s.whatsapp.net",
        "senderName": "Sam",
        "chatName": "Family",
        "isGroup": True,
        "body": body,
        "hasMedia": False,
        "mediaUrls": [],
        "mentionedIds": [],
        "botIds": [BOT],
        "quotedParticipant": "",
    }
    data.update(overrides)
    return data


def _adapter():
    adapter = _make_adapter()
    adapter.config.extra = {"require_mention": False}
    adapter._dm_policy = "pairing"  # admits a DM at intake; pairing itself runs later
    return adapter


@pytest.mark.asyncio
async def test_unaddressed_group_message_may_go_unanswered():
    event = await _adapter()._build_message_event(_group())
    assert event is not None
    assert event.reply_expected is False


@pytest.mark.asyncio
@pytest.mark.parametrize("overrides", [
    {"mentionedIds": [BOT], "body": "@15551230000 when is soccer?"},
    {"hasQuotedMessage": True, "quotedMessageId": "out1", "quotedParticipant": BOT, "body": "thanks"},
    {"body": "/status"},
])
async def test_group_message_addressed_to_the_bot_keeps_the_default(overrides):
    event = await _adapter()._build_message_event(_group(**overrides))
    assert event is not None
    assert event.reply_expected is None


@pytest.mark.asyncio
async def test_dm_keeps_the_default():
    data = _group(isGroup=False, chatId="15557654321@s.whatsapp.net", chatName="Sam")
    event = await _adapter()._build_message_event(data)
    assert event is not None
    assert event.reply_expected is None
