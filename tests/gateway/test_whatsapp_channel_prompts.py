"""Tests for WhatsApp per-chat ephemeral system prompts (``channel_prompts``).

One WhatsApp connection can give different chats a different persona by keying
``channel_prompts`` on the chat JID (group ``@g.us`` / DM ``@s.whatsapp.net``),
matching the support already present in the telegram/slack/discord adapters.
The adapter resolves the prompt in ``_build_message_event`` via the shared
``gateway.platforms.base.resolve_channel_prompt`` helper and attaches it to
``MessageEvent.channel_prompt``; the agent then applies it as an ephemeral
system-prompt override for that turn. No match → ``None`` → no override.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from plugins.platforms.whatsapp.adapter import WhatsAppAdapter


@pytest.fixture(autouse=True)
def _whatsapp_open_optin(monkeypatch):
    monkeypatch.setenv("WHATSAPP_ALLOW_ALL_USERS", "true")


def _make_adapter(channel_prompts=None):
    adapter = WhatsAppAdapter.__new__(WhatsAppAdapter)
    adapter.platform = Platform.WHATSAPP
    adapter.config = PlatformConfig(
        enabled=True,
        extra={"channel_prompts": channel_prompts or {}},
    )
    adapter._message_handler = AsyncMock()
    adapter._dm_policy = "open"
    adapter._allow_from = set()
    adapter._group_policy = "open"
    adapter._group_allow_from = set()
    adapter._mention_patterns = []
    adapter._free_response_chats = set()
    adapter._whatsapp_free_response_chats = lambda: set()
    return adapter


_DM_JID = "6281234567890@s.whatsapp.net"
_GROUP_JID = "120363000000000000@g.us"


def _payload(chat_id, is_group=False, **overrides):
    payload = {
        "messageId": "M1",
        "chatId": chat_id,
        "senderId": _DM_JID,
        "senderName": "Customer",
        "chatName": "Customer",
        "isGroup": is_group,
        "body": "hello",
        "hasMedia": False,
        "mediaType": "",
        "mediaUrls": [],
        "mentionedIds": [],
        "quotedParticipant": "",
        "botIds": [],
        "timestamp": 0,
    }
    payload.update(overrides)
    return payload


def test_dm_channel_prompt_applied():
    adapter = _make_adapter({_DM_JID: "You are the after-sales concierge."})
    event = asyncio.run(adapter._build_message_event(_payload(_DM_JID)))
    assert event is not None
    assert event.channel_prompt == "You are the after-sales concierge."


def test_group_channel_prompt_applied():
    adapter = _make_adapter({_GROUP_JID: "You are the recruitment desk for this group."})
    event = asyncio.run(adapter._build_message_event(_payload(_GROUP_JID, is_group=True)))
    assert event is not None
    assert event.channel_prompt == "You are the recruitment desk for this group."


def test_unlisted_chat_gets_no_prompt():
    adapter = _make_adapter({_GROUP_JID: "group persona"})
    event = asyncio.run(adapter._build_message_event(_payload(_DM_JID)))
    assert event is not None
    assert event.channel_prompt is None


def test_empty_config_is_none():
    adapter = _make_adapter({})
    event = asyncio.run(adapter._build_message_event(_payload(_DM_JID)))
    assert event is not None
    assert event.channel_prompt is None


def test_blank_prompt_treated_as_absent():
    adapter = _make_adapter({_DM_JID: "   "})
    event = asyncio.run(adapter._build_message_event(_payload(_DM_JID)))
    assert event is not None
    assert event.channel_prompt is None
