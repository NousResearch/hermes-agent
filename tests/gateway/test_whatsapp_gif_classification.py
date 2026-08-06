"""Tests for WhatsApp inbound animated-GIF classification.

The Baileys bridge reports animated GIFs with ``mediaType: 'gif'`` and
``gifPlayback: true``.  The adapter must classify these as photos so the
agent's vision path attaches them as images, not as unknown-mime documents.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageType
from plugins.platforms.whatsapp.adapter import WhatsAppAdapter


@pytest.fixture(autouse=True)
def _whatsapp_open_optin(monkeypatch):
    """Opt into WhatsApp allow-all so ``dm_policy: open`` tests run."""
    monkeypatch.setenv("WHATSAPP_ALLOW_ALL_USERS", "true")


def _make_adapter():
    adapter = WhatsAppAdapter.__new__(WhatsAppAdapter)
    adapter.platform = Platform.WHATSAPP
    adapter.config = PlatformConfig(enabled=True)
    adapter._message_handler = AsyncMock()
    adapter._dm_policy = "open"
    adapter._allow_from = set()
    adapter._group_policy = "open"
    adapter._group_allow_from = set()
    adapter._mention_patterns = []
    adapter._free_response_chats = set()
    adapter._whatsapp_free_response_chats = lambda: set()
    return adapter


def _media_payload(media_type, **overrides):
    payload = {
        "messageId": "M1",
        "chatId": "6281234567890@s.whatsapp.net",
        "senderId": "6281234567890@s.whatsapp.net",
        "senderName": "Customer",
        "chatName": "Customer",
        "isGroup": False,
        "body": "",
        "hasMedia": True,
        "mediaType": media_type,
        "mediaUrls": [],
        "mentionedIds": [],
        "quotedParticipant": "",
        "botIds": [],
        "timestamp": 0,
    }
    payload.update(overrides)
    return payload


def test_animated_gif_classifies_as_photo():
    """Inbound animated GIFs (mediaType 'gif') must be MessageType.PHOTO."""
    adapter = _make_adapter()

    event = asyncio.run(adapter._build_message_event(_media_payload("gif")))

    assert event is not None
    assert event.message_type == MessageType.PHOTO


def test_real_video_still_classifies_as_video():
    """Videos with mediaType 'video' must remain MessageType.VIDEO."""
    adapter = _make_adapter()

    event = asyncio.run(adapter._build_message_event(_media_payload("video")))

    assert event is not None
    assert event.message_type == MessageType.VIDEO


def test_real_image_still_classifies_as_photo():
    """Static images with mediaType 'image' must remain MessageType.PHOTO."""
    adapter = _make_adapter()

    event = asyncio.run(adapter._build_message_event(_media_payload("image")))

    assert event is not None
    assert event.message_type == MessageType.PHOTO
