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
from plugins.platforms.whatsapp import adapter as whatsapp_adapter
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


def test_gifplayback_mp4_keeps_truthful_local_container(monkeypatch, tmp_path):
    """Baileys gifPlayback media is an MP4, not a GIF-named image."""
    monkeypatch.setattr(whatsapp_adapter, "_is_allowed_bridge_path", lambda _url: True)
    path = str(tmp_path / "vid_abc123.mp4")
    adapter = _make_adapter()

    event = asyncio.run(
        adapter._build_message_event(
            _media_payload("gif", mime="video/mp4", mediaUrls=[path])
        )
    )

    assert event is not None
    assert event.message_type == MessageType.PHOTO
    assert event.media_urls == [path]
    assert event.media_types == ["video/mp4"]


def test_gifplayback_mp4_does_not_enter_image_cache(monkeypatch):
    """A remote MP4 must not be cached with a misleading .gif suffix."""
    cache_image = AsyncMock()
    monkeypatch.setattr(whatsapp_adapter, "cache_image_from_url", cache_image)
    url = "https://cdn.example.test/inbound-gif"
    adapter = _make_adapter()

    event = asyncio.run(
        adapter._build_message_event(
            _media_payload("gif", mime="video/mp4", mediaUrls=[url])
        )
    )

    assert event is not None
    assert event.message_type == MessageType.PHOTO
    assert event.media_urls == [url]
    assert event.media_types == ["video/mp4"]
    cache_image.assert_not_awaited()


def test_real_gif_url_uses_gif_image_cache(monkeypatch):
    """Only an actual image/gif payload gets the .gif image-cache path."""
    cache_image = AsyncMock(return_value="/cache/img_abc123.gif")
    monkeypatch.setattr(whatsapp_adapter, "cache_image_from_url", cache_image)
    adapter = _make_adapter()

    event = asyncio.run(
        adapter._build_message_event(
            _media_payload(
                "gif",
                mime="image/gif",
                mediaUrls=["https://cdn.example.test/inbound.gif"],
            )
        )
    )

    assert event is not None
    assert event.message_type == MessageType.PHOTO
    assert event.media_urls == ["/cache/img_abc123.gif"]
    assert event.media_types == ["image/gif"]
    cache_image.assert_awaited_once_with("https://cdn.example.test/inbound.gif", ext=".gif")


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
