"""Inbound dispatch + dedup tests for PhotonAdapter.

These tests bypass the aiohttp server — they call ``_dispatch_inbound``
and ``_is_duplicate`` directly. That keeps them fast and means we can
exercise the message-shape parsing logic without binding ports.
"""
from __future__ import annotations

from typing import List

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from plugins.platforms.photon.adapter import PhotonAdapter


def _make_adapter(monkeypatch: pytest.MonkeyPatch) -> PhotonAdapter:
    # Avoid touching real auth.json / env.
    monkeypatch.setenv("PHOTON_PROJECT_ID", "test-project-id")
    monkeypatch.setenv("PHOTON_PROJECT_SECRET", "test-project-secret")
    monkeypatch.delenv("PHOTON_WEBHOOK_SECRET", raising=False)
    cfg = PlatformConfig(enabled=True, token="", extra={})
    return PhotonAdapter(cfg)


@pytest.mark.asyncio
async def test_dispatch_text_dm(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _make_adapter(monkeypatch)
    captured: List[MessageEvent] = []

    async def fake_handle(event: MessageEvent) -> None:
        captured.append(event)

    monkeypatch.setattr(adapter, "handle_message", fake_handle)

    payload = {
        "event": "messages",
        "space": {"id": "any;-;+15551234567", "platform": "iMessage"},
        "message": {
            "id": "spc-msg-abc",
            "platform": "iMessage",
            "direction": "inbound",
            "timestamp": "2026-05-14T19:06:32.000Z",
            "sender": {"id": "+15551234567", "platform": "iMessage"},
            "space": {"id": "any;-;+15551234567", "platform": "iMessage"},
            "content": {"type": "text", "text": "hello world"},
        },
    }
    await adapter._dispatch_inbound(payload)

    assert len(captured) == 1
    event = captured[0]
    assert event.text == "hello world"
    assert event.message_type == MessageType.TEXT
    assert event.message_id == "spc-msg-abc"
    src = event.source
    assert src is not None
    assert src.platform == Platform("photon")
    assert src.chat_id == "any;-;+15551234567"
    assert src.chat_type == "dm"
    assert src.user_id == "+15551234567"


@pytest.mark.asyncio
async def test_dispatch_group_id_detected(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _make_adapter(monkeypatch)
    captured: List[MessageEvent] = []

    async def fake_handle(event: MessageEvent) -> None:
        captured.append(event)

    monkeypatch.setattr(adapter, "handle_message", fake_handle)

    payload = {
        "event": "messages",
        "space": {"id": "any;+;group-guid-xyz", "platform": "iMessage"},
        "message": {
            "id": "spc-msg-grp",
            "timestamp": "2026-05-14T19:06:32.000Z",
            "sender": {"id": "+15551234567"},
            "space": {"id": "any;+;group-guid-xyz"},
            "content": {"type": "text", "text": "hi group"},
        },
    }
    await adapter._dispatch_inbound(payload)
    assert captured[0].source.chat_type == "group"


@pytest.mark.asyncio
async def test_dispatch_attachment_surfaces_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter(monkeypatch)
    captured: List[MessageEvent] = []

    async def fake_handle(event: MessageEvent) -> None:
        captured.append(event)

    monkeypatch.setattr(adapter, "handle_message", fake_handle)

    payload = {
        "event": "messages",
        "message": {
            "id": "spc-msg-att",
            "timestamp": "2026-05-14T19:06:32.000Z",
            "sender": {"id": "+15551234567"},
            "space": {"id": "any;-;+15551234567"},
            "content": {
                "type": "attachment",
                "name": "IMG_4127.HEIC",
                "mimeType": "image/heic",
                "size": 12345,
            },
        },
    }
    await adapter._dispatch_inbound(payload)
    assert len(captured) == 1
    event = captured[0]
    # Attachment carries metadata marker; mime → MessageType.PHOTO.
    assert "Photon attachment received" in event.text
    assert "IMG_4127.HEIC" in event.text
    assert event.message_type == MessageType.PHOTO


@pytest.mark.asyncio
async def test_dispatch_attachment_with_local_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """When the sidecar has downloaded the bytes and forwarded ``localPath``,
    the adapter surfaces the real file via ``media_urls`` (vision-tool access)
    instead of the metadata-only marker."""
    adapter = _make_adapter(monkeypatch)
    captured: List[MessageEvent] = []

    async def fake_handle(event: MessageEvent) -> None:
        captured.append(event)

    monkeypatch.setattr(adapter, "handle_message", fake_handle)

    img = tmp_path / "IMG_4127.jpg"
    img.write_bytes(b"\xff\xd8\xff\xe0 jpeg bytes")

    payload = {
        "event": "messages",
        "message": {
            "id": "spc-msg-att-dl",
            "timestamp": "2026-05-14T19:06:32.000Z",
            "sender": {"id": "+15551234567"},
            "space": {"id": "any;-;+15551234567"},
            "content": {
                "type": "attachment",
                "name": "IMG_4127.jpg",
                "mimeType": "image/jpeg",
                "size": 12,
                "localPath": str(img),
            },
        },
    }
    await adapter._dispatch_inbound(payload)
    assert len(captured) == 1
    event = captured[0]
    assert event.media_urls == [str(img)]
    assert event.media_types == ["image/jpeg"]
    assert event.message_type == MessageType.PHOTO
    # No metadata marker once the real file is available.
    assert "Photon attachment received" not in event.text


@pytest.mark.asyncio
async def test_dispatch_attachment_missing_local_path_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-existent ``localPath`` (download failed in the sidecar) falls back
    to the metadata marker rather than handing the agent a dead path."""
    adapter = _make_adapter(monkeypatch)
    captured: List[MessageEvent] = []

    async def fake_handle(event: MessageEvent) -> None:
        captured.append(event)

    monkeypatch.setattr(adapter, "handle_message", fake_handle)

    payload = {
        "event": "messages",
        "message": {
            "id": "spc-msg-att-fail",
            "timestamp": "2026-05-14T19:06:32.000Z",
            "sender": {"id": "+15551234567"},
            "space": {"id": "any;-;+15551234567"},
            "content": {
                "type": "attachment",
                "name": "IMG_4127.jpg",
                "mimeType": "image/jpeg",
                "size": 12,
                "localPath": "/tmp/does-not-exist-xyz.jpg",
            },
        },
    }
    await adapter._dispatch_inbound(payload)
    event = captured[0]
    assert event.media_urls == []
    assert "Photon attachment received" in event.text


@pytest.mark.asyncio
async def test_dispatch_reaction(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tapback / emoji reactions arrive on the same stream and are surfaced."""
    adapter = _make_adapter(monkeypatch)
    captured: List[MessageEvent] = []

    async def fake_handle(event: MessageEvent) -> None:
        captured.append(event)

    monkeypatch.setattr(adapter, "handle_message", fake_handle)

    payload = {
        "event": "messages",
        "message": {
            "id": "spc-msg-react",
            "timestamp": "2026-05-14T19:06:32.000Z",
            "sender": {"id": "+15551234567"},
            "space": {"id": "any;-;+15551234567"},
            "content": {
                "type": "reaction",
                "emoji": "❤️",
                "target": "spc-msg-abc",
            },
        },
    }
    await adapter._dispatch_inbound(payload)
    event = captured[0]
    assert "❤️" in event.text
    assert event.message_type == MessageType.TEXT


def test_is_duplicate_window(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _make_adapter(monkeypatch)
    assert adapter._is_duplicate("id-1") is False
    assert adapter._is_duplicate("id-1") is True
    assert adapter._is_duplicate("id-2") is False
    assert adapter._is_duplicate("id-1") is True  # still dup


def test_check_requirements_without_node(monkeypatch: pytest.MonkeyPatch) -> None:
    # If no node binary on PATH the adapter should refuse to start.
    from plugins.platforms.photon import adapter as adapter_mod

    monkeypatch.setattr(adapter_mod.shutil, "which", lambda _name: None)
    assert adapter_mod.check_requirements() is False
