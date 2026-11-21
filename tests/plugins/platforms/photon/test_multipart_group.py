"""One multi-part iMessage ("intro text + photo + caption") reaches the adapter as ONE
``group`` event now that the sidecar runs Spectrum with ``flattenGroups: false``; the
adapter must fold it into a single agent turn carrying every text part and the media.
With flattening on, the parts arrived one by one and the text started a turn seconds
before its still-downloading photo, so the agent answered without the image."""
from __future__ import annotations

from typing import Any

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType
from plugins.platforms.photon.adapter import PhotonAdapter

_PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
_PHONE = "+15550004567"
_CHAT = f"any;-;{_PHONE}"
_GUID = "spc-msg-0000-multipart"


def _adapter(monkeypatch: pytest.MonkeyPatch) -> tuple[PhotonAdapter, list[MessageEvent]]:
    monkeypatch.setenv("PHOTON_PROJECT_ID", "test-project-id")
    monkeypatch.setenv("PHOTON_PROJECT_SECRET", "test-project-secret")
    adapter = PhotonAdapter(PlatformConfig(enabled=True, token="", extra={}))
    captured: list[MessageEvent] = []

    async def fake_handle(event: MessageEvent) -> None:
        captured.append(event)

    monkeypatch.setattr(adapter, "handle_message", fake_handle)
    return adapter, captured


def _part(index: int, content: dict[str, Any]) -> dict[str, Any]:
    return {"id": f"p:{index}/{_GUID}", "content": content}


def _group_event(*items: dict[str, Any]) -> dict[str, Any]:
    return {"messageId": _GUID, "space": {"id": _CHAT, "type": "dm", "phone": _PHONE},
            "sender": {"id": _PHONE}, "content": {"type": "group", "items": list(items)},
            "timestamp": "2026-11-04T10:16:43.000Z"}


_PHOTO = {"type": "attachment", "name": "IMG_1.png", "mimeType": "image/png", "data": _PNG_B64,
          "encoding": "base64"}


@pytest.mark.asyncio
async def test_text_photo_caption_imessage_is_one_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, captured = _adapter(monkeypatch)
    await adapter._dispatch_inbound(_group_event(
        _part(0, {"type": "text", "text": "Can I plant an orange tree with this:"}),
        _part(1, _PHOTO),
        _part(2, {"type": "text", "text": "Looks hard, right?"}),
    ))
    assert len(captured) == 1
    event = captured[0]
    assert event.text == "Can I plant an orange tree with this:\nLooks hard, right?"
    assert event.message_type == MessageType.PHOTO
    assert event.media_types == ["image/png"] and len(event.media_urls) == 1
    assert event.message_id == _GUID


@pytest.mark.asyncio
async def test_photo_then_caption_imessage_is_one_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, captured = _adapter(monkeypatch)
    await adapter._dispatch_inbound(_group_event(
        _part(0, _PHOTO), _part(1, {"type": "text", "text": "What game is this?"})))
    assert len(captured) == 1
    assert captured[0].text == "What game is this?"
    assert captured[0].message_type == MessageType.PHOTO and len(captured[0].media_urls) == 1
