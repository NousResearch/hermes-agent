"""Inbound dispatch + dedup tests for PhotonAdapter.

These bypass the loopback HTTP stream — they call ``_dispatch_inbound`` /
``_on_inbound_line`` / ``_is_duplicate`` directly, exercising the
sidecar-event parsing without spawning the Node sidecar or binding ports.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from plugins.platforms.photon.adapter import PhotonAdapter


def _make_adapter(monkeypatch: pytest.MonkeyPatch) -> PhotonAdapter:
    monkeypatch.setenv("PHOTON_PROJECT_ID", "test-project-id")
    monkeypatch.setenv("PHOTON_PROJECT_SECRET", "test-project-secret")
    cfg = PlatformConfig(enabled=True, token="", extra={})
    return PhotonAdapter(cfg)


def _capture(adapter: PhotonAdapter, monkeypatch: pytest.MonkeyPatch) -> List[MessageEvent]:
    captured: List[MessageEvent] = []

    async def fake_handle(event: MessageEvent) -> None:
        captured.append(event)

    monkeypatch.setattr(adapter, "handle_message", fake_handle)
    return captured


def _dm_event(text: str, msg_id: str = "spc-msg-abc") -> Dict[str, Any]:
    return {
        "messageId": msg_id,
        "platform": "iMessage",
        "space": {"id": "+15551234567", "type": "dm", "phone": "+15551234567"},
        "sender": {"id": "+15551234567"},
        "content": {"type": "text", "text": text},
        "timestamp": "2026-05-14T19:06:32.000Z",
    }


@pytest.mark.asyncio
async def test_dispatch_text_dm(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _make_adapter(monkeypatch)
    captured = _capture(adapter, monkeypatch)

    await adapter._dispatch_inbound(_dm_event("hello world"))

    assert len(captured) == 1
    event = captured[0]
    assert event.text == "hello world"
    assert event.message_type == MessageType.TEXT
    assert event.message_id == "spc-msg-abc"
    src = event.source
    assert src is not None
    assert src.platform == Platform("photon")
    assert src.chat_id == "+15551234567"
    assert src.chat_type == "dm"
    assert src.user_id == "+15551234567"


@pytest.mark.asyncio
async def test_dispatch_group_type(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _make_adapter(monkeypatch)
    captured = _capture(adapter, monkeypatch)

    event = {
        "messageId": "spc-msg-grp",
        "space": {"id": "group-guid-xyz", "type": "group", "phone": None},
        "sender": {"id": "+15551234567"},
        "content": {"type": "text", "text": "hi group"},
        "timestamp": "2026-05-14T19:06:32.000Z",
    }
    await adapter._dispatch_inbound(event)
    assert captured[0].source.chat_type == "group"


# A real 1x1 transparent PNG (passes base.py's _looks_like_image magic check).
_PNG_1X1_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYPhf"
    "DwAChwGA60e6kgAAAABJRU5ErkJggg=="
)
_HEIC_STUB = b"\x00\x00\x00\x18ftypheic" + b"\x00" * 32
_HEIC_STUB_B64 = base64.b64encode(_HEIC_STUB).decode("ascii")
_JPEG_STUB = b"\xff\xd8\xff\xe0" + b"converted-jpeg" + b"\xff\xd9"


def _attachment_event(
    content: Dict[str, Any], msg_id: str = "spc-msg-att"
) -> Dict[str, Any]:
    return {
        "messageId": msg_id,
        "space": {"id": "+15551234567", "type": "dm", "phone": "+15551234567"},
        "sender": {"id": "+15551234567"},
        "content": {"type": "attachment", **content},
        "timestamp": "2026-05-14T19:06:32.000Z",
    }


def _voice_event(
    content: Dict[str, Any], msg_id: str = "spc-msg-voice"
) -> Dict[str, Any]:
    return {
        "messageId": msg_id,
        "space": {"id": "+15551234567", "type": "dm", "phone": "+15551234567"},
        "sender": {"id": "+15551234567"},
        "content": {"type": "voice", **content},
        "timestamp": "2026-05-14T19:06:32.000Z",
    }


@pytest.mark.asyncio
async def test_dispatch_attachment_without_bytes_surfaces_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No inline ``data`` (over cap / failed sidecar read) -> text marker, no media."""
    adapter = _make_adapter(monkeypatch)
    captured = _capture(adapter, monkeypatch)

    event = _attachment_event(
        {"name": "IMG_4127.HEIC", "mimeType": "image/heic", "size": 12345}
    )
    await adapter._dispatch_inbound(event)
    assert len(captured) == 1
    ev = captured[0]
    assert "Photon attachment received" in ev.text
    assert "IMG_4127.HEIC" in ev.text
    assert ev.message_type == MessageType.PHOTO
    assert ev.media_urls == []
    assert ev.media_types == []


@pytest.mark.asyncio
async def test_dispatch_attachment_downloads_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inline base64 image bytes are decoded, cached, and exposed as media."""
    adapter = _make_adapter(monkeypatch)
    captured = _capture(adapter, monkeypatch)

    raw = base64.b64decode(_PNG_1X1_B64)
    event = _attachment_event(
        {
            "name": "photo.png",
            "mimeType": "image/png",
            "size": len(raw),
            "data": _PNG_1X1_B64,
            "encoding": "base64",
        }
    )
    await adapter._dispatch_inbound(event)

    assert len(captured) == 1
    ev = captured[0]
    assert ev.message_type == MessageType.PHOTO
    assert ev.media_types == ["image/png"]
    assert len(ev.media_urls) == 1
    cached = Path(ev.media_urls[0])
    try:
        assert cached.is_file()
        assert cached.read_bytes() == raw
        assert ev.text == "(attachment)"
    finally:
        cached.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_dispatch_heic_attachment_transcodes_to_supported_jpeg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import gateway.platforms.base as base

    monkeypatch.setattr(
        base,
        "_transcode_heic_to_jpeg",
        lambda data: _JPEG_STUB if data == _HEIC_STUB else None,
    )
    adapter = _make_adapter(monkeypatch)
    captured = _capture(adapter, monkeypatch)

    event = _attachment_event(
        {
            "name": "IMG_4127.HEIC",
            "mimeType": "image/heic",
            "size": len(_HEIC_STUB),
            "data": _HEIC_STUB_B64,
            "encoding": "base64",
        }
    )
    await adapter._dispatch_inbound(event)

    assert len(captured) == 1
    ev = captured[0]
    assert ev.message_type == MessageType.PHOTO
    assert ev.media_types == ["image/jpeg"]
    assert len(ev.media_urls) == 1
    cached = Path(ev.media_urls[0])
    try:
        assert cached.is_file()
        assert cached.suffix == ".jpg"
        assert cached.read_bytes() == _JPEG_STUB
        assert ev.text == "(attachment)"
    finally:
        cached.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_dispatch_group_preserves_text_and_attachment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Spectrum group content from a mixed text+image iMessage must not drop text."""
    adapter = _make_adapter(monkeypatch)
    captured = _capture(adapter, monkeypatch)
    raw = base64.b64decode(_PNG_1X1_B64)

    event = _attachment_event(
        {},
        msg_id="spc-msg-mixed",
    )
    event["content"] = {
        "type": "group",
        "items": [
            {
                "id": "p:0/spc-msg-mixed",
                "content": {"type": "text", "text": "请分析这张图的重点"},
            },
            {
                "id": "p:1/spc-msg-mixed",
                "content": {
                    "type": "attachment",
                    "name": "photo.png",
                    "mimeType": "image/png",
                    "size": len(raw),
                    "data": _PNG_1X1_B64,
                    "encoding": "base64",
                },
            },
        ],
    }

    await adapter._dispatch_inbound(event)

    assert len(captured) == 1
    ev = captured[0]
    assert ev.text == "请分析这张图的重点"
    assert ev.message_type == MessageType.PHOTO
    assert ev.media_types == ["image/png"]
    assert len(ev.media_urls) == 1
    cached = Path(ev.media_urls[0])
    try:
        assert cached.is_file()
        assert cached.read_bytes() == raw
    finally:
        cached.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_dispatch_group_heic_without_converter_uses_document_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mixed text+unconverted HEIC keeps text and avoids invalid image media."""
    import gateway.platforms.base as base

    monkeypatch.setattr(base, "_transcode_heic_to_jpeg", lambda data: None)
    adapter = _make_adapter(monkeypatch)
    captured = _capture(adapter, monkeypatch)

    event = _attachment_event({}, msg_id="spc-msg-mixed-heic-doc")
    event["content"] = {
        "type": "group",
        "items": [
            {
                "id": "p:0/spc-msg-mixed-heic-doc",
                "content": {"type": "text", "text": "can you check this one?"},
            },
            {
                "id": "p:1/spc-msg-mixed-heic-doc",
                "content": {
                    "type": "attachment",
                    "name": "IMG_4127.HEIC",
                    "mimeType": "image/heic",
                    "size": len(_HEIC_STUB),
                    "data": _HEIC_STUB_B64,
                    "encoding": "base64",
                },
            },
        ],
    }

    await adapter._dispatch_inbound(event)

    assert len(captured) == 1
    ev = captured[0]
    assert ev.text == "can you check this one?"
    assert ev.message_type == MessageType.DOCUMENT
    assert ev.media_types == ["application/octet-stream"]
    assert len(ev.media_urls) == 1
    cached = Path(ev.media_urls[0])
    try:
        assert cached.is_file()
        assert cached.read_bytes() == _HEIC_STUB
    finally:
        cached.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_dispatch_voice_downloads_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inbound Spectrum voice content is cached and routed to auto-STT."""
    adapter = _make_adapter(monkeypatch)
    captured = _capture(adapter, monkeypatch)

    raw = b"OggS" + b"\x00" * 32
    event = _voice_event(
        {
            "name": "note.ogg",
            "mimeType": "audio/ogg",
            "duration": 7,
            "size": len(raw),
            "data": base64.b64encode(raw).decode("ascii"),
            "encoding": "base64",
        }
    )
    await adapter._dispatch_inbound(event)

    assert len(captured) == 1
    ev = captured[0]
    assert ev.message_type == MessageType.VOICE
    assert ev.media_types == ["audio/ogg"]
    assert len(ev.media_urls) == 1
    cached = Path(ev.media_urls[0])
    try:
        assert cached.is_file()
        assert cached.read_bytes() == raw
        assert ev.text == "(voice)"
    finally:
        cached.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_dispatch_voice_without_bytes_surfaces_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Metadata-only voice still tells the agent a voice note arrived."""
    adapter = _make_adapter(monkeypatch)
    captured = _capture(adapter, monkeypatch)

    event = _voice_event(
        {"name": "note.m4a", "mimeType": "audio/mp4", "duration": 12, "size": 12345}
    )
    await adapter._dispatch_inbound(event)

    assert len(captured) == 1
    ev = captured[0]
    assert "Photon voice received" in ev.text
    assert "note.m4a" in ev.text
    assert "duration: 12s" in ev.text
    assert ev.message_type == MessageType.VOICE
    assert ev.media_urls == []
    assert ev.media_types == []


@pytest.mark.asyncio
async def test_dispatch_attachment_downloads_document(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-image attachments route through the document cache as DOCUMENT."""
    adapter = _make_adapter(monkeypatch)
    captured = _capture(adapter, monkeypatch)

    raw = b"%PDF-1.4 hermes test document"
    event = _attachment_event(
        {
            "name": "report.pdf",
            "mimeType": "application/pdf",
            "size": len(raw),
            "data": base64.b64encode(raw).decode("ascii"),
            "encoding": "base64",
        }
    )
    await adapter._dispatch_inbound(event)

    assert len(captured) == 1
    ev = captured[0]
    assert ev.message_type == MessageType.DOCUMENT
    assert ev.media_types == ["application/pdf"]
    assert len(ev.media_urls) == 1
    cached = Path(ev.media_urls[0])
    try:
        assert cached.is_file()
        assert cached.read_bytes() == raw
        assert ev.text == "(attachment)"
    finally:
        cached.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_on_inbound_line_dispatches_and_dedups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter(monkeypatch)
    captured = _capture(adapter, monkeypatch)

    line = json.dumps(_dm_event("ping", msg_id="dup-1"))
    await adapter._on_inbound_line(line)
    await adapter._on_inbound_line(line)  # same messageId -> deduped

    assert len(captured) == 1
    assert captured[0].text == "ping"


@pytest.mark.asyncio
async def test_on_inbound_line_ignores_bad_json(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _make_adapter(monkeypatch)
    captured = _capture(adapter, monkeypatch)

    await adapter._on_inbound_line("{not json")
    assert captured == []


def test_is_duplicate_window(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _make_adapter(monkeypatch)
    assert adapter._is_duplicate("id-1") is False
    assert adapter._is_duplicate("id-1") is True
    assert adapter._is_duplicate("id-2") is False
    assert adapter._is_duplicate("id-1") is True  # still dup


def test_is_duplicate_hard_size_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    # A burst of unique ids within the window must not grow the dedup map past
    # its bound — evict oldest (LRU), not only expired entries.
    import plugins.platforms.photon.adapter as ad

    monkeypatch.setattr(ad, "_DEDUP_MAX_SIZE", 5)
    adapter = _make_adapter(monkeypatch)
    for i in range(100):
        adapter._is_duplicate(f"id-{i}")
    assert len(adapter._seen_messages) <= 5
    assert adapter._is_duplicate("id-99") is True  # recent still deduped
    assert adapter._is_duplicate("id-0") is False  # oldest evicted


def test_check_requirements_without_node(monkeypatch: pytest.MonkeyPatch) -> None:
    # If no node binary on PATH the adapter should refuse to start.
    from plugins.platforms.photon import adapter as adapter_mod

    monkeypatch.setattr(adapter_mod.shutil, "which", lambda _name: None)
    assert adapter_mod.check_requirements() is False
