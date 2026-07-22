"""Inbound dispatch + dedup tests for PhotonAdapter.

These bypass the loopback HTTP stream — they call ``_dispatch_inbound`` /
``_on_inbound_line`` / ``_is_duplicate`` directly, exercising the
sidecar-event parsing without spawning the Node sidecar or binding ports.
"""
from __future__ import annotations

import asyncio
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


# ---------------------------------------------------------------------------
# CAF attachment promotion + U+FFFC placeholder tests
# ---------------------------------------------------------------------------

_CAF_BYTES = b"caff" + b"\x00" * 60  # Minimal CAF header magic


def _caf_attachment_event(
    content: Dict[str, Any], msg_id: str = "spc-msg-caf"
) -> Dict[str, Any]:
    return {
        "messageId": msg_id,
        "space": {"id": "+155****4567", "type": "dm", "phone": "+155****4567"},
        "sender": {"id": "+155****4567"},
        "content": {"type": "attachment", **content},
        "timestamp": "2026-05-14T19:06:32.000Z",
    }


@pytest.mark.asyncio
async def test_caf_attachment_named_promoted_to_voice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A named .caf attachment is promoted to VOICE for STT routing."""
    adapter = _make_adapter(monkeypatch)
    captured = _capture(adapter, monkeypatch)

    raw = _CAF_BYTES
    event = _caf_attachment_event(
        {
            "name": "voice_note.caf",
            "mimeType": "audio/x-caf",
            "size": len(raw),
            "data": base64.b64encode(raw).decode("ascii"),
            "encoding": "base64",
        }
    )
    await adapter._dispatch_inbound(event)

    assert len(captured) == 1
    ev = captured[0]
    assert ev.message_type == MessageType.VOICE
    assert ev.media_types == ["audio/x-caf"]
    assert len(ev.media_urls) == 1
    cached = Path(ev.media_urls[0])
    try:
        assert cached.is_file()
        assert cached.read_bytes() == raw
        assert ev.text == "(voice)"
    finally:
        cached.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_caf_attachment_unnamed_promoted_via_mime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unnamed attachment with mimeType audio/x-caf is promoted to VOICE.

    The sidecar sends "(unnamed)" when no filename is supplied, so the MIME
    type must be the fallback signal for CAF promotion.
    """
    adapter = _make_adapter(monkeypatch)
    captured = _capture(adapter, monkeypatch)

    raw = _CAF_BYTES
    event = _caf_attachment_event(
        {
            "name": "(unnamed)",
            "mimeType": "audio/x-caf",
            "size": len(raw),
            "data": base64.b64encode(raw).decode("ascii"),
            "encoding": "base64",
        }
    )
    await adapter._dispatch_inbound(event)

    assert len(captured) == 1
    ev = captured[0]
    assert ev.message_type == MessageType.VOICE
    assert ev.media_types == ["audio/x-caf"]
    cached = Path(ev.media_urls[0])
    try:
        assert cached.is_file()
        assert cached.suffix == ".caf"
    finally:
        cached.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_fffc_placeholder_no_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A U+FFFC placeholder text does not trigger a message dispatch."""
    adapter = _make_adapter(monkeypatch)
    captured = _capture(adapter, monkeypatch)

    event = _dm_event("\ufffc", msg_id="spc-msg-fffc")
    chat_key = event["space"]["id"]
    await adapter._dispatch_inbound(event)

    assert len(captured) == 0
    assert chat_key in adapter._pending_fffc


@pytest.mark.asyncio
async def test_fffc_placeholder_not_recorded_as_last_inbound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The U+FFFC placeholder must not be recorded as the reaction target.

    _record_last_inbound runs after the U+FFFC early-return, so the placeholder
    message id is never stored. A subsequent real message will be recorded.
    """
    adapter = _make_adapter(monkeypatch)
    _capture(adapter, monkeypatch)

    fffc_event = _dm_event("\ufffc", msg_id="spc-msg-fffc")
    chat_key = fffc_event["space"]["id"]
    await adapter._dispatch_inbound(fffc_event)
    assert chat_key not in adapter._last_inbound_by_chat

    real_event = _dm_event("hello", msg_id="spc-msg-real")
    await adapter._dispatch_inbound(real_event)
    assert adapter._last_inbound_by_chat.get(chat_key) == "spc-msg-real"


@pytest.mark.asyncio
async def test_fffc_then_attachment_cancels_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When an attachment arrives after a U+FFFC placeholder, the pending
    timeout task is cancelled and the attachment is dispatched normally."""
    adapter = _make_adapter(monkeypatch)
    captured = _capture(adapter, monkeypatch)

    fffc_event = _dm_event("\ufffc", msg_id="spc-msg-fffc")
    chat_key = fffc_event["space"]["id"]
    await adapter._dispatch_inbound(fffc_event)
    assert len(captured) == 0
    assert chat_key in adapter._pending_fffc

    raw = _CAF_BYTES
    att_event = _caf_attachment_event(
        {
            "name": "voice.caf",
            "mimeType": "audio/x-caf",
            "size": len(raw),
            "data": base64.b64encode(raw).decode("ascii"),
            "encoding": "base64",
        },
        msg_id="spc-msg-att",
    )
    att_event["space"]["id"] = chat_key
    await adapter._dispatch_inbound(att_event)

    assert len(captured) == 1
    assert captured[0].message_type == MessageType.VOICE
    assert chat_key not in adapter._pending_fffc
    assert adapter._last_inbound_by_chat.get(chat_key) == "spc-msg-att"

    cached = Path(captured[0].media_urls[0])
    try:
        assert cached.is_file()
    finally:
        cached.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_fffc_timeout_fires_when_no_attachment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When no attachment arrives within the timeout, the pending entry is
    cleaned up and a warning is logged."""
    import plugins.platforms.photon.adapter as adapter_mod

    monkeypatch.setattr(adapter_mod, "_FFFC_WAIT_SECONDS", 0.1)

    adapter = _make_adapter(monkeypatch)
    captured = _capture(adapter, monkeypatch)

    fffc_event = _dm_event("\ufffc", msg_id="spc-msg-fffc")
    chat_key = fffc_event["space"]["id"]
    await adapter._dispatch_inbound(fffc_event)
    assert chat_key in adapter._pending_fffc

    await asyncio.sleep(0.3)

    assert chat_key not in adapter._pending_fffc
    assert len(captured) == 0


@pytest.mark.asyncio
async def test_disconnect_cancels_pending_fffc_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """disconnect() cancels any pending U+FFFC placeholder tasks."""
    adapter = _make_adapter(monkeypatch)
    _capture(adapter, monkeypatch)

    await adapter._dispatch_inbound(_dm_event("\ufffc", msg_id="spc-msg-fffc"))
    assert len(adapter._pending_fffc) == 1

    async def _noop_stop_sidecar():
        pass

    monkeypatch.setattr(adapter, "_stop_sidecar", _noop_stop_sidecar)
    monkeypatch.setattr(adapter, "_inbound_running", False)
    monkeypatch.setattr(adapter, "_inbound_task", None)
    monkeypatch.setattr(adapter, "_sidecar_health_task", None)
    monkeypatch.setattr(adapter, "_http_client", None)

    await adapter.disconnect()

    assert len(adapter._pending_fffc) == 0
