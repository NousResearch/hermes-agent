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


# ---------------------------------------------------------------------------
# Empty-text + poison-line guards (#55105)
#
# A blank text envelope (the sidecar normalizes a missing body to "") used to
# dispatch as a real user turn: it woke — or post-restart, resumed — the
# chat's session with a blank prompt and burned a full agent turn for nothing.
# While that phantom turn spun, every further inbound interrupted-and-requeued
# behind it (the reported "stuck after restart / typing dots forever"). Same
# bug class as Signal's contentless-envelope skip.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_text_event_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty/whitespace text events dispatch nothing and are not recorded as
    the chat's reactable last-inbound."""
    adapter = _make_adapter(monkeypatch)
    captured = _capture(adapter, monkeypatch)

    await adapter._dispatch_inbound(_dm_event("", msg_id="spc-msg-empty"))
    await adapter._dispatch_inbound(_dm_event("   ", msg_id="spc-msg-blank"))

    assert captured == []
    assert adapter._last_inbound_by_chat == {}


@pytest.mark.asyncio
async def test_resume_proceeds_past_previously_in_flight_empty_item(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After a restart the fresh adapter's loop must get PAST the item that
    was in flight when the old process died: blank items are consumed without
    producing a turn, and the next real message dispatches normally."""
    adapter = _make_adapter(monkeypatch)
    captured = _capture(adapter, monkeypatch)

    # The previously-in-flight blank item (redelivered or replayed).
    await adapter._on_inbound_line(json.dumps(_dm_event("", msg_id="stuck-1")))
    await adapter._on_inbound_line(json.dumps(_dm_event("   ", msg_id="stuck-2")))
    # Life goes on.
    await adapter._on_inbound_line(json.dumps(_dm_event("ahoj", msg_id="real-1")))

    assert [e.text for e in captured] == ["ahoj"]
    assert captured[0].message_id == "real-1"


@pytest.mark.asyncio
async def test_poison_line_skipped_without_requeue_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A valid-JSON non-object line is poison-consumed, not raised: it must
    not tear down the inbound stream (reconnect-churn forever on a repeated
    bad line). Subsequent good lines proceed."""
    adapter = _make_adapter(monkeypatch)
    captured = _capture(adapter, monkeypatch)

    await adapter._on_inbound_line("null")
    await adapter._on_inbound_line('["not","an","object"]')
    await adapter._on_inbound_line('"bare string"')
    # A repeat of the same poison line must also be harmless (no requeue).
    await adapter._on_inbound_line("null")
    await adapter._on_inbound_line(json.dumps(_dm_event("still here")))

    assert [e.text for e in captured] == ["still here"]


@pytest.mark.asyncio
async def test_fffc_placeholder_still_dispatches_after_empty_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The empty-text guard must not swallow the U+FFFC placeholder path:
    \\ufffc is not whitespace and still arms the attachment wait."""
    adapter = _make_adapter(monkeypatch)
    captured = _capture(adapter, monkeypatch)

    event = _dm_event("", msg_id="spc-msg-empty")
    chat_key = event["space"]["id"]
    await adapter._dispatch_inbound(event)
    fffc = dict(event, messageId="spc-msg-fffc")
    fffc["content"] = {"type": "text", "text": "\ufffc"}
    await adapter._dispatch_inbound(fffc)

    assert captured == []
    assert chat_key in adapter._pending_fffc
