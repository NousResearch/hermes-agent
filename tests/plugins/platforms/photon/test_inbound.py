"""Inbound dispatch + dedup tests for PhotonAdapter.

These bypass the loopback HTTP stream — they call ``_dispatch_inbound`` /
``_on_inbound_line`` / ``_is_duplicate`` directly, exercising the
sidecar-event parsing without spawning the Node sidecar or binding ports.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType, ProcessingOutcome
from plugins.platforms.photon.adapter import (
    PhotonAdapter,
    PhotonAttachmentConsumerUnavailable,
)


def _make_adapter(monkeypatch: pytest.MonkeyPatch) -> PhotonAdapter:
    monkeypatch.setenv("PHOTON_PROJECT_ID", "test-project-id")
    monkeypatch.setenv("PHOTON_PROJECT_SECRET", "test-project-secret")
    cfg = PlatformConfig(enabled=True, token="", extra={})
    return PhotonAdapter(cfg)


def _capture(
    adapter: PhotonAdapter, monkeypatch: pytest.MonkeyPatch
) -> List[MessageEvent]:
    captured: List[MessageEvent] = []

    async def fake_handle(event: MessageEvent) -> None:
        captured.append(event)
        if isinstance(event.raw_message, dict) and event.raw_message.get("deliveryId"):
            await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)

    monkeypatch.setattr(adapter, "handle_message", fake_handle)
    return captured


def _accept_secure_handles(adapter: PhotonAdapter) -> None:
    adapter.set_attachment_sender_authorizer(lambda _source, _event: True)
    adapter.set_attachment_handle_consumer(lambda _event: True)


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

    event = _attachment_event({
        "name": "IMG_4127.HEIC",
        "mimeType": "image/heic",
        "size": 12345,
    })
    await adapter._dispatch_inbound(event)
    assert len(captured) == 1
    ev = captured[0]
    assert "Photon attachment received" in ev.text
    assert "IMG_4127.HEIC" in ev.text
    assert ev.message_type == MessageType.PHOTO
    assert ev.media_urls == []
    assert ev.media_types == []


@pytest.mark.asyncio
async def test_dispatch_attachment_preserves_secure_handle_without_plaintext_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The opaque handle stays in raw_message and never becomes a disk path."""
    adapter = _make_adapter(monkeypatch)
    _accept_secure_handles(adapter)
    captured = _capture(adapter, monkeypatch)

    handle = "a" * 48
    event = _attachment_event({
        "name": "photo.png",
        "mimeType": "image/png",
        "size": 67,
        "handle": handle,
    })
    await adapter._dispatch_inbound(event)

    assert len(captured) == 1
    ev = captured[0]
    assert ev.message_type == MessageType.PHOTO
    assert ev.media_types == []
    assert ev.media_urls == []
    assert "Photon attachment received" in ev.text
    assert ev.raw_message["content"]["handle"] == handle


@pytest.mark.asyncio
async def test_dispatch_ignores_legacy_inline_base64(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale or compromised sidecar cannot revive plaintext disk caching."""
    adapter = _make_adapter(monkeypatch)
    captured = _capture(adapter, monkeypatch)
    event = _attachment_event({
        "name": "legacy.txt",
        "mimeType": "text/plain",
        "size": 6,
        "data": "c2VjcmV0",
        "encoding": "base64",
    })

    await adapter._dispatch_inbound(event)

    assert captured[0].media_urls == []
    assert captured[0].media_types == []
    assert "Photon attachment received" in captured[0].text


@pytest.mark.asyncio
async def test_handle_without_secure_consumer_fails_retryably_instead_of_dispatching(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter(monkeypatch)
    captured = _capture(adapter, monkeypatch)
    notified: list[bool] = []

    async def notify() -> None:
        notified.append(True)

    monkeypatch.setattr(adapter, "_notify_fatal_error", notify)
    event = _attachment_event(
        {
            "name": "photo.png",
            "mimeType": "image/png",
            "size": 67,
            "handle": "e" * 48,
        },
        msg_id="secure-handle-no-consumer",
    )
    event["deliveryId"] = "9" * 48

    with pytest.raises(PhotonAttachmentConsumerUnavailable):
        await adapter._on_inbound_line(json.dumps(event))

    assert captured == []
    assert "secure-handle-no-consumer" not in adapter._seen_messages
    assert adapter.fatal_error_code == "ATTACHMENT_CONSUMER_UNAVAILABLE"
    assert adapter.fatal_error_retryable is True
    assert "e" * 48 not in str(adapter.fatal_error_message)
    assert notified == [True]


@pytest.mark.asyncio
async def test_registered_secure_consumer_receives_raw_handle_before_generic_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter(monkeypatch)
    captured = _capture(adapter, monkeypatch)
    consumed: list[str] = []

    async def consume(event: Dict[str, Any]) -> bool:
        consumed.append(event["content"]["handle"])
        return True

    adapter.set_attachment_sender_authorizer(lambda _source, _event: True)
    adapter.set_attachment_handle_consumer(consume)
    event = _attachment_event(
        {
            "name": "photo.png",
            "mimeType": "image/png",
            "size": 67,
            "handle": "f" * 48,
        },
        msg_id="secure-handle-consumed",
    )
    event["deliveryId"] = "8" * 48

    async def ack(_path: str, _body: Dict[str, Any]) -> Dict[str, Any]:
        return {"ok": True}

    monkeypatch.setattr(adapter, "_sidecar_call", ack)

    await adapter._on_inbound_line(json.dumps(event))

    assert consumed == ["f" * 48]
    assert len(captured) == 1
    assert captured[0].media_urls == []
    assert captured[0].raw_message["content"]["handle"] == "f" * 48


@pytest.mark.asyncio
async def test_group_attachment_without_mention_never_crosses_secure_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter(monkeypatch)
    adapter.require_mention = True
    captured = _capture(adapter, monkeypatch)
    calls: list[str] = []
    adapter.set_attachment_sender_authorizer(
        lambda _source, _event: calls.append("authorize") or True
    )
    adapter.set_attachment_handle_consumer(
        lambda _event: calls.append("consume") or True
    )
    event = _attachment_event({"handle": "1" * 48}, "ignored-group-handle")
    event["space"] = {"id": "group-1", "type": "group", "phone": None}

    await adapter._dispatch_inbound(event)

    assert calls == []
    assert captured == []


@pytest.mark.asyncio
async def test_ignored_group_attachment_is_acked_without_secure_redemption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter(monkeypatch)
    adapter.require_mention = True
    calls: list[str] = []
    adapter.set_attachment_sender_authorizer(
        lambda _source, _event: calls.append("authorize") or True
    )
    adapter.set_attachment_handle_consumer(
        lambda _event: calls.append("consume") or True
    )

    async def ack(path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        calls.append(f"ack:{path}:{body['deliveryId']}")
        return {"ok": True}

    monkeypatch.setattr(adapter, "_sidecar_call", ack)
    event = _attachment_event({"handle": "a" * 48}, "ignored-group-ack")
    event["space"] = {"id": "group-2", "type": "group", "phone": None}
    event["deliveryId"] = "b" * 48

    await adapter._on_inbound_line(json.dumps(event))

    assert calls == [f"ack:/inbound-ack:{'b' * 48}"]


@pytest.mark.asyncio
async def test_attachment_authorizes_then_consumes_then_dispatches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter(monkeypatch)
    calls: list[str] = []

    async def dispatch(_event: MessageEvent) -> None:
        calls.append("dispatch")
        await adapter.on_processing_complete(_event, ProcessingOutcome.SUCCESS)

    monkeypatch.setattr(adapter, "handle_message", dispatch)
    adapter.set_attachment_sender_authorizer(
        lambda _source, _event: calls.append("authorize") or True
    )
    adapter.set_attachment_handle_consumer(
        lambda _event: calls.append("consume") or True
    )

    await adapter._dispatch_inbound(_attachment_event({"handle": "2" * 48}))

    assert calls == ["authorize", "consume", "dispatch"]


@pytest.mark.asyncio
async def test_unauthorized_attachment_never_reaches_consumer_or_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter(monkeypatch)
    calls: list[str] = []
    monkeypatch.setattr(
        adapter,
        "handle_message",
        lambda _event: calls.append("dispatch"),
    )
    adapter.set_attachment_sender_authorizer(
        lambda _source, _event: calls.append("authorize") or False
    )
    adapter.set_attachment_handle_consumer(
        lambda _event: calls.append("consume") or True
    )

    with pytest.raises(PhotonAttachmentConsumerUnavailable):
        await adapter._dispatch_inbound(_attachment_event({"handle": "3" * 48}))

    assert calls == ["authorize"]


@pytest.mark.asyncio
async def test_delivery_ack_follows_secure_acceptance_and_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter(monkeypatch)
    calls: list[str] = []
    adapter.set_attachment_sender_authorizer(
        lambda _source, _event: calls.append("authorize") or True
    )
    adapter.set_attachment_handle_consumer(
        lambda _event: calls.append("consume") or True
    )

    async def dispatch(_event: MessageEvent) -> None:
        calls.append("dispatch")
        await adapter.on_processing_complete(_event, ProcessingOutcome.SUCCESS)

    async def sidecar_call(path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        calls.append(f"ack:{path}:{body['deliveryId']}")
        return {"ok": True}

    monkeypatch.setattr(adapter, "handle_message", dispatch)
    monkeypatch.setattr(adapter, "_sidecar_call", sidecar_call)
    event = _attachment_event({"handle": "4" * 48}, "acked-handle")
    event["deliveryId"] = "5" * 48

    await adapter._on_inbound_line(json.dumps(event))

    assert calls == ["authorize", "consume", "dispatch", f"ack:/inbound-ack:{'5' * 48}"]


@pytest.mark.asyncio
async def test_ack_failure_replays_without_double_consuming(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter(monkeypatch)
    calls: list[str] = []
    adapter.set_attachment_sender_authorizer(
        lambda _source, _event: calls.append("authorize") or True
    )
    adapter.set_attachment_handle_consumer(
        lambda _event: calls.append("consume") or True
    )

    async def dispatch(_event: MessageEvent) -> None:
        calls.append("dispatch")
        await adapter.on_processing_complete(_event, ProcessingOutcome.SUCCESS)

    monkeypatch.setattr(adapter, "handle_message", dispatch)
    attempts = 0

    async def sidecar_call(_path: str, _body: Dict[str, Any]) -> Dict[str, Any]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("connection lost")
        return {"ok": True}

    monkeypatch.setattr(adapter, "_sidecar_call", sidecar_call)
    event = _attachment_event({"handle": "6" * 48}, "replay-handle")
    event["deliveryId"] = "7" * 48
    line = json.dumps(event)

    with pytest.raises(RuntimeError, match="connection lost"):
        await adapter._on_inbound_line(line)
    await adapter._on_inbound_line(line)

    assert calls == ["authorize", "consume", "dispatch"]
    assert attempts == 2


@pytest.mark.asyncio
async def test_failed_background_processing_is_not_acked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter(monkeypatch)
    _accept_secure_handles(adapter)
    sidecar_calls: list[str] = []

    async def dispatch(event: MessageEvent) -> None:
        await adapter.on_processing_complete(event, ProcessingOutcome.FAILURE)

    async def sidecar_call(path: str, _body: Dict[str, Any]) -> Dict[str, Any]:
        sidecar_calls.append(path)
        return {"ok": True}

    monkeypatch.setattr(adapter, "handle_message", dispatch)
    monkeypatch.setattr(adapter, "_sidecar_call", sidecar_call)
    event = _attachment_event({"handle": "c" * 48}, "failed-background")
    event["deliveryId"] = "e" * 48

    with pytest.raises(RuntimeError, match="secure_attachment_dispatch_failed"):
        await adapter._on_inbound_line(json.dumps(event))

    assert "/inbound-ack" not in sidecar_calls
    assert "failed-background" not in adapter._seen_messages


@pytest.mark.asyncio
async def test_dispatch_group_preserves_text_and_attachment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Spectrum group content from a mixed text+image iMessage must not drop text."""
    adapter = _make_adapter(monkeypatch)
    _accept_secure_handles(adapter)
    captured = _capture(adapter, monkeypatch)

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
                    "size": 67,
                    "handle": "b" * 48,
                },
            },
        ],
    }

    await adapter._dispatch_inbound(event)

    assert len(captured) == 1
    ev = captured[0]
    assert ev.text.startswith("请分析这张图的重点")
    assert "Photon attachment received" in ev.text
    assert ev.message_type == MessageType.PHOTO
    assert ev.media_types == []
    assert ev.media_urls == []


@pytest.mark.asyncio
async def test_dispatch_voice_preserves_handle_without_plaintext_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inbound voice bytes remain behind the one-shot sidecar handle."""
    adapter = _make_adapter(monkeypatch)
    _accept_secure_handles(adapter)
    captured = _capture(adapter, monkeypatch)

    event = _voice_event({
        "name": "note.ogg",
        "mimeType": "audio/ogg",
        "duration": 7,
        "size": 36,
        "handle": "c" * 48,
    })
    await adapter._dispatch_inbound(event)

    assert len(captured) == 1
    ev = captured[0]
    assert ev.message_type == MessageType.VOICE
    assert ev.media_types == []
    assert ev.media_urls == []
    assert "Photon voice received" in ev.text
    assert ev.raw_message["content"]["handle"] == "c" * 48


@pytest.mark.asyncio
async def test_dispatch_voice_without_bytes_surfaces_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Metadata-only voice still tells the agent a voice note arrived."""
    adapter = _make_adapter(monkeypatch)
    captured = _capture(adapter, monkeypatch)

    event = _voice_event({
        "name": "note.m4a",
        "mimeType": "audio/mp4",
        "duration": 12,
        "size": 12345,
    })
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
async def test_dispatch_attachment_document_stays_behind_handle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-image attachments retain type without writing a document cache."""
    adapter = _make_adapter(monkeypatch)
    _accept_secure_handles(adapter)
    captured = _capture(adapter, monkeypatch)

    event = _attachment_event({
        "name": "report.pdf",
        "mimeType": "application/pdf",
        "size": 29,
        "handle": "d" * 48,
    })
    await adapter._dispatch_inbound(event)

    assert len(captured) == 1
    ev = captured[0]
    assert ev.message_type == MessageType.DOCUMENT
    assert ev.media_types == []
    assert ev.media_urls == []
    assert "Photon attachment received" in ev.text


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
async def test_on_inbound_line_ignores_bad_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    _accept_secure_handles(adapter)
    captured = _capture(adapter, monkeypatch)

    event = _caf_attachment_event(
        {
            "name": "voice_note.caf",
            "mimeType": "audio/x-caf",
            "size": len(_CAF_BYTES),
            "handle": "e" * 48,
        }
    )
    await adapter._dispatch_inbound(event)

    assert len(captured) == 1
    ev = captured[0]
    assert ev.message_type == MessageType.VOICE
    assert ev.media_types == []
    assert ev.media_urls == []
    assert "Photon voice received" in ev.text
    assert ev.raw_message["content"]["handle"] == "e" * 48


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
