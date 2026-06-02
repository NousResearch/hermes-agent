"""Inbound SDK-stream tests for the Photon adapter."""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from gateway.config import PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from plugins.platforms.photon import adapter as photon_adapter


def _make_adapter(monkeypatch: Any, tmp_path: Path) -> photon_adapter.PhotonAdapter:
    monkeypatch.delenv("PHOTON_PROJECT_ID", raising=False)
    monkeypatch.delenv("PHOTON_PROJECT_SECRET", raising=False)
    monkeypatch.delenv("PHOTON_OPERATOR_PHONE", raising=False)
    monkeypatch.setattr(photon_adapter, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(
        photon_adapter,
        "load_project_credentials",
        lambda: ("project-id", "project-secret"),
    )
    return photon_adapter.PhotonAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "project_name": "hermes-agent",
                "operator_phone": "+15550001000",
            },
        )
    )


def _sdk_text_event(message_id: str = "msg-1") -> dict[str, Any]:
    return {
        "id": message_id,
        "timestamp": "2026-05-30T12:00:00Z",
        "space": {"id": "any;-;+15550001000", "name": "Operator"},
        "sender": {"id": "+15550001000", "name": "Operator"},
        "content": {"type": "text", "text": "hello Hermes"},
    }


def test_sidecar_event_dispatches_message_event(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    adapter = _make_adapter(monkeypatch, tmp_path)
    events: list[MessageEvent] = []

    async def capture(event: MessageEvent) -> None:
        events.append(event)

    monkeypatch.setattr(adapter, "handle_message", capture)

    asyncio.run(
        adapter._handle_sidecar_message(
            {"type": "event", "event": _sdk_text_event("msg-1")}
        )
    )

    assert len(events) == 1
    event = events[0]
    assert event.text == "hello Hermes"
    assert event.message_type is MessageType.TEXT
    assert event.message_id == "msg-1"
    assert event.source.platform.value == "photon"
    assert event.source.chat_id == "any;-;+15550001000"
    assert event.source.chat_type == "dm"
    assert event.source.user_id == "+15550001000"


def test_sidecar_event_dedupes_repeated_message_id(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    adapter = _make_adapter(monkeypatch, tmp_path)
    events: list[MessageEvent] = []

    async def capture(event: MessageEvent) -> None:
        events.append(event)

    monkeypatch.setattr(adapter, "handle_message", capture)
    sidecar_message = {"type": "event", "event": _sdk_text_event("msg-duplicate")}

    asyncio.run(adapter._handle_sidecar_message(sidecar_message))
    asyncio.run(adapter._handle_sidecar_message(sidecar_message))

    assert [event.message_id for event in events] == ["msg-duplicate"]


def test_sidecar_event_attachment_becomes_metadata_marker(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    adapter = _make_adapter(monkeypatch, tmp_path)
    events: list[MessageEvent] = []

    async def capture(event: MessageEvent) -> None:
        events.append(event)

    monkeypatch.setattr(adapter, "handle_message", capture)

    asyncio.run(
        adapter._handle_sidecar_message(
            {
                "type": "event",
                "event": {
                    "id": "msg-attachment",
                    "space": {"id": "any;-;+15550001000"},
                    "sender": {"id": "+15550001000"},
                    "content": {
                        "type": "attachment",
                        "name": "photo.heic",
                        "mimeType": "image/heic",
                    },
                },
            }
        )
    )

    assert len(events) == 1
    assert "Photon attachment received" in events[0].text
    assert "photo.heic" in events[0].text
    assert events[0].message_type is MessageType.PHOTO


def test_malformed_sidecar_event_is_ignored(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    adapter = _make_adapter(monkeypatch, tmp_path)
    events: list[MessageEvent] = []

    async def capture(event: MessageEvent) -> None:
        events.append(event)

    monkeypatch.setattr(adapter, "handle_message", capture)

    asyncio.run(
        adapter._handle_sidecar_message(
            {
                "type": "event",
                "event": {
                    "id": "missing-space",
                    "sender": {"id": "+15550001000"},
                    "content": {"type": "text", "text": "ignored"},
                },
            }
        )
    )

    assert events == []


def test_sidecar_event_updates_runtime_state(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    adapter = _make_adapter(monkeypatch, tmp_path)

    async def capture(_event: MessageEvent) -> None:
        return None

    monkeypatch.setattr(adapter, "handle_message", capture)

    asyncio.run(
        adapter._handle_sidecar_message(
            {"type": "event", "event": _sdk_text_event("msg-state")}
        )
    )

    state = photon_adapter.read_adapter_runtime_state()
    assert state["health"]["last_event_at"]
    assert state["health"]["project_id"] == "project-id"
