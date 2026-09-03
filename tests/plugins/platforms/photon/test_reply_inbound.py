"""Threaded-reply (iMessage reply) tests for PhotonAdapter.

iMessage replies arrive wrapped as spectrum {content, target}; the sidecar
flattens them into {type:"reply", text, reply_to_message_id, reply_to_text,
reply_to_direction} so the user's actual reply text survives (previously it
fell through to "[Photon content type not handled: reply]" and the message
was lost). These tests feed flattened reply events straight to
``_dispatch_inbound`` and assert the resulting MessageEvent carries the text
plus reply correlation context — same pattern as test_reactions.py.
"""
from __future__ import annotations

from typing import Any, Dict, List

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from plugins.platforms.photon.adapter import PhotonAdapter

_SPACE = "+155****4567"


def _make_adapter(monkeypatch: pytest.MonkeyPatch) -> PhotonAdapter:
    monkeypatch.setenv("PHOTON_PROJECT_ID", "test-project-id")
    monkeypatch.setenv("PHOTON_PROJECT_SECRET", "test-project-secret")
    cfg = PlatformConfig(enabled=True, token="", extra={})
    return PhotonAdapter(cfg)


def _capture_handled(
    adapter: PhotonAdapter, monkeypatch: pytest.MonkeyPatch
) -> List[MessageEvent]:
    captured: List[MessageEvent] = []

    async def fake_handle(event: MessageEvent) -> None:
        captured.append(event)

    monkeypatch.setattr(adapter, "handle_message", fake_handle)
    return captured


def _reply_event(
    text: str = "the user's reply",
    target_id: str = "bot-msg-1",
    target_direction: Any = "outbound",
    target_text: Any = "the bot's earlier reply",
) -> Dict[str, Any]:
    """Shape the sidecar emits after flattening spectrum's reply content."""
    return {
        "messageId": "reply-evt-1",
        "platform": "iMessage",
        "space": {"id": _SPACE, "type": "dm", "phone": _SPACE},
        "sender": {"id": _SPACE},
        "content": {
            "type": "reply",
            "text": text,
            "reply_to_message_id": "bot-msg-1",
            "reply_to_text": target_text,
            "reply_to_direction": target_direction,
            # Normalized inner content (media etc.) for future handling.
            "content": {"type": "text", "text": text},
        },
        "timestamp": "2026-08-26T10:00:00.000Z",
    }


@pytest.mark.asyncio
async def test_threaded_reply_routed_with_text_and_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter(monkeypatch)
    captured = _capture_handled(adapter, monkeypatch)

    await adapter._dispatch_inbound(_reply_event())

    assert len(captured) == 1
    event = captured[0]
    # The user's actual reply text must survive (the bug: it was dropped as
    # "content type not handled").
    assert event.text == "the user's reply"
    assert event.message_type == MessageType.TEXT
    assert event.source.chat_id == _SPACE
    # Reply context so the gateway can inject [Replying to: "..."].
    assert event.reply_to_message_id == "bot-msg-1"
    assert event.reply_to_text == "the bot's earlier reply"
    # The target was one of the bot's own messages.
    assert event.reply_to_is_own_message is True


@pytest.mark.asyncio
async def test_reply_to_inbound_message_not_own(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter(monkeypatch)
    captured = _capture_handled(adapter, monkeypatch)

    # User replies to their OWN earlier message (direction inbound).
    await adapter._dispatch_inbound(
        _reply_event(
            target_direction="inbound",
            target_text="the user's own earlier message",
        )
    )

    assert len(captured) == 1
    assert captured[0].reply_to_is_own_message is False
    assert captured[0].reply_to_text == "the user's own earlier message"


@pytest.mark.asyncio
async def test_reply_to_attachment_keeps_event_alive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter(monkeypatch)
    captured = _capture_handled(adapter, monkeypatch)

    # Replying to an attachment/voice-only target yields no inner text.
    await adapter._dispatch_inbound(_reply_event(text="", target_text=None))

    assert len(captured) == 1
    # The user's intent is preserved with a marker instead of being dropped.
    assert captured[0].text == "(empty reply)"
    assert captured[0].reply_to_message_id == "bot-msg-1"