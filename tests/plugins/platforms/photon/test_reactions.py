"""Reaction (tapback) tests for PhotonAdapter.

Outbound reactions go through the sidecar's ``/react`` / ``/unreact``
endpoints; these tests stub ``_sidecar_call`` to assert endpoint + body
shape. Inbound reaction events are fed straight to ``_dispatch_inbound``.
Neither path spawns the Node sidecar or binds ports.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType, ProcessingOutcome
from plugins.platforms.photon.adapter import PhotonAdapter

_EYES = "\U0001f440"
_THUMBS_UP = "\U0001f44d"
_THUMBS_DOWN = "\U0001f44e"


def _make_adapter(monkeypatch: pytest.MonkeyPatch) -> PhotonAdapter:
    monkeypatch.setenv("PHOTON_PROJECT_ID", "test-project-id")
    monkeypatch.setenv("PHOTON_PROJECT_SECRET", "test-project-secret")
    cfg = PlatformConfig(enabled=True, token="", extra={})
    return PhotonAdapter(cfg)


def _capture_sidecar(adapter: PhotonAdapter) -> List[Tuple[str, Dict[str, Any]]]:
    calls: List[Tuple[str, Dict[str, Any]]] = []

    async def _fake_call(path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        calls.append((path, body))
        return {"ok": True, "messageId": "msg-123", "reactionId": "react-1"}

    adapter._sidecar_call = _fake_call  # type: ignore[assignment]
    return calls


def _capture_handled(
    adapter: PhotonAdapter, monkeypatch: pytest.MonkeyPatch
) -> List[MessageEvent]:
    captured: List[MessageEvent] = []

    async def fake_handle(event: MessageEvent) -> None:
        captured.append(event)

    monkeypatch.setattr(adapter, "handle_message", fake_handle)
    return captured


def _message_event(adapter: PhotonAdapter) -> MessageEvent:
    return MessageEvent(
        text="hi",
        message_type=MessageType.TEXT,
        source=adapter.build_source(
            chat_id="+15551234567",
            chat_name="+15551234567",
            chat_type="dm",
            user_id="+15551234567",
            user_name=None,
        ),
        message_id="target-msg-1",
        timestamp=datetime.now(tz=timezone.utc),
    )


def _reaction_event(
    emoji: str = "❤️",
    target_id: str = "bot-msg-1",
    target_direction: Any = "outbound",
    space_type: str = "dm",
) -> Dict[str, Any]:
    return {
        "messageId": "reaction-evt-1",
        "platform": "iMessage",
        "space": {"id": "+15551234567", "type": space_type, "phone": "+15551234567"},
        "sender": {"id": "+15551234567"},
        "content": {
            "type": "reaction",
            "emoji": emoji,
            "targetMessageId": target_id,
            "targetDirection": target_direction,
        },
        "timestamp": "2026-06-11T10:00:00.000Z",
    }


# -- Outbound: /react and /unreact body shapes ------------------------------

@pytest.mark.asyncio
async def test_add_reaction_posts_react(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _make_adapter(monkeypatch)
    calls = _capture_sidecar(adapter)

    ok = await adapter._add_reaction("+15551234567", "target-msg-1", _EYES)

    assert ok is True
    assert calls == [
        (
            "/react",
            {
                "spaceId": "+15551234567",
                "messageId": "target-msg-1",
                "emoji": _EYES,
            },
        )
    ]


@pytest.mark.asyncio
async def test_remove_reaction_posts_unreact(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _make_adapter(monkeypatch)
    calls = _capture_sidecar(adapter)

    ok = await adapter._remove_reaction("+15551234567", "target-msg-1")

    assert ok is True
    assert calls == [
        ("/unreact", {"spaceId": "+15551234567", "messageId": "target-msg-1"})
    ]


@pytest.mark.asyncio
async def test_reaction_failure_is_soft(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _make_adapter(monkeypatch)

    async def _boom(path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        raise RuntimeError("sidecar down")

    adapter._sidecar_call = _boom  # type: ignore[assignment]

    assert await adapter._add_reaction("+1", "m", _EYES) is False
    assert await adapter._remove_reaction("+1", "m") is False


# -- Lifecycle hooks ---------------------------------------------------------

@pytest.mark.asyncio
async def test_hooks_noop_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PHOTON_REACTIONS", raising=False)
    adapter = _make_adapter(monkeypatch)
    calls = _capture_sidecar(adapter)

    event = _message_event(adapter)
    await adapter.on_processing_start(event)
    await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)

    assert calls == []


@pytest.mark.asyncio
async def test_processing_start_adds_eyes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PHOTON_REACTIONS", "true")
    adapter = _make_adapter(monkeypatch)
    calls = _capture_sidecar(adapter)

    await adapter.on_processing_start(_message_event(adapter))

    assert len(calls) == 1
    path, body = calls[0]
    assert path == "/react"
    assert body["emoji"] == _EYES
    assert body["messageId"] == "target-msg-1"


@pytest.mark.asyncio
async def test_processing_success_swaps_to_thumbs_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PHOTON_REACTIONS", "true")
    adapter = _make_adapter(monkeypatch)
    calls = _capture_sidecar(adapter)

    await adapter.on_processing_complete(
        _message_event(adapter), ProcessingOutcome.SUCCESS
    )

    assert [path for path, _ in calls] == ["/unreact", "/react"]
    assert calls[1][1]["emoji"] == _THUMBS_UP


@pytest.mark.asyncio
async def test_processing_failure_swaps_to_thumbs_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PHOTON_REACTIONS", "true")
    adapter = _make_adapter(monkeypatch)
    calls = _capture_sidecar(adapter)

    await adapter.on_processing_complete(
        _message_event(adapter), ProcessingOutcome.FAILURE
    )

    assert [path for path, _ in calls] == ["/unreact", "/react"]
    assert calls[1][1]["emoji"] == _THUMBS_DOWN


@pytest.mark.asyncio
async def test_processing_cancelled_only_removes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PHOTON_REACTIONS", "true")
    adapter = _make_adapter(monkeypatch)
    calls = _capture_sidecar(adapter)

    await adapter.on_processing_complete(
        _message_event(adapter), ProcessingOutcome.CANCELLED
    )

    assert [path for path, _ in calls] == ["/unreact"]


# -- Inbound reaction routing ------------------------------------------------

@pytest.mark.asyncio
async def test_inbound_reaction_on_bot_message_routed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter(monkeypatch)
    captured = _capture_handled(adapter, monkeypatch)

    await adapter._dispatch_inbound(_reaction_event(emoji="❤️"))

    assert len(captured) == 1
    event = captured[0]
    assert event.text == "reaction:added:❤️"
    assert event.message_type == MessageType.TEXT
    assert event.source.chat_id == "+15551234567"


@pytest.mark.asyncio
async def test_inbound_reaction_sent_ids_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No targetDirection from the provider — gate on our own sent-id cache."""
    adapter = _make_adapter(monkeypatch)
    captured = _capture_handled(adapter, monkeypatch)
    adapter._record_sent_message("bot-msg-1")

    await adapter._dispatch_inbound(
        _reaction_event(target_id="bot-msg-1", target_direction=None)
    )

    assert len(captured) == 1


@pytest.mark.asyncio
async def test_inbound_reaction_on_foreign_message_dropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter(monkeypatch)
    captured = _capture_handled(adapter, monkeypatch)

    await adapter._dispatch_inbound(
        _reaction_event(target_id="someone-elses-msg", target_direction=None)
    )

    assert captured == []


@pytest.mark.asyncio
async def test_inbound_reaction_bypasses_require_mention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tapback never carries a wake word — it must skip group gating."""
    monkeypatch.setenv("PHOTON_REQUIRE_MENTION", "true")
    adapter = _make_adapter(monkeypatch)
    captured = _capture_handled(adapter, monkeypatch)

    await adapter._dispatch_inbound(_reaction_event(space_type="group"))

    assert len(captured) == 1
