"""Inbound dispatch, dedup, cursor, and draft-streaming tests for RelayAdapter.

These bypass the network — they call ``_on_event`` / ``_render_text`` /
``send_draft`` / ``send`` directly with a stubbed HTTP client, exercising
event parsing, at-least-once dedup, cursor persistence, and the
draft → append → finalize streaming lifecycle without any Relay server.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from plugins.platforms.relay.adapter import RelayAdapter, _env_enablement


def _make_adapter(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> RelayAdapter:
    monkeypatch.setenv("RELAY_AGENT_TOKEN", "relay_agt_live_testtoken")
    monkeypatch.setenv("RELAY_CURSOR_PATH", str(tmp_path / "relay_cursor"))
    cfg = PlatformConfig(enabled=True, token="", extra={})
    adapter = RelayAdapter(cfg)
    adapter._agent_id = "agt_self"
    return adapter


def _capture(adapter: RelayAdapter, monkeypatch: pytest.MonkeyPatch) -> List[MessageEvent]:
    captured: List[MessageEvent] = []

    async def fake_handle(event: MessageEvent) -> None:
        captured.append(event)

    monkeypatch.setattr(adapter, "handle_message", fake_handle)
    return captured


def _message_event(
    text: str = "hello world",
    event_id: str = "evt_1",
    message_id: str = "msg_1",
    sender_id: str = "usr_1",
    parts: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    return {
        "event_id": event_id,
        "event_type": "message.received",
        "agent_id": "agt_self",
        "created_at": "2026-07-13T04:00:00.000Z",
        "data": {
            "message": {
                "id": message_id,
                "conversation_id": "cnv_1",
                "sequence": 1,
                "sender": {"kind": "user", "id": sender_id},
                "parts": parts if parts is not None else [{"part_index": 0, "type": "text", "text": text}],
                "reply_to": None,
                "fallback_text": text,
                "status": "sent",
                "created_at": "2026-07-13T04:00:00.000Z",
            }
        },
    }


class _StubResponse:
    def __init__(self, status_code: int = 202, payload: Optional[Dict[str, Any]] = None):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = str(self._payload)

    def json(self) -> Dict[str, Any]:
        return self._payload


class _StubHttpClient:
    """Records POSTs; returns canned responses keyed by URL suffix."""

    def __init__(self):
        self.posts: List[Dict[str, Any]] = []

    async def post(self, url: str, json: Optional[Dict[str, Any]] = None, headers=None, timeout=None):
        self.posts.append({"url": url, "json": json or {}})
        if url.endswith("/v1/messages") and (json or {}).get("draft"):
            return _StubResponse(202, {"message_id": "msg_draft_1", "draft": True})
        if url.endswith("/append"):
            return _StubResponse(202, {})
        if url.endswith("/finalize"):
            return _StubResponse(202, {"message_id": "msg_draft_1"})
        return _StubResponse(202, {"message_id": "msg_fresh_1"})


@pytest.mark.asyncio
async def test_dispatch_text_dm(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    adapter = _make_adapter(monkeypatch, tmp_path)
    captured = _capture(adapter, monkeypatch)

    await adapter._on_event(_message_event("hello world"))

    assert len(captured) == 1
    event = captured[0]
    assert event.text == "hello world"
    assert event.message_type == MessageType.TEXT
    assert event.message_id == "msg_1"
    src = event.source
    assert src is not None
    assert src.platform == Platform("relay")
    assert src.chat_id == "cnv_1"
    assert src.chat_type == "dm"
    assert src.user_id == "usr_1"


@pytest.mark.asyncio
async def test_duplicate_event_dispatched_once(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    adapter = _make_adapter(monkeypatch, tmp_path)
    captured = _capture(adapter, monkeypatch)

    await adapter._on_event(_message_event(event_id="evt_dup"))
    await adapter._on_event(_message_event(event_id="evt_dup"))

    assert len(captured) == 1


@pytest.mark.asyncio
async def test_own_messages_skipped(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    adapter = _make_adapter(monkeypatch, tmp_path)
    captured = _capture(adapter, monkeypatch)

    await adapter._on_event(_message_event(sender_id="agt_self", event_id="evt_echo"))

    assert captured == []


@pytest.mark.asyncio
async def test_non_message_events_ignored(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    adapter = _make_adapter(monkeypatch, tmp_path)
    captured = _capture(adapter, monkeypatch)

    event = _message_event(event_id="evt_other")
    event["event_type"] = "reaction.added"
    await adapter._on_event(event)

    assert captured == []


def test_render_text_joins_parts_in_order() -> None:
    message = {
        "parts": [
            {"part_index": 0, "type": "text", "text": "Here is the link:"},
            {"part_index": 1, "type": "link", "text": "https://example.com"},
        ],
        "fallback_text": "Here is the link:",
    }
    assert RelayAdapter._render_text(message) == "Here is the link:\nhttps://example.com"


def test_render_text_falls_back_for_attachment_only() -> None:
    message = {
        "parts": [{"part_index": 0, "type": "voice", "attachment_id": "att_1"}],
        "fallback_text": "Audio message",
    }
    assert RelayAdapter._render_text(message) == "Audio message"


def test_cursor_persists_across_instances(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    adapter = _make_adapter(monkeypatch, tmp_path)
    adapter._save_cursor("djE6MQ==")

    fresh = _make_adapter(monkeypatch, tmp_path)
    assert fresh._cursor == "djE6MQ=="


def test_env_enablement_seeds_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RELAY_AGENT_TOKEN", "relay_agt_live_testtoken")
    monkeypatch.setenv("RELAY_HOME_CHANNEL", "cnv_home")
    seed = _env_enablement()
    assert seed is not None
    assert seed["token"] == "relay_agt_live_testtoken"
    assert seed["home_channel"]["chat_id"] == "cnv_home"

    monkeypatch.delenv("RELAY_AGENT_TOKEN")
    assert _env_enablement() is None


@pytest.mark.asyncio
async def test_draft_frames_append_deltas_and_send_finalizes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    adapter = _make_adapter(monkeypatch, tmp_path)
    stub = _StubHttpClient()
    adapter._http_client = stub  # type: ignore[assignment]

    first = await adapter.send_draft("cnv_1", 7, "Hello")
    second = await adapter.send_draft("cnv_1", 7, "Hello world")
    assert first.success and second.success
    assert first.message_id == second.message_id == "msg_draft_1"

    appends = [p for p in stub.posts if p["url"].endswith("/append")]
    assert [p["json"]["text"] for p in appends] == ["Hello", " world"]

    final = await adapter.send("cnv_1", "Hello world!")
    assert final.success
    assert final.message_id == "msg_draft_1"

    finalizes = [p for p in stub.posts if p["url"].endswith("/finalize")]
    assert len(finalizes) == 1
    assert finalizes[0]["json"]["parts"] == [{"type": "text", "text": "Hello world!"}]

    # Draft is closed: the next send creates a fresh message.
    fresh = await adapter.send("cnv_1", "another")
    assert fresh.message_id == "msg_fresh_1"


@pytest.mark.asyncio
async def test_rewritten_draft_content_declined(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    adapter = _make_adapter(monkeypatch, tmp_path)
    stub = _StubHttpClient()
    adapter._http_client = stub  # type: ignore[assignment]

    assert (await adapter.send_draft("cnv_1", 9, "Hello world")).success
    # Consumer rewrote earlier text — append-only transport must decline
    # so the stream consumer falls back and send() finalizes.
    declined = await adapter.send_draft("cnv_1", 9, "Goodbye world")
    assert not declined.success
