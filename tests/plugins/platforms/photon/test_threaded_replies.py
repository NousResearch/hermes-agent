"""iMessage threaded (swipe) replies for PhotonAdapter (#100663, #105320).

Inbound:

spectrum-ts 12.x wraps a threaded reply as ``{type: "reply", content, target}``. The
sidecar normalises it to ``{type: "reply", content, targetMessageId, targetDirection,
targetText}``; the adapter must unwrap it instead of emitting
"[Photon content type not handled: reply]", and carry the quoted message as reply
context. spectrum usually can't hydrate the text of our own outbound bubbles, so the
adapter records what it sends in ``gateway.rich_sent_store`` and falls back to that.

Outbound: the answer to a threaded message goes back into that thread via ``replyToId``
on ``/send`` and ``/send-attachment``; the shared ``reply_to_mode`` (first = only when the user threaded | all | off).
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType
from plugins.platforms.photon import adapter as photon_adapter
from plugins.platforms.photon.adapter import PhotonAdapter

PHONE = "+15550001234"
DM = f"any;-;{PHONE}"


@pytest.fixture(autouse=True)
def _isolated_sent_index(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    # rich_sent_store writes under HERMES_HOME/state; keep each test's index private.
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))


def _make_adapter(monkeypatch: pytest.MonkeyPatch, reply_to_mode: str = "first", **extra: Any) -> PhotonAdapter:
    monkeypatch.setenv("PHOTON_PROJECT_ID", "test-project-id")
    monkeypatch.setenv("PHOTON_PROJECT_SECRET", "test-project-secret")
    return PhotonAdapter(PlatformConfig(enabled=True, token="", extra=dict(extra), reply_to_mode=reply_to_mode))


def _capture_handled(adapter: PhotonAdapter, monkeypatch: pytest.MonkeyPatch) -> List[MessageEvent]:
    captured: List[MessageEvent] = []

    async def fake_handle(event: MessageEvent) -> None:
        captured.append(event)

    monkeypatch.setattr(adapter, "handle_message", fake_handle)
    return captured


def _capture_sidecar(adapter: PhotonAdapter, message_id: str = "out-1") -> List[Tuple[str, Dict[str, Any]]]:
    calls: List[Tuple[str, Dict[str, Any]]] = []

    async def fake_call(path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        calls.append((path, body))
        return {"ok": True, "messageId": message_id}

    adapter._sidecar_call = fake_call  # type: ignore[assignment]
    return calls


def _reply_event(inner: Dict[str, Any], *, message_id: str = "in-2", target_id: str = "out-0",
                 direction: str = "outbound", target_text: str | None = "earlier answer") -> Dict[str, Any]:
    return {
        "messageId": message_id,
        "space": {"id": DM, "type": "dm", "phone": PHONE},
        "sender": {"id": PHONE},
        "timestamp": "2026-09-25T18:00:00Z",
        "content": {"type": "reply", "content": inner, "targetMessageId": target_id,
                    "targetDirection": direction, "targetText": target_text},
    }


@pytest.mark.asyncio
async def test_threaded_text_reply_reaches_agent_with_context(monkeypatch):
    adapter = _make_adapter(monkeypatch)
    handled = _capture_handled(adapter, monkeypatch)

    await adapter._dispatch_inbound(_reply_event({"type": "text", "text": "yes do it"}))

    assert len(handled) == 1
    event = handled[0]
    assert event.text == "yes do it"
    assert event.reply_to_message_id == "out-0"
    assert event.reply_to_text == "earlier answer"
    assert event.reply_to_is_own_message is True


# 1x1 transparent PNG (passes the base adapter's image magic check).
_PNG_1X1_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="


@pytest.mark.asyncio
async def test_threaded_photo_reply_keeps_the_photo(monkeypatch):
    """The inner content goes through the normal ladder, so a threaded photo is not reduced to text."""
    adapter = _make_adapter(monkeypatch)
    handled = _capture_handled(adapter, monkeypatch)

    inner = {"type": "attachment", "name": "pic.png", "mimeType": "image/png", "size": 68,
             "data": _PNG_1X1_B64, "encoding": "base64"}
    await adapter._dispatch_inbound(_reply_event(inner))

    event = handled[0]
    assert event.message_type == MessageType.PHOTO
    assert event.media_types == ["image/png"]
    assert len(event.media_urls) == 1
    assert event.reply_to_message_id == "out-0"


@pytest.mark.asyncio
async def test_reply_to_users_own_message_is_not_marked_own(monkeypatch):
    adapter = _make_adapter(monkeypatch)
    handled = _capture_handled(adapter, monkeypatch)

    await adapter._dispatch_inbound(_reply_event({"type": "text", "text": "ctx"}, direction="inbound"))

    assert handled[0].text == "ctx"
    assert handled[0].reply_to_is_own_message is False


@pytest.mark.asyncio
async def test_malformed_reply_envelope_keeps_fallback_marker(monkeypatch):
    adapter = _make_adapter(monkeypatch)
    handled = _capture_handled(adapter, monkeypatch)

    await adapter._dispatch_inbound(_reply_event({"type": "unknown"}))

    assert handled[0].text == "[Photon content type not handled: reply]"


@pytest.mark.asyncio
async def test_reply_to_our_message_hydrates_quoted_text_from_sent_index(monkeypatch):
    adapter = _make_adapter(monkeypatch)
    handled = _capture_handled(adapter, monkeypatch)
    calls = _capture_sidecar(adapter, message_id="out-1")

    # We send, the user later swipe-replies to that bubble; spectrum gives no target text.
    await adapter.send(DM, "the plan is A then B")
    assert calls[-1][0] == "/send"
    await adapter._dispatch_inbound(
        _reply_event({"type": "text", "text": "do B first"}, target_id="out-1", target_text=None))

    assert handled[-1].text == "do B first"
    assert handled[-1].reply_to_message_id == "out-1"
    assert handled[-1].reply_to_text == "the plan is A then B"
    assert handled[-1].reply_to_is_own_message is True


@pytest.mark.asyncio
async def test_send_to_bare_phone_is_found_from_dm_guid(monkeypatch):
    """Cron/standalone sends often target the bare E.164; replies arrive in the DM GUID space."""
    adapter = _make_adapter(monkeypatch)
    handled = _capture_handled(adapter, monkeypatch)
    _capture_sidecar(adapter, message_id="out-7")

    await adapter.send(PHONE, "morning briefing")
    await adapter._dispatch_inbound(
        _reply_event({"type": "text", "text": "more on item 2"}, target_id="out-7", target_text=None))

    assert handled[-1].reply_to_text == "morning briefing"


@pytest.mark.asyncio
async def test_attachment_send_is_recorded_as_label(monkeypatch, tmp_path):
    monkeypatch.setattr(PhotonAdapter, "validate_media_delivery_path",
                        staticmethod(lambda p: p if os.path.exists(p) else None))
    img = tmp_path / "chart.png"
    img.write_bytes(b"\x89PNG fake")
    adapter = _make_adapter(monkeypatch)
    handled = _capture_handled(adapter, monkeypatch)
    _capture_sidecar(adapter, message_id="out-3")

    await adapter.send_image_file(DM, str(img))
    await adapter._dispatch_inbound(
        _reply_event({"type": "text", "text": "what is this"}, target_id="out-3", target_text=None))

    assert handled[-1].reply_to_text == "[attachment: chart.png]"


@pytest.mark.asyncio
async def test_standalone_send_records_text(monkeypatch):
    monkeypatch.setenv("PHOTON_SIDECAR_TOKEN", "tok")

    class _Resp:
        status_code = 200

        @staticmethod
        def json() -> Dict[str, Any]:
            return {"ok": True, "messageId": "cron-1"}

    class _FakeClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url: str, json: Dict[str, Any], headers=None):
            return _Resp()

    monkeypatch.setattr(photon_adapter.httpx, "AsyncClient", _FakeClient)
    cfg = PlatformConfig(enabled=True, token="", extra={})
    result = await photon_adapter._standalone_send(cfg, PHONE, "daily digest")
    assert result.get("success") is True

    adapter = _make_adapter(monkeypatch)
    handled = _capture_handled(adapter, monkeypatch)
    await adapter._dispatch_inbound(
        _reply_event({"type": "text", "text": "thanks"}, target_id="cron-1", target_text=None))

    assert handled[-1].reply_to_text == "daily digest"


@pytest.mark.asyncio
async def test_unknown_reply_target_leaves_text_empty(monkeypatch):
    adapter = _make_adapter(monkeypatch)
    handled = _capture_handled(adapter, monkeypatch)

    await adapter._dispatch_inbound(_reply_event({"type": "text", "text": "?"}, target_text=None))

    assert handled[-1].text == "?"
    assert handled[-1].reply_to_text is None


# --- outbound: answer inside the thread ------------------------------------------------


@pytest.mark.asyncio
async def test_answer_to_threaded_message_is_threaded(monkeypatch):
    adapter = _make_adapter(monkeypatch)
    _capture_handled(adapter, monkeypatch)
    calls = _capture_sidecar(adapter)

    await adapter._dispatch_inbound(_reply_event({"type": "text", "text": "hi"}, message_id="in-9"))
    result = await adapter.send(DM, "answer", reply_to="in-9")

    assert result.success
    path, body = calls[-1]
    assert path == "/send"
    assert body["replyToId"] == "in-9"


@pytest.mark.asyncio
async def test_answer_to_plain_message_is_not_threaded_by_default(monkeypatch):
    adapter = _make_adapter(monkeypatch)
    calls = _capture_sidecar(adapter)

    await adapter.send(DM, "answer", reply_to="plain-1")

    assert "replyToId" not in calls[-1][1]


@pytest.mark.asyncio
async def test_threading_modes(monkeypatch):
    all_adapter = _make_adapter(monkeypatch, reply_to_mode="all")
    calls = _capture_sidecar(all_adapter)
    await all_adapter.send(DM, "answer", reply_to="plain-1")
    assert calls[-1][1]["replyToId"] == "plain-1"

    off_adapter = _make_adapter(monkeypatch, reply_to_mode="off")
    _capture_handled(off_adapter, monkeypatch)
    calls = _capture_sidecar(off_adapter)
    await off_adapter._dispatch_inbound(_reply_event({"type": "text", "text": "hi"}, message_id="in-3"))
    await off_adapter.send(DM, "answer", reply_to="in-3")
    assert "replyToId" not in calls[-1][1]


def test_threading_mode_comes_from_platform_reply_to_mode(monkeypatch):
    monkeypatch.setenv("PHOTON_PROJECT_ID", "test-project-id")
    monkeypatch.setenv("PHOTON_PROJECT_SECRET", "test-project-secret")
    cfg = PlatformConfig.from_dict({"enabled": True, "reply_to_mode": "off"})
    assert PhotonAdapter(cfg)._reply_to_mode == "off"
    assert PhotonAdapter(PlatformConfig.from_dict({"enabled": True}))._reply_to_mode == "first"


@pytest.mark.asyncio
async def test_threaded_answer_skips_richlink_path(monkeypatch):
    adapter = _make_adapter(monkeypatch, reply_to_mode="all")
    calls = _capture_sidecar(adapter)

    await adapter.send(DM, "https://example.com", reply_to="in-1")

    assert [path for path, _ in calls] == ["/send"]
    assert calls[-1][1]["replyToId"] == "in-1"


@pytest.mark.asyncio
async def test_attachment_answer_to_threaded_message_is_threaded(monkeypatch, tmp_path):
    monkeypatch.setattr(PhotonAdapter, "validate_media_delivery_path",
                        staticmethod(lambda p: p if os.path.exists(p) else None))
    img = tmp_path / "chart.png"
    img.write_bytes(b"\x89PNG fake")
    adapter = _make_adapter(monkeypatch)
    _capture_handled(adapter, monkeypatch)
    calls = _capture_sidecar(adapter)

    await adapter._dispatch_inbound(_reply_event({"type": "text", "text": "chart?"}, message_id="in-4"))
    await adapter.send_image_file(DM, str(img), reply_to="in-4")

    path, body = calls[-1]
    assert path == "/send-attachment"
    assert body["replyToId"] == "in-4"
