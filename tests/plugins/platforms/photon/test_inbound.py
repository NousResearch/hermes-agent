"""Inbound dispatch + dedup tests for PhotonAdapter.

These exercise the sidecar-event stream and parsing without spawning the
Node sidecar or binding ports.
"""
from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType
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
async def test_findmy_card_preserves_only_sdk_owned_marker(monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess
    from plugins.platforms.photon import adapter as module
    source = (Path(module.__file__).parent / "sidecar/index.mjs").read_text()
    content = source[source.index("async function normalizeContent(content) {"):source.index("// The iMessage SDK exposes")]
    event = source[source.index("async function normalizeEvent(space, message) {"):source.index("function inboundStreamErrorMessage(e) {")]
    script = content + event + r"""
const assert = (await import('node:assert/strict')).default;
const bundle='com.apple.messages.MSMessageExtensionBalloonPlugin:1234567890:com.apple.findmy.FindMyMessagesApp';
const space={id:'own-chat',type:'dm'};
const message={id:'own-message',direction:'inbound',sender:{id:'own-sender'},balloonBundleId:bundle,
  content:{type:'custom',imessage_type:'unsupported-message',latitude:99,appleFindMyShare:true}};
const normalized=await normalizeEvent(space,message);
assert.deepEqual(normalized.content,{type:'custom',appleFindMyShare:true});
for(const patch of [{balloonBundleId:undefined},{balloonBundleId:bundle+'\n'},
  {balloonBundleId:bundle+'.spoof'},{balloonBundleId:bundle.replace('findmy','other')},
  {direction:'outbound'},{content:{type:'text',text:bundle,appleFindMyShare:true}}]) {
  const result=await normalizeEvent(space,{...message,...patch});
  assert.equal(result.content.appleFindMyShare,undefined);
}
assert.equal(normalized.balloonBundleId,undefined);
console.log(JSON.stringify(normalized));
"""
    result = subprocess.run(["node", "--input-type=module", "-e", script],
                            check=True, capture_output=True, text=True, timeout=5)
    raw = json.loads(result.stdout)
    adapter = _make_adapter(monkeypatch)
    captured = _capture(adapter, monkeypatch)
    await adapter._dispatch_inbound(raw)
    assert len(captured) == 1
    assert captured[0].raw_message["content"] == {"type": "custom", "appleFindMyShare": True}
    assert captured[0].raw_message["messageId"] == captured[0].message_id
    assert captured[0].text == "[Photon content type not handled: custom]"
    assert captured[0].media_urls == []


@pytest.mark.asyncio
async def test_dispatch_read_receipt_does_not_wake_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter(monkeypatch)
    captured = _capture(adapter, monkeypatch)
    receipt = _dm_event("", msg_id="spc-read-1")
    receipt["content"] = {
        "type": "read",
        "targetMessageId": "bot-msg-1",
        "targetDirection": "outbound",
    }

    await adapter._dispatch_inbound(receipt)

    assert captured == []


@pytest.mark.asyncio
async def test_dispatch_read_receipt_alias_does_not_wake_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Some spectrum-ts streams label receipts ``read_receipt`` — same drop."""
    adapter = _make_adapter(monkeypatch)
    captured = _capture(adapter, monkeypatch)
    receipt = _dm_event("", msg_id="spc-read-2")
    receipt["content"] = {
        "type": "read_receipt",
        "targetMessageId": "bot-msg-2",
        "targetDirection": "outbound",
    }

    await adapter._dispatch_inbound(receipt)

    assert captured == []


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


@pytest.mark.asyncio
async def test_ndjson_stream_preserves_unicode_line_separators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter(monkeypatch)
    separated = "first\u2028second\u2029third\u0085fourth"
    payloads = [
        json.dumps(_dm_event(separated, msg_id="unicode-lines"), ensure_ascii=False),
        json.dumps(_dm_event("ordinary", msg_id="ordinary-line")),
    ]
    stream = payloads[0] + "\n\n" + payloads[1]
    received: List[str] = []

    class ChunkedResponse:
        status_code = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def aiter_text(self):
            for chunk in (stream[:17], stream[17:43], stream[43:]):
                yield chunk

        async def aiter_lines(self):
            for line in stream.splitlines():
                yield line

    class Client:
        def stream(self, *_args, **_kwargs):
            return ChunkedResponse()

    async def capture_line(line: str) -> None:
        received.append(line)
        if len(received) == 2:
            adapter._inbound_running = False

    adapter._http_client = Client()
    adapter._inbound_running = True
    monkeypatch.setattr(adapter, "_on_inbound_line", capture_line)

    await adapter._inbound_loop()

    assert received == payloads


def test_is_duplicate_window(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _make_adapter(monkeypatch)
    assert adapter._dedup.is_duplicate("id-1") is False
    assert adapter._dedup.is_duplicate("id-1") is True
    assert adapter._dedup.is_duplicate("id-2") is False
    assert adapter._dedup.is_duplicate("id-1") is True  # still dup


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
