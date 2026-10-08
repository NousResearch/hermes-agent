"""Real gateway delivery with recording-only sidecar transport (no Photon network)."""
import asyncio
import os

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import _ExtractedResponse
from plugins.platforms.photon.adapter import PhotonAdapter, PhotonSidecarError


@pytest.fixture
def adapter(monkeypatch, request):
    monkeypatch.setenv("PHOTON_PROJECT_ID", "test-project")
    monkeypatch.setenv("PHOTON_PROJECT_SECRET", "test-secret")
    return PhotonAdapter(PlatformConfig(enabled=True, reply_to_mode=getattr(request, "param", "all"), typing_indicator=False))


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter,inbound_threaded", [("all", False), ("first", True), ("first", False), ("off", True)], indirect=["adapter"])
async def test_gateway_attachment_delivery_carries_turn_anchor(adapter, monkeypatch, tmp_path, inbound_threaded):
    paths = []
    for name in ("image.png", "voice.m4a", "video.mp4", "document.pdf"):
        path = tmp_path / name
        path.write_bytes(b"fixture")
        paths.append(str(path))
    monkeypatch.setattr(adapter, "validate_media_delivery_path", lambda p: p if os.path.exists(p) else None)
    async def cache(url):
        return paths[0]
    monkeypatch.setattr("gateway.platforms.base.cache_image_from_url", cache)
    calls = []
    async def transport(path, body):
        await asyncio.sleep(0)
        calls.append((path, body))
        return {"ok": True, "messageId": "sent"}
    monkeypatch.setattr(adapter, "_sidecar_call", transport)
    metadata = {"notify": True, "hermes_profile": "fixture"}
    extracted = _ExtractedResponse(text_content="", images=[("https://example.test/image.png", "caption")],
        media_files=[(paths[0], False), (paths[1], True), (paths[2], False), (paths[3], False)],
        local_files=[], force_document_attachments=False, pre_extract="media")
    results = []
    handled = {}
    async def handle(event):
        handled[event.message_id] = event
    monkeypatch.setattr(adapter, "handle_message", handle)
    for anchor in ("turn-a", "turn-b"):
        content = {"type": "text", "text": "generate media"}
        if inbound_threaded:
            content = {"type": "reply", "content": content, "targetMessageId": "earlier"}
        await adapter._dispatch_inbound({"messageId": anchor, "space": {"id": anchor, "type": "dm"},
            "sender": {"id": "fixture-user"}, "content": content})
    async def deliver(anchor):
        await adapter._deliver_attachments(handled[anchor], extracted, metadata, anything_sent=False, record_delivery=results.append)
    await asyncio.gather(deliver("turn-a"), deliver("turn-b"))
    assert len(calls) == 10
    assert all(path == "/send-attachment" for path, _ in calls)
    expected_thread = adapter.config.reply_to_mode == "all" or (adapter.config.reply_to_mode == "first" and inbound_threaded)
    assert all(body.get("replyToId") == (body["spaceId"] if expected_thread else None) for _, body in calls)
    assert all(result.success for result in results)
    assert metadata == {"notify": True, "hermes_profile": "fixture"}


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["timeout", "socket hang up", "upstream 503", "unknown failure"])
async def test_threaded_gateway_send_never_retries_ambiguous_failure(adapter, monkeypatch, failure):
    calls = []
    async def transport(path, body):
        calls.append((path, body))
        raise PhotonSidecarError(path=path, status_code=503, error=failure, error_class="upstream_transient", retryable=True)
    monkeypatch.setattr(adapter, "_sidecar_call", transport)
    result = await adapter._send_with_retry("chat", "answer", reply_to="anchor", max_retries=2, base_delay=0)
    assert not result.success
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_gateway_tts_voice_delivery_carries_event_anchor(adapter, monkeypatch, tmp_path):
    from types import SimpleNamespace

    path = tmp_path / "speech.m4a"
    path.write_bytes(b"voice fixture")
    monkeypatch.setattr(adapter, "validate_media_delivery_path", lambda p: p)
    calls = []
    async def transport(route, body):
        calls.append((route, body))
        return {"ok": True, "messageId": "voice-sent"}
    monkeypatch.setattr(adapter, "_sidecar_call", transport)
    event = SimpleNamespace(message_id="tts-trigger", source=SimpleNamespace(chat_id="chat", platform="photon"))
    results = []
    await adapter._play_tts_file(event, "answer", str(path), True, {"notify": True}, results.append)
    assert results[0].success
    assert calls[0][1].get("replyToId") == "tts-trigger"


@pytest.mark.asyncio
async def test_background_gateway_turn_delivers_generated_media(adapter, monkeypatch, tmp_path):
    path = tmp_path / "generated.pdf"
    path.write_bytes(b"document fixture")
    # Path policy is not under test; extraction and the background delivery lifecycle are real.
    monkeypatch.setattr(PhotonAdapter, "validate_media_delivery_path", staticmethod(lambda p, **kw: p))
    calls = []
    async def transport(route, body):
        calls.append((route, body))
        return {"ok": True, "messageId": "delivered"}
    monkeypatch.setattr(adapter, "_sidecar_call", transport)
    async def response(event):
        return f"MEDIA:{path}"
    adapter._message_handler = response
    async def handle(event):
        await adapter._process_message_background(event, "fixture-session")
    monkeypatch.setattr(adapter, "handle_message", handle)
    await adapter._dispatch_inbound({"messageId": "real-trigger", "space": {"id": "chat", "type": "dm"},
        "sender": {"id": "fixture-user"}, "content": {"type": "text", "text": "generate a document"}})
    sends = [body for route, body in calls if route == "/send-attachment"]
    assert len(sends) == 1
    assert sends[0]["replyToId"] == "real-trigger"
