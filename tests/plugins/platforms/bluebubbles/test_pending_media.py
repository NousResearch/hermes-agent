"""Known attachments must recover without requiring another provider webhook (#30989/#127423)."""

from unittest.mock import AsyncMock

import pytest

from tests.gateway.bluebubbles_test_support import Request, adapter, settle


@pytest.mark.asyncio
async def test_caption_and_known_image_recover_without_another_webhook(monkeypatch):
    instance = adapter()
    delivered = []
    attempts = 0

    async def download(*_):
        nonlocal attempts
        attempts += 1
        return "/tmp/synthetic-image.jpg" if attempts >= 4 else None

    async def accept(event):
        event._gateway_accepted = True
        delivered.append(event)

    async def wait(_delay):
        return None

    monkeypatch.setattr(instance, "_download_attachment", download)
    monkeypatch.setattr(instance, "handle_message", accept)
    monkeypatch.setattr(instance, "_wait_for_attachment_retry", wait, raising=False)
    result = await instance._handle_webhook(Request(attachments=[{"guid": "image", "mimeType": "image/jpeg"}]))
    await settle(instance)
    assert [(e.text, e.media_urls) for e in delivered] == [("caption", ["/tmp/synthetic-image.jpg"])]
    assert result.status == 200


@pytest.mark.asyncio
async def test_timeout_delivers_caption_and_notice_once_and_late_webhook_only_media(monkeypatch):
    from types import SimpleNamespace
    from plugins.platforms.bluebubbles import inbound

    instance = adapter()
    delivered = []
    now = [0.0]
    ready = [False]
    monkeypatch.setattr(inbound, "time", SimpleNamespace(monotonic=lambda: now[0]))

    async def wait(delay):
        now[0] += 100.0  # Advance a virtual clock; exercise the five-minute deadline without wall-clock waits.

    async def download(*_):
        return "/tmp/synthetic-image.jpg" if ready[0] else None

    async def accept(event):
        event._gateway_accepted = True
        delivered.append(event)

    monkeypatch.setattr(instance, "_wait_for_attachment_retry", wait)
    monkeypatch.setattr(instance, "_download_attachment", download)
    monkeypatch.setattr(instance, "handle_message", accept)
    request = Request(attachments=[{"guid": "image", "mimeType": "image/jpeg"}])
    assert (await instance._handle_webhook(request)).status == 200
    await settle(instance)
    assert len(delivered) == 1
    assert delivered[0].text.startswith("caption\n\n")
    assert "could not be retrieved" in delivered[0].text
    assert delivered[0].media_urls == []
    ready[0] = True
    assert (await instance._handle_webhook(request)).status == 200
    assert delivered[1].text == "(attachment)"
    assert delivered[1].media_urls == ["/tmp/synthetic-image.jpg"]
    await instance._handle_webhook(request)
    assert len(delivered) == 2


@pytest.mark.asyncio
async def test_pending_image_does_not_block_another_message_and_survives_replacement(monkeypatch):
    import asyncio
    from tests.gateway.bluebubbles_test_support import _FakeBlueBubblesRequest

    old = adapter()
    waiting = asyncio.Event()
    release = asyncio.Event()
    delivered = []

    async def wait(_delay):
        waiting.set()
        await release.wait()

    async def accept(event):
        event._gateway_accepted = True
        delivered.append(event)

    monkeypatch.setattr(old, "_wait_for_attachment_retry", wait)
    monkeypatch.setattr(old, "_download_attachment", AsyncMock(return_value=None))
    monkeypatch.setattr(old, "handle_message", accept)
    await old._handle_webhook(Request(attachments=[{"guid": "image", "mimeType": "image/jpeg"}]))
    await waiting.wait()
    await old._handle_webhook(_FakeBlueBubblesRequest({"type": "new-message", "data": {
        "guid": "distinct", "text": "another message", "handle": {"address": "sender@example.test"},
        "chats": [{"guid": "iMessage;-;sender@example.test", "style": 45}],
    }}, password="synthetic-secret"))
    assert [e.text for e in delivered] == ["another message"]
    await old.disconnect()
    replacement = adapter()
    replacement._dedup = old._dedup
    monkeypatch.setattr(replacement, "_download_attachment", AsyncMock(return_value="/tmp/synthetic.jpg"))
    monkeypatch.setattr(replacement, "handle_message", accept)
    monkeypatch.setattr(replacement, "_wait_for_attachment_retry", AsyncMock())
    replacement.adopt_held_inbound(old)
    await settle(replacement)
    assert [(e.text, e.media_urls) for e in delivered] == [
        ("another message", []), ("caption", ["/tmp/synthetic.jpg"])]


@pytest.mark.asyncio
async def test_known_image_http_500_for_69_seconds_then_real_cache_write(monkeypatch):
    import base64
    from pathlib import Path
    from types import SimpleNamespace
    import httpx
    from plugins.platforms.bluebubbles import inbound

    instance = adapter()
    now = [0.0]
    delivered = []
    png = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jR7kAAAAASUVORK5CYII=")
    monkeypatch.setattr(inbound, "time", SimpleNamespace(monotonic=lambda: now[0]))
    attachment = {"guid": "image", "mimeType": "image/png"}

    async def wait(delay):
        now[0] += delay

    def transport(request):
        if "/attachment/" in request.url.path:
            return httpx.Response(500) if now[0] < 69 else httpx.Response(200, content=png)
        return httpx.Response(200, json={"data": {"attachments": [attachment],
            "chats": [{"guid": "iMessage;-;sender@example.test", "style": 45}]}})

    async def accept(event):
        event._gateway_accepted = True
        delivered.append(event)

    monkeypatch.setattr(instance, "_wait_for_attachment_retry", wait)
    monkeypatch.setattr(instance, "handle_message", accept)
    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
        instance.client = client
        assert (await instance._handle_webhook(Request(attachments=[attachment]))).status == 200
        assert delivered == []
        await settle(instance)
    assert len(delivered) == 1
    assert delivered[0].text == "caption"
    assert Path(delivered[0].media_urls[0]).read_bytes() == png
    assert now[0] >= 69
