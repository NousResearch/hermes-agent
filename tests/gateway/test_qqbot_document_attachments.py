"""QQBot document attachments reach the gateway's shared document path (#15595).

A PDF/Office file sent over QQ used to become a bare ``[file: name (path)]`` text line, so the
gateway never added its document note and the agent did not know to extract the contents.
"""

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.event import MessageType
from gateway.run import GatewayRunner


def _make_adapter(image_path="/cache/img_abc.png"):
    from gateway.platforms.qqbot.adapter import QQAdapter
    adapter = QQAdapter(PlatformConfig(enabled=True, extra={"app_id": "a", "client_secret": "b"}))
    cached = {"https://qq-cdn/invoice": "/cache/doc_abc_invoice.pdf", "https://qq-cdn/pic": image_path}

    async def fake_download(url, ct, original_name=""):
        return cached[url]

    adapter._download_and_cache = fake_download  # type: ignore[assignment]
    return adapter


async def _ingest(adapter, d):
    events = []

    async def capture(event):
        events.append(event)

    adapter.handle_message = capture  # type: ignore[assignment]
    await adapter._ingest(
        d, "msg-1", d.get("content", ""), d.get("attachments"), "",
        chat_id="user-1", qq_chat_type="c2c", chat_type="dm", user_id="user-1")
    assert len(events) == 1
    return events[0]


@pytest.mark.asyncio
async def test_pdf_upload_reaches_agent_with_extraction_note():
    adapter = _make_adapter()
    event = await _ingest(adapter, {"content": "", "attachments": [
        {"content_type": "file", "url": "https://qq-cdn/invoice", "filename": "invoice.pdf"}]})

    assert event.message_type == MessageType.DOCUMENT
    assert event.media_urls == ["/cache/doc_abc_invoice.pdf"]
    assert event.media_types == ["application/pdf"]

    prepared = GatewayRunner._prepend_inbound_document_notes(event, event.text)
    assert "invoice.pdf" in prepared
    assert "/cache/doc_abc_invoice.pdf" in prepared
    assert "read_file" in prepared


@pytest.mark.asyncio
async def test_quoted_document_with_image_keeps_both_attachments(tmp_path):
    image = tmp_path / "img_abc.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    adapter = _make_adapter(str(image))
    event = await _ingest(adapter, {
        "content": "what does this say?",
        "attachments": [{"content_type": "image/png", "url": "https://qq-cdn/pic", "filename": "p.png"}],
        "message_type": 103,
        "msg_elements": [{"content": "", "attachments": [
            {"content_type": "application/pdf", "url": "https://qq-cdn/invoice", "filename": "invoice.pdf"}]}],
    })

    assert event.source.platform == Platform.QQBOT
    assert event.media_urls == [str(image), "/cache/doc_abc_invoice.pdf"]
    assert event.media_types == ["image/png", "application/pdf"]
    # The image stays on the vision path; only the PDF gets a document note, and it is not
    # claimed as already inlined.
    prepared = GatewayRunner._prepend_inbound_document_notes(event, event.text)
    assert prepared.count("[The user sent a document") == 1
    assert "img_abc.png" not in prepared
    assert "[Quoted message]: (file)" in event.text
