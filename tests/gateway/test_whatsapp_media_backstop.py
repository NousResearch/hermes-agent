"""WhatsApp MEDIA: directive backstop tests (issue #43656).

Verify that MEDIA:<path> directives in outgoing content are extracted and
delivered as native attachments, not leaked as visible text.
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.platforms.whatsapp import WhatsAppAdapter
from gateway.config import PlatformConfig


class _FakeResponse:
    """Async-context-manager response for aiohttp."""
    def __init__(self, status=200, data=None):
        self.status = status
        self._data = data or {"messageId": "msg"}

    async def json(self):
        return self._data

    async def text(self):
        return "error"

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        pass


def _make_adapter():
    adapter = WhatsAppAdapter(PlatformConfig(enabled=True, extra={"session_name": "test"}))
    adapter._running = True
    adapter._http_session = MagicMock()
    adapter._bridge_port = 8080
    return adapter


@pytest.mark.asyncio
async def test_media_directive_stripped_from_text(tmp_path):
    """MEDIA:<path> tags must not appear in the text sent to the bridge."""
    media_file = tmp_path / "report.png"
    media_file.write_bytes(b"\x89PNG")

    adapter = _make_adapter()
    sent_payloads = []

    def _fake_post(url, json=None, **kw):
        sent_payloads.append(json)
        return _FakeResponse(200)

    adapter._http_session.post = _fake_post

    with patch.object(adapter, "send_image_file", new_callable=AsyncMock) as mock_img:
        mock_img.return_value = MagicMock(success=True)
        await adapter.send("chat1", f"Here is the report:\nMEDIA:{media_file}")

    # Text sent to bridge must NOT contain MEDIA: or the file path
    assert sent_payloads, "Bridge should have received a text message"
    text = sent_payloads[0]["message"]
    assert "MEDIA:" not in text
    assert str(media_file) not in text

    # Image file should have been sent natively
    mock_img.assert_awaited_once()
    call_kwargs = mock_img.call_args
    assert str(media_file) in str(call_kwargs)


@pytest.mark.asyncio
async def test_no_media_directive_passthrough():
    """Content without MEDIA: tags should pass through unchanged."""
    adapter = _make_adapter()
    sent_payloads = []

    def _fake_post(url, json=None, **kw):
        sent_payloads.append(json)
        return _FakeResponse(200)

    adapter._http_session.post = _fake_post

    result = await adapter.send("chat1", "Hello, world!")
    assert result.success is True
    assert len(sent_payloads) == 1
    assert "Hello, world!" in sent_payloads[0]["message"]


@pytest.mark.asyncio
async def test_multiple_media_directives(tmp_path):
    """Multiple MEDIA: tags should all be extracted and delivered."""
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"\xff\xd8\xff")
    doc = tmp_path / "notes.pdf"
    doc.write_bytes(b"%PDF")

    adapter = _make_adapter()

    def _fake_post(url, json=None, **kw):
        return _FakeResponse(200)

    adapter._http_session.post = _fake_post

    with patch.object(adapter, "send_image_file", new_callable=AsyncMock) as mock_img, \
         patch.object(adapter, "send_document", new_callable=AsyncMock) as mock_doc:
        mock_img.return_value = MagicMock(success=True)
        mock_doc.return_value = MagicMock(success=True)
        await adapter.send("chat1", f"Here:\nMEDIA:{img}\nMEDIA:{doc}")

    mock_img.assert_awaited_once()
    mock_doc.assert_awaited_once()


@pytest.mark.asyncio
async def test_media_delivery_failure_does_not_block_text(tmp_path, caplog):
    """If media delivery fails, text should still be sent."""
    media_file = tmp_path / "broken.mp4"
    media_file.write_bytes(b"\x00\x00\x00\x18ftyp")

    adapter = _make_adapter()
    sent_payloads = []

    def _fake_post(url, json=None, **kw):
        sent_payloads.append(json)
        return _FakeResponse(200)

    adapter._http_session.post = _fake_post

    with patch.object(adapter, "send_video", new_callable=AsyncMock) as mock_video:
        mock_video.side_effect = RuntimeError("bridge down")
        result = await adapter.send("chat1", f"Video:\nMEDIA:{media_file}")

    assert result.success is True
    assert sent_payloads
    assert "MEDIA:" not in sent_payloads[0]["message"]


@pytest.mark.asyncio
async def test_only_whitespace_content_after_strip_skipped(tmp_path):
    """If content is only MEDIA: tags, no empty text message should be sent."""
    media_file = tmp_path / "image.png"
    media_file.write_bytes(b"\x89PNG")

    adapter = _make_adapter()
    sent_payloads = []

    def _fake_post(url, json=None, **kw):
        sent_payloads.append(json)
        return _FakeResponse(200)

    adapter._http_session.post = _fake_post

    with patch.object(adapter, "send_image_file", new_callable=AsyncMock) as mock_img:
        mock_img.return_value = MagicMock(success=True)
        result = await adapter.send("chat1", f"MEDIA:{media_file}")

    assert result.success is True
    # Image delivered, no empty text chunk sent
    mock_img.assert_awaited_once()
    assert len(sent_payloads) == 0


@pytest.mark.asyncio
async def test_protected_span_media_directive_stripped_from_visible_text():
    """MEDIA: inside fenced code blocks must not be uploaded but must not leak
    as visible text either (reviewer feedback on #43679)."""
    adapter = _make_adapter()
    sent_payloads = []

    def _fake_post(url, json=None, **kw):
        sent_payloads.append(json)
        return _FakeResponse(200)

    adapter._http_session.post = _fake_post

    # MEDIA: inside a fenced code block — extract_media skips it (protected span)
    # but _strip_media_directives should still clean it from visible text.
    content = (
        "Here is an example:\n"
        "```json\n"
        '  "path": "MEDIA:/home/agent/private/report.docx"\n'
        "```\n"
        "Done."
    )

    with patch.object(adapter, "send_document", new_callable=AsyncMock) as mock_doc:
        result = await adapter.send("chat1", content)

    assert result.success is True
    # No attachment should be sent (protected span)
    mock_doc.assert_not_awaited()
    # Visible text must not contain MEDIA: or the internal path
    assert sent_payloads, "Bridge should have received a text message"
    text = sent_payloads[0]["message"]
    assert "MEDIA:" not in text
    assert "/home/agent/private/report.docx" not in text
    # Normal trailing text should survive
    assert "Done." in text
