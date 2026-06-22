"""Text document injection through Baileys and Cloud inbound event builders."""

from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.whatsapp_cloud import WhatsAppCloudAdapter
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from plugins.platforms.whatsapp.adapter import WhatsAppAdapter


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["baileys", "cloud"])
@pytest.mark.parametrize(
    "filename,mime,content,inline",
    [
        ("config.toml", "application/octet-stream", '[tool]\nname = "x"\n', True),
        ("main.go", "application/octet-stream", "package main\n", True),
        ("deploy.sh", "application/octet-stream", "#!/bin/sh\necho hi\n", True),
        ("notes.md", "", "# heading\n", True),
        ("notes.custom", "text/plain", "unknown extension text\n", True),
        ("blob.bin", "application/octet-stream", "binary placeholder", False),
        ("large.toml", "text/plain", "x" * 102401, False),
    ],
)
async def test_whatsapp_document_injection(tmp_path, transport, filename, mime, content, inline):
    token = set_hermes_home_override(str(tmp_path))
    try:
        doc = tmp_path / "cache" / "documents" / filename
        doc.parent.mkdir(parents=True)
        doc.write_text(content, encoding="utf-8")
        adapter_type = WhatsAppAdapter if transport == "baileys" else WhatsAppCloudAdapter
        adapter = adapter_type(PlatformConfig(enabled=True))
        adapter._should_process_message = lambda data: True
        if transport == "baileys":
            event = await adapter._build_message_event({
                "isGroup": False, "chatId": "123@s.whatsapp.net",
                "senderId": "123@s.whatsapp.net", "senderName": "Alice",
                "body": "caption", "hasMedia": True, "mediaType": "document",
                "mediaUrls": [str(doc)], "mime": mime,
            })
        else:
            adapter._download_media_to_cache = AsyncMock(return_value=(str(doc), mime))
            event = await adapter._build_message_event_from_cloud({
                "from": "123", "type": "document",
                "document": {"id": "media-id", "filename": filename,
                             "mime_type": mime, "caption": "caption"},
            }, {}, {})
        assert event is not None
        assert event.media_urls == [str(doc)]
        assert ("[Content of" in event.text) is inline
        assert "caption" in event.text
        if inline:
            assert content.strip() in event.text
        else:
            assert content not in event.text
    finally:
        reset_hermes_home_override(token)
