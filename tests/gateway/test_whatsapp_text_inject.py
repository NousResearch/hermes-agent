"""Inbound document text-injection for WhatsApp.

#50563 unified inline text-injection across discord/slack/telegram via the shared
`_TEXT_INJECT_EXTENSIONS` set + a text/* MIME fallback, but left WhatsApp on its
old narrow hardcoded allowlist. These tests assert WhatsApp now inlines the same
broad text/code/config set (e.g. .toml/.go/.sh) while still skipping binary
documents.
"""

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.whatsapp.adapter import WhatsAppAdapter


def _make_adapter():
    adapter = WhatsAppAdapter(PlatformConfig(enabled=True))
    # Focus the test on the inline-injection gate, not the allow/deny policy.
    adapter._should_process_message = lambda data: True
    return adapter


def _doc_data(doc_path):
    return {
        "isGroup": False,
        "chatId": "6281234567890@s.whatsapp.net",
        "senderId": "6281234567890@s.whatsapp.net",
        "senderName": "Alice",
        "body": "",
        "hasMedia": True,
        "mediaType": "document",
        "mediaUrls": [str(doc_path)],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "filename,content",
    [
        ("config.toml", '[tool]\nname = "x"\n'),  # broadened by #50563, not in old set
        ("main.go", "package main\n"),            # broadened by #50563
        ("deploy.sh", "#!/bin/sh\necho hi\n"),    # broadened by #50563
        ("notes.md", "# heading\n"),              # already in old set — must still inline
    ],
)
async def test_whatsapp_inlines_text_documents(tmp_path, filename, content):
    adapter = _make_adapter()
    doc = tmp_path / filename
    doc.write_text(content, encoding="utf-8")

    event = await adapter._build_message_event(_doc_data(doc))

    assert event is not None
    assert "[Content of" in event.text
    assert content.strip() in event.text


@pytest.mark.asyncio
async def test_whatsapp_skips_binary_document(tmp_path):
    adapter = _make_adapter()
    doc = tmp_path / "blob.bin"
    doc.write_bytes(b"\x00\x01\x02not-text")

    event = await adapter._build_message_event(_doc_data(doc))

    assert event is not None
    assert "[Content of" not in event.text
