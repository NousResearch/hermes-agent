"""WhatsApp adapter diagnostics never replace live payloads or error results."""
import logging
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform
from plugins.platforms.whatsapp.adapter import WhatsAppAdapter


@pytest.mark.asyncio
async def test_whatsapp_chat_info_exception_log_omits_raw_jid(caplog, monkeypatch):
    chat_id = "15551234567@s.whatsapp.net"
    exception = RuntimeError(
        f"GET http://127.0.0.1:3000/chat/{chat_id} failed for {chat_id}"
    )
    adapter = WhatsAppAdapter.__new__(WhatsAppAdapter)
    adapter._running = True
    adapter._http_session = MagicMock()
    adapter._bridge_port = 3000
    adapter._bridge_process = None
    adapter._http_session.get = MagicMock(side_effect=exception)
    monkeypatch.setitem(
        sys.modules,
        "aiohttp",
        SimpleNamespace(ClientTimeout=lambda **_kwargs: None),
    )

    with caplog.at_level(
        logging.DEBUG, logger="plugins.platforms.whatsapp.adapter"
    ):
        result = await adapter.get_chat_info(chat_id)

    assert result == {"name": chat_id, "type": "dm"}
    records = [
        record.message
        for record in caplog.records
        if record.name == "plugins.platforms.whatsapp.adapter"
        and record.message.startswith("Could not get WhatsApp chat info")
    ]
    assert len(records) == 1
    record = records[0]
    assert chat_id not in record
    assert "RuntimeError" in record

@pytest.mark.asyncio
async def test_whatsapp_clarify_error_log_omits_bridge_body(caplog, monkeypatch):
    private_marker = "15551234567-private-poll-question-marker"
    adapter = WhatsAppAdapter.__new__(WhatsAppAdapter)
    adapter.platform = Platform.WHATSAPP
    adapter.send_poll = AsyncMock(
        return_value=SimpleNamespace(
            success=False,
            error=private_marker,
        )
    )
    from gateway.platforms.base import BasePlatformAdapter, SendResult

    monkeypatch.setattr(
        BasePlatformAdapter,
        "send_clarify",
        AsyncMock(return_value=SendResult(success=False, error=private_marker)),
    )

    with caplog.at_level(
        logging.WARNING, logger="plugins.platforms.whatsapp.adapter"
    ):
        await adapter.send_clarify(
            "15551234567",
            "Pick one",
            ["A", "B"],
            "clarify-id",
            "session",
        )

    records = [
        record
        for record in caplog.records
        if record.name == "plugins.platforms.whatsapp.adapter"
        and "Native WhatsApp clarify poll failed" in record.message
    ]
    assert len(records) == 1
    assert private_marker not in records[0].message
    assert "error_detail_present=True" in records[0].message

@pytest.mark.asyncio
async def test_whatsapp_document_media_logs_omit_bridge_filename(
    tmp_path, monkeypatch, capsys
):
    """Inbound bridge paths and read errors never enter adapter stdout."""
    private_name = "PRIVATE_INBOUND_FILENAME_MARKER.txt"
    document = tmp_path / private_name
    document.write_text("private document body")

    adapter = WhatsAppAdapter.__new__(WhatsAppAdapter)
    adapter.platform = Platform.WHATSAPP
    adapter._should_process_message = MagicMock(return_value=True)
    adapter.build_source = MagicMock(return_value=SimpleNamespace())
    monkeypatch.setattr(
        "plugins.platforms.whatsapp.adapter._is_allowed_bridge_path",
        lambda _path: True,
    )

    event = await adapter._build_message_event(
        {
            "chatId": "15551234567",
            "senderId": "15551234567",
            "mediaType": "document",
            "hasMedia": True,
            "mime": "text/plain",
            "mediaUrls": [str(document)],
            "body": "",
        }
    )

    assert event is not None
    assert event.media_urls == [str(document)]
    assert "private document body" in event.text
    output = capsys.readouterr().out
    assert private_name not in output
    assert str(document) not in output
