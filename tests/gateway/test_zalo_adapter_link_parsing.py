import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import MessageType
from tests.gateway._plugin_adapter_loader import load_plugin_adapter


_zalo = load_plugin_adapter("zalo")
EVENT_UNSUPPORTED = _zalo.EVENT_UNSUPPORTED
ZaloAdapter = _zalo.ZaloAdapter


@pytest.mark.asyncio
async def test_zalo_unsupported_event_extracts_nested_link_attachment():
    adapter = ZaloAdapter(PlatformConfig(enabled=True))
    update = {
        "event_name": EVENT_UNSUPPORTED,
        "message": {
            "chat": {"id": "chat-1"},
            "sender": {"id": "user-1"},
            "attachments": [
                {
                    "type": "link",
                    "payload": {
                        "title": "Rental sheet",
                        "description": "Google Sheets",
                        "url": "https://docs.google.com/spreadsheets/d/sheet-id/edit?usp=drivesdk",
                    },
                }
            ],
        },
    }

    text, message_type, media_urls, media_types = await adapter._event_content(
        update,
        update["message"],
    )

    assert message_type is MessageType.TEXT
    assert media_urls == []
    assert media_types == []
    assert "Rental sheet" in text
    assert "Google Sheets" in text
    assert "https://docs.google.com/spreadsheets/d/sheet-id/edit?usp=drivesdk" in text
    assert "content unavailable" not in text
