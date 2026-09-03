from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import MessageType
from plugins.platforms.photon.adapter import PhotonAdapter, _format_location_content


def test_format_location_content_is_model_readable():
    text = _format_location_content(
        {
            "latitude": 29.7,
            "longitude": -95.4,
            "accuracy": 8,
            "shortAddress": "Houston",
        }
    )
    assert text == "[Location shared] Houston (29.700000, -95.400000; accuracy ~8 m)"


@pytest.mark.asyncio
async def test_dispatch_findmy_location_as_native_location():
    adapter = PhotonAdapter(PlatformConfig(enabled=True, extra={}))
    adapter.handle_message = AsyncMock()
    event = {
        "messageId": "location-1",
        "space": {"id": "any;-;+15551234567", "type": "dm", "phone": "shared"},
        "sender": {"id": "+15551234567"},
        "content": {
            "type": "location",
            "latitude": 29.7,
            "longitude": -95.4,
            "accuracy": 8,
            "shortAddress": "Houston",
        },
        "timestamp": "2026-09-03T20:27:30Z",
    }

    await adapter._dispatch_inbound(event)

    adapter.handle_message.assert_awaited_once()
    message = adapter.handle_message.await_args.args[0]
    assert message.message_type is MessageType.LOCATION
    assert message.message_id == "location-1"
    assert "Houston" in message.text
    assert "29.700000" in message.text
    assert "-95.400000" in message.text
