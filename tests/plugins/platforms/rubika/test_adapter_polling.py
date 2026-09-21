import asyncio
import pytest
from unittest.mock import AsyncMock, patch

from gateway.config import Platform, PlatformConfig
from plugins.platforms.rubika.adapter import RubikaAdapter


def _config(token="TESTTOKEN") -> PlatformConfig:
    cfg = PlatformConfig()
    cfg.extra = {"token": token}
    return cfg


@pytest.mark.asyncio
async def test_connect_marks_connected_and_starts_poll_task():
    adapter = RubikaAdapter(_config())
    with patch.object(adapter, "_poll_loop", AsyncMock()):
        ok = await adapter.connect()
    assert ok is True
    assert adapter._running is True
    await adapter.disconnect()


@pytest.mark.asyncio
async def test_poll_loop_dispatches_new_message_to_handle_message():
    adapter = RubikaAdapter(_config())
    adapter._running = True
    updates_payload = {
        "updates": [{
            "type": "NewMessage", "chat_id": "c1", "chat_type": "User",
            "new_message": {"message_id": "m1", "text": "hi", "sender_id": "u1",
                            "reply_to_message_id": None, "aux_data": None},
        }],
        "next_offset_id": "off-2",
    }
    call_mock = AsyncMock(side_effect=[updates_payload, asyncio.CancelledError()])
    adapter._client.call = call_mock
    with patch.object(adapter, "handle_message", AsyncMock()) as mock_handle:
        with pytest.raises(asyncio.CancelledError):
            await adapter._poll_loop()
    mock_handle.assert_awaited_once()
    sent_event = mock_handle.call_args.args[0]
    assert sent_event.text == "hi"
    assert sent_event.source.chat_id == "c1"
    assert adapter._offset_id == "off-2"
