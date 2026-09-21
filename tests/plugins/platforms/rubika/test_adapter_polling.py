import asyncio
import pytest
from unittest.mock import AsyncMock, patch

from gateway.config import Platform, PlatformConfig
from plugins.platforms.rubika.adapter import RubikaAdapter, POLL_INTERVAL_SECONDS


def _config(token="TESTTOKEN") -> PlatformConfig:
    cfg = PlatformConfig()
    cfg.extra = {"token": token}
    return cfg


@pytest.mark.asyncio
async def test_connect_marks_connected_and_starts_poll_task():
    adapter = RubikaAdapter(_config())
    # side_effect (not return_value) makes the mocked poll loop a real, still-running coroutine
    # that only completes when cancelled — so disconnect()'s cancel-a-live-task branch is
    # genuinely exercised instead of racing a task that already finished on its own.
    async def _hang_until_cancelled(*_args, **_kwargs):
        await asyncio.sleep(3600)

    with patch.object(adapter, "_poll_loop", AsyncMock(side_effect=_hang_until_cancelled)):
        ok = await adapter.connect()
        assert ok is True
        assert adapter._running is True
        assert not adapter._poll_task.done()
        await adapter.disconnect()
    assert adapter._poll_task is None


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
    with patch.object(adapter, "handle_message", AsyncMock()) as mock_handle, \
         patch("plugins.platforms.rubika.adapter.asyncio.sleep", AsyncMock()) as mock_sleep:
        with pytest.raises(asyncio.CancelledError):
            await adapter._poll_loop()
    mock_handle.assert_awaited_once()
    sent_event = mock_handle.call_args.args[0]
    assert sent_event.text == "hi"
    assert sent_event.source.chat_id == "c1"
    assert adapter._offset_id == "off-2"
    # The inter-iteration delay actually ran (not just present in source) — proves the loop
    # doesn't spin at zero delay between successful getUpdates calls.
    mock_sleep.assert_awaited_once_with(POLL_INTERVAL_SECONDS)
