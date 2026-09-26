import pytest
from unittest.mock import AsyncMock

from gateway.config import PlatformConfig
from plugins.platforms.rubika.adapter import RubikaAdapter


def _adapter() -> RubikaAdapter:
    cfg = PlatformConfig()
    cfg.extra = {"token": "TESTTOKEN"}
    return RubikaAdapter(cfg)


@pytest.mark.asyncio
async def test_edit_message_success():
    adapter = _adapter()
    adapter._client.call = AsyncMock(return_value={})
    result = await adapter.edit_message(chat_id="c1", message_id="m1", content="updated text")
    assert result.success is True
    assert result.message_id == "m1"
    adapter._client.call.assert_awaited_once_with(
        "editMessageText", chat_id="c1", message_id="m1", text="updated text")


@pytest.mark.asyncio
async def test_delete_message_success():
    adapter = _adapter()
    adapter._client.call = AsyncMock(return_value={})
    ok = await adapter.delete_message(chat_id="c1", message_id="m1")
    assert ok is True
    adapter._client.call.assert_awaited_once_with("deleteMessage", chat_id="c1", message_id="m1")


@pytest.mark.asyncio
async def test_delete_message_failure_returns_false():
    from plugins.platforms.rubika.client import RubikaAPIError
    adapter = _adapter()
    adapter._client.call = AsyncMock(side_effect=RubikaAPIError("gone", status="NOT_FOUND"))
    ok = await adapter.delete_message(chat_id="c1", message_id="m1")
    assert ok is False
