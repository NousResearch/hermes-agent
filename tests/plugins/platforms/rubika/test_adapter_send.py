import pytest
from unittest.mock import AsyncMock

from gateway.config import PlatformConfig
from plugins.platforms.rubika.adapter import RubikaAdapter


def _adapter() -> RubikaAdapter:
    cfg = PlatformConfig()
    cfg.extra = {"token": "TESTTOKEN"}
    return RubikaAdapter(cfg)


@pytest.mark.asyncio
async def test_send_text_success():
    adapter = _adapter()
    adapter._client.call = AsyncMock(return_value={"message_id": "m9"})
    result = await adapter.send(chat_id="c1", content="hello there")
    assert result.success is True
    assert result.message_id == "m9"
    adapter._client.call.assert_awaited_once_with("sendMessage", chat_id="c1", text="hello there")


@pytest.mark.asyncio
async def test_send_text_with_inline_keypad():
    adapter = _adapter()
    adapter._client.call = AsyncMock(return_value={"message_id": "m10"})
    result = await adapter.send(
        chat_id="c1", content="pick one",
        metadata={"inline_keypad": [("yes", "Yes"), ("no", "No")]})
    assert result.success is True
    call_kwargs = adapter._client.call.call_args.kwargs
    assert call_kwargs["inline_keypad"] == {
        "rows": [
            {"buttons": [{"id": "yes", "type": "Simple", "button_text": "Yes"}]},
            {"buttons": [{"id": "no", "type": "Simple", "button_text": "No"}]},
        ]
    }


@pytest.mark.asyncio
async def test_send_text_api_error_returns_failure_result():
    from plugins.platforms.rubika.client import RubikaAPIError
    adapter = _adapter()
    adapter._client.call = AsyncMock(side_effect=RubikaAPIError("boom", status="BAD"))
    result = await adapter.send(chat_id="c1", content="hello")
    assert result.success is False
    assert "boom" in result.error
