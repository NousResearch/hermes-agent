import httpx
import pytest
from unittest.mock import AsyncMock, patch

from plugins.platforms.rubika.client import RubikaClient, RubikaAPIError


@pytest.mark.asyncio
async def test_call_returns_data_on_ok_status():
    client = RubikaClient(token="TESTTOKEN")
    mock_response = AsyncMock()
    mock_response.json = lambda: {"status": "OK", "data": {"message_id": "42"}}
    mock_response.raise_for_status = lambda: None
    with patch.object(httpx.AsyncClient, "post", AsyncMock(return_value=mock_response)) as mock_post:
        result = await client.call("sendMessage", chat_id="c1", text="hi")
    assert result == {"message_id": "42"}
    mock_post.assert_awaited_once()
    call_args = mock_post.call_args
    assert call_args.args[0] == "https://botapi.rubika.ir/v3/TESTTOKEN/sendMessage"
    assert call_args.kwargs["json"] == {"chat_id": "c1", "text": "hi"}


@pytest.mark.asyncio
async def test_call_raises_on_non_ok_status():
    client = RubikaClient(token="TESTTOKEN")
    mock_response = AsyncMock()
    mock_response.json = lambda: {"status": "INVALID_INPUT", "data": {}}
    mock_response.raise_for_status = lambda: None
    with patch.object(httpx.AsyncClient, "post", AsyncMock(return_value=mock_response)):
        with pytest.raises(RubikaAPIError) as exc_info:
            await client.call("sendMessage", chat_id="c1", text="hi")
    assert exc_info.value.status == "INVALID_INPUT"
