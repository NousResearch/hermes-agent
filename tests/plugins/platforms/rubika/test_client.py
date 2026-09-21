import httpx
import pytest
from unittest.mock import AsyncMock, Mock, patch
import tempfile
import os

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


@pytest.mark.asyncio
async def test_call_raises_rubikaapierror_on_http_error():
    client = RubikaClient(token="TESTTOKEN")
    mock_response = AsyncMock()
    mock_response.status_code = 429
    # Mock raise_for_status to raise an HTTPStatusError (synchronously)
    http_error = httpx.HTTPStatusError("429 Too Many Requests", request=AsyncMock(), response=mock_response)
    mock_response.raise_for_status = Mock(side_effect=http_error)
    with patch.object(httpx.AsyncClient, "post", AsyncMock(return_value=mock_response)):
        with pytest.raises(RubikaAPIError) as exc_info:
            await client.call("sendMessage", chat_id="c1", text="hi")
    assert exc_info.value.status == "429"
    assert "HTTP error" in str(exc_info.value)


@pytest.mark.asyncio
async def test_upload_file_returns_file_id():
    client = RubikaClient(token="TESTTOKEN")
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(b"fake-image-bytes")
        tmp_path = f.name
    try:
        request_response = AsyncMock()
        request_response.json = lambda: {
            "status": "OK", "data": {"upload_url": "https://upload.rubika.ir/xyz"}}
        request_response.raise_for_status = lambda: None

        upload_response = AsyncMock()
        upload_response.json = lambda: {"status": "OK", "data": {"file_id": "file123"}}
        upload_response.raise_for_status = lambda: None

        with patch.object(
            httpx.AsyncClient, "post",
            AsyncMock(side_effect=[request_response, upload_response]),
        ) as mock_post:
            file_id = await client.upload_file(tmp_path, file_type="Image")
        assert file_id == "file123"
        assert mock_post.await_count == 2
        first_call, second_call = mock_post.call_args_list
        assert first_call.args[0] == "https://botapi.rubika.ir/v3/TESTTOKEN/requestSendFile"
        assert first_call.kwargs["json"] == {"type": "Image"}
        assert second_call.args[0] == "https://upload.rubika.ir/xyz"
    finally:
        os.unlink(tmp_path)


@pytest.mark.asyncio
async def test_upload_file_raises_on_missing_file():
    """Verify that missing files raise RubikaAPIError (not FileNotFoundError)
    and no HTTP calls are made (file check happens first)."""
    client = RubikaClient(token="TESTTOKEN")
    with patch.object(httpx.AsyncClient, "post", AsyncMock()) as mock_post:
        with pytest.raises(RubikaAPIError) as exc_info:
            await client.upload_file("/nonexistent/path/file.png", file_type="Image")
    assert exc_info.value.status == "FILE_READ_ERROR"
    assert "Could not read file" in str(exc_info.value)
    # Verify no HTTP calls were made (file check happens before API call)
    mock_post.assert_not_awaited()
