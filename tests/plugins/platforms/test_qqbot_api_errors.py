"""QQ business rejection must not turn HTTP success into delivered messages."""

import asyncio
from unittest.mock import AsyncMock

import httpx
import pytest

from gateway.config import PlatformConfig
from plugins.platforms.qqbot.adapter import QQAdapter
from plugins.platforms.qqbot.send import _qqbot_send_text_message, _qqbot_send_media_message


def response(payload, status=200):
    return httpx.Response(status, json=payload, request=httpx.Request("POST", "https://example.test"))


@pytest.mark.parametrize("code", [None, 0, "0"])
def test_direct_text_accepts_success_codes_with_real_id(code):
    payload = {"id": "real-message"}
    if code is not None:
        payload["code"] = code
    client = AsyncMock()
    client.post.return_value = response(payload)
    result = asyncio.run(_qqbot_send_text_message(client, {}, "openid", "hello"))
    assert result["success"] is True
    assert result["message_id"] == "real-message"
    assert client.post.await_count == 1


def test_direct_text_business_rejection_is_not_success():
    client = AsyncMock()
    client.post.return_value = response({"code": 40001, "message": "denied"})
    result = asyncio.run(_qqbot_send_text_message(client, {}, "openid", "hello"))
    assert result.get("success") is not True
    assert "40001" in result["error"]
    assert not result.get("message_id")


def test_media_message_rejects_success_status_business_error():
    client = AsyncMock()
    client.request.return_value = response({"code": 40001, "message": "denied"})
    with pytest.raises(RuntimeError, match="40001"):
        asyncio.run(_qqbot_send_media_message(client, {}, "c2c", "openid", "file-info"))


def test_standalone_http_error_preserves_raw_diagnostic():
    from plugins.platforms.qqbot.send import _qqbot_api_json

    client = AsyncMock()
    client.request.return_value = httpx.Response(500, text="upstream unavailable")
    with pytest.raises(RuntimeError, match="upstream unavailable"):
        asyncio.run(_qqbot_api_json(client, {}, "POST", "/files", {}))


def test_standalone_uploads_reject_business_errors(tmp_path):
    from plugins.platforms.qqbot.send import _qqbot_upload_local_file, _qqbot_upload_url

    client = AsyncMock()
    client.request.return_value = response({"code": 40001, "message": "denied"})
    with pytest.raises(RuntimeError, match="40001"):
        asyncio.run(_qqbot_upload_url(client, {}, "c2c", "openid", "https://example.test/image.png", 1))
    media = tmp_path / "image.png"
    media.write_bytes(b"image")
    with pytest.raises(RuntimeError, match="40001"):
        asyncio.run(_qqbot_upload_local_file(client, {}, "c2c", "openid", str(media), 1))


def test_connected_send_does_not_retry_business_rejection(monkeypatch):
    adapter = QQAdapter(PlatformConfig(enabled=True, extra={"app_id": "app", "client_secret": "secret"}))
    adapter._http_client = AsyncMock()
    adapter._http_client.request.return_value = response({"code": 40001, "message": "denied"})
    adapter._auth_headers = AsyncMock(return_value={})
    adapter._ensure_connected = AsyncMock(return_value=True)
    # Test the real sender and HTTP response path, not a mocked send result.
    monkeypatch.setattr("plugins.platforms.qqbot.adapter.asyncio.sleep", AsyncMock())
    result = asyncio.run(adapter.send("openid", "hello"))
    assert result.success is False
    assert result.retryable is False
    assert not result.message_id
    assert adapter._http_client.request.await_count == 1


@pytest.mark.parametrize("code", [None, 0, 12345, 40001])
def test_http_server_error_preserves_connected_send_and_upload_retries(code, monkeypatch):
    payload = {"code": code, "message": "temporary backend failure"}
    adapter = QQAdapter(PlatformConfig(enabled=True))
    adapter._http_client = AsyncMock()
    adapter._http_client.request.return_value = response(payload, 500)
    adapter._auth_headers = AsyncMock(return_value={})
    adapter._ensure_connected = AsyncMock(return_value=True)
    monkeypatch.setattr("plugins.platforms.qqbot.adapter.asyncio.sleep", AsyncMock())
    result = asyncio.run(adapter.send("openid", "hello"))
    assert result.success is False
    assert result.retryable is True
    assert adapter._http_client.request.await_count == 3
    adapter._http_client.request.reset_mock()
    with pytest.raises(RuntimeError, match="500"):
        asyncio.run(adapter._upload_media("c2c", "openid", 1, url="https://example.test/image.png"))
    assert adapter._http_client.request.await_count == 3


def test_connected_upload_business_rejection_is_not_retried(monkeypatch):
    adapter = QQAdapter(PlatformConfig(enabled=True))
    adapter._http_client = AsyncMock()
    adapter._http_client.request.return_value = response({"code": 40001, "message": "denied"})
    adapter._auth_headers = AsyncMock(return_value={})
    monkeypatch.setattr("plugins.platforms.qqbot.adapter.asyncio.sleep", AsyncMock())
    with pytest.raises(RuntimeError, match="40001"):
        asyncio.run(adapter._upload_media("c2c", "openid", 1, url="https://example.test/image.png"))
    assert adapter._http_client.request.await_count == 1


@pytest.mark.parametrize("code", [None, 0, "0"])
def test_connected_api_accepts_success_codes(code):
    payload = {"id": "real-message", "code": code}
    adapter = QQAdapter(PlatformConfig(enabled=True))
    adapter._http_client = AsyncMock()
    adapter._http_client.request.return_value = response(payload)
    adapter._auth_headers = AsyncMock(return_value={})
    assert asyncio.run(adapter._api_request("POST", "/v2/users/openid/messages", {})) == payload
