"""Standalone delivery retries only the failed text or attachment request."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock

import httpx
import pytest

from gateway.config import PlatformConfig
from plugins.platforms.photon import adapter


@pytest.mark.asyncio
@pytest.mark.parametrize("attachment", [False, True], ids=["text", "attachment"])
@pytest.mark.parametrize(
    "responses, expected_error, expected_requests, expected_sleeps",
    [
        ([httpx.Response(429, json={"error": "overflow", "retryable": True}),
          httpx.Response(200, json={"ok": True, "messageId": "delivered"})],
         None, 2, 1),
        ([httpx.Response(500, json={"error": "busy", "retryable": True})] * 2,
         "sidecar_error", 2, 1),
        ([httpx.Response(200, json={"ok": False, "error": "invalid credentials",
                                    "error_class": "auth_or_config", "retryable": False})],
         "auth_or_config", 1, 0),
        ([httpx.Response(403, json={"error": "private upstream detail",
                                    "error_class": "target_not_allowed", "retryable": True})],
         "target_not_allowed", 1, 0),
        ([httpx.Response(200, content=b'{"ok":')], "sidecar_error", 1, 0),
        ([httpx.Response(200, json=["unexpected"])], "sidecar_error", 1, 0),
    ],
    ids=["recovers", "exhausted", "permanent", "target-denied", "malformed-json", "non-object-json"],
)
async def test_standalone_retry_contract(
    monkeypatch, tmp_path, attachment, responses, expected_error,
    expected_requests, expected_sleeps,
):
    monkeypatch.setenv("PHOTON_SIDECAR_TOKEN", "test-sidecar-token")
    sleep = AsyncMock()
    monkeypatch.setattr(adapter.asyncio, "sleep", sleep)
    requests = []
    pending = iter(responses)

    def handle(request):
        requests.append(request)
        # A successful text must not be resent when its attachment fails.
        if attachment and request.url.path == "/send":
            return httpx.Response(200, json={"ok": True, "messageId": "text-id"})
        return next(pending)

    client_type = httpx.AsyncClient
    monkeypatch.setattr(
        adapter.httpx, "AsyncClient",
        lambda **kwargs: client_type(transport=httpx.MockTransport(handle), **kwargs),
    )
    media_files = None
    if attachment:
        image = tmp_path / "photo.png"
        image.write_bytes(b"test-image")
        media_files = [(str(image), False)]

    result = await adapter._standalone_send(
        PlatformConfig(enabled=True, token="", extra={}), "test-chat", "hello",
        media_files=media_files,
    )

    endpoint = "/send-attachment" if attachment else "/send"
    retried = [request for request in requests if request.url.path == endpoint]
    assert len(retried) == expected_requests
    assert len(requests) == expected_requests + int(attachment)
    assert all(request.content == retried[0].content for request in retried)
    assert all(request.headers["X-Hermes-Sidecar-Token"] == "test-sidecar-token"
               for request in requests)
    body = json.loads(retried[0].content)
    assert body["spaceId"] == "test-chat"
    if attachment:
        assert body["path"] == str(image.resolve())
        assert body["kind"] == "attachment"
        assert body["mimeType"] == "image/png"
    else:
        assert body["text"] == "hello"
    assert sleep.await_count == expected_sleeps
    if expected_sleeps:
        sleep.assert_awaited_once_with(2.0)
    if expected_error:
        assert result["error_class"] == expected_error
        assert result["error"]
        assert not result.get("success")
        if expected_error == "target_not_allowed":
            assert result["retryable"] is False
            assert "private upstream detail" not in result["error"]
    else:
        assert result == {"success": True, "message_id": "delivered"}
