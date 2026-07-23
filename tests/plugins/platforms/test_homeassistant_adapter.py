from types import SimpleNamespace

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.homeassistant import adapter as ha


class _BodyStream:
    def __init__(self, response):
        self._response = response

    async def iter_chunked(self, size):
        body = self._response.body
        if isinstance(body, str):
            body = body.encode("utf-8")
        for idx in range(0, len(body), size):
            yield body[idx:idx + size]


class _Response:
    def __init__(self, *, status=500, body=b""):
        self.status = status
        self.body = body
        self.charset = "utf-8"
        self.content = _BodyStream(self)
        self.closed = False
        self.text_called = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False

    async def text(self):
        self.text_called = True
        return self.body.decode("utf-8") if isinstance(self.body, bytes) else self.body

    def close(self):
        self.closed = True


class _Session:
    def __init__(self, response=None, *args, **kwargs):
        self.response = response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False

    def post(self, *args, **kwargs):
        return self.response


@pytest.mark.asyncio
async def test_send_bounds_homeassistant_notification_error_response():
    response = _Response(body=b"x" * (ha.HA_ERROR_BODY_MAX_BYTES + 1))
    adapter = ha.HomeAssistantAdapter(
        PlatformConfig(enabled=True, token="token", extra={"url": "http://ha.local"})
    )
    adapter._rest_session = _Session(response)

    result = await adapter.send("ha_events", "hello")

    assert result.success is False
    assert (
        f"Home Assistant notification error response exceeded {ha.HA_ERROR_BODY_MAX_BYTES} bytes"
        in (result.error or "")
    )
    assert response.closed is True
    assert response.text_called is False


@pytest.mark.asyncio
async def test_standalone_send_bounds_homeassistant_error_response(monkeypatch):
    response = _Response(body=b"x" * (ha.HA_ERROR_BODY_MAX_BYTES + 1))

    class _ClientSession(_Session):
        def __init__(self, *args, **kwargs):
            super().__init__(response)

    monkeypatch.setattr(ha.aiohttp, "ClientSession", _ClientSession)

    result = await ha._standalone_send(
        SimpleNamespace(token="token", extra={"url": "http://ha.local"}),
        "ha_events",
        "hello",
    )

    assert (
        f"Home Assistant standalone error response exceeded {ha.HA_ERROR_BODY_MAX_BYTES} bytes"
        in result["error"]
    )
    assert response.closed is True
    assert response.text_called is False
