from types import SimpleNamespace

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.homeassistant import adapter as ha


class _BodyStream:
    def __init__(self, response):
        self._response = response
        self._offset = 0
        self.bytes_read = 0
        self.read_sizes = []

    async def read(self, size):
        self.read_sizes.append(size)
        body = self._response.body
        if isinstance(body, str):
            body = body.encode("utf-8")
        chunk = body[self._offset : self._offset + size]
        self._offset += len(chunk)
        self.bytes_read += len(chunk)
        return chunk

    async def iter_chunked(self, size):
        while chunk := await self.read(size):
            yield chunk


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


def _assert_bounded_read(response):
    assert response.content.bytes_read == ha.HA_ERROR_BODY_MAX_BYTES + 1
    assert response.content.read_sizes == [ha.HA_ERROR_BODY_MAX_BYTES + 1]
    assert response.closed is True
    assert response.text_called is False


@pytest.mark.parametrize("use_existing_session", [True, False])
@pytest.mark.asyncio
async def test_send_bounds_homeassistant_notification_error_response(
    monkeypatch, use_existing_session
):
    response = _Response(body=b"x" * (ha.HA_ERROR_BODY_MAX_BYTES + 4096))
    adapter = ha.HomeAssistantAdapter(PlatformConfig(enabled=True, token="token"))
    if use_existing_session:
        adapter._rest_session = _Session(response)
    else:
        monkeypatch.setattr(
            ha.aiohttp,
            "ClientSession",
            lambda *_args, **_kwargs: _Session(response),
        )

    result = await adapter.send("ha_events", "hello")

    assert (
        f"Home Assistant notification error response exceeded {ha.HA_ERROR_BODY_MAX_BYTES} bytes"
        in (result.error or "")
    )
    _assert_bounded_read(response)


@pytest.mark.asyncio
async def test_send_falls_back_from_unknown_response_charset():
    response = _Response(body=b"diagnostic")
    response.charset = "not-a-codec"
    adapter = ha.HomeAssistantAdapter(PlatformConfig(enabled=True, token="token"))
    adapter._rest_session = _Session(response)

    result = await adapter.send("ha_events", "hello")

    assert result.error == "HTTP 500: diagnostic"


@pytest.mark.asyncio
async def test_standalone_send_bounds_homeassistant_error_response(monkeypatch):
    response = _Response(body=b"x" * (ha.HA_ERROR_BODY_MAX_BYTES + 4096))

    monkeypatch.setattr(
        ha.aiohttp,
        "ClientSession",
        lambda *_args, **_kwargs: _Session(response),
    )

    result = await ha._standalone_send(
        SimpleNamespace(token="token", extra={"url": "http://ha.local"}),
        "ha_events",
        "hello",
    )

    assert (
        f"Home Assistant standalone error response exceeded {ha.HA_ERROR_BODY_MAX_BYTES} bytes"
        in result["error"]
    )
    _assert_bounded_read(response)
