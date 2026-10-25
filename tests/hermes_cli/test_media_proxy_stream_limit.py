"""The proxy's byte cap must apply while receiving, not after buffering the body."""
import httpx
import pytest
from fastapi import HTTPException

from hermes_cli.web_routers import files


@pytest.mark.asyncio
async def test_media_proxy_stops_reading_after_size_limit(monkeypatch):
    yielded = []

    class Body(httpx.AsyncByteStream):
        async def __aiter__(self):
            for i in range(20):
                yielded.append(i)
                yield b"x" * 8

    def respond(request):
        return httpx.Response(200, headers={"content-type": "image/png"}, stream=Body())

    original = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient",
                        lambda **kwargs: original(transport=httpx.MockTransport(respond), **kwargs))
    monkeypatch.setattr(files, "_require_token", lambda request: None)
    monkeypatch.setattr(files, "_MEDIA_MAX_BYTES", 10)
    with pytest.raises(HTTPException) as error:
        await files.proxy_remote_media("https://fal.media/image.png", object())
    assert error.value.status_code == 413
    assert len(yielded) <= 2


@pytest.mark.asyncio
async def test_media_proxy_accepts_body_exactly_at_limit(monkeypatch):
    def respond(request):
        return httpx.Response(200, headers={"content-type": "image/png"}, content=b"image")

    original = httpx.AsyncClient
    monkeypatch.setattr(
        httpx, "AsyncClient",
        lambda **kwargs: original(transport=httpx.MockTransport(respond), **kwargs),
    )
    monkeypatch.setattr(files, "_require_token", lambda request: None)
    monkeypatch.setattr(files, "_MEDIA_MAX_BYTES", 5)
    result = await files.proxy_remote_media("https://fal.media/image.png", object())
    assert result == {"data_url": "data:image/png;base64,aW1hZ2U="}
