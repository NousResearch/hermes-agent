"""The authenticated CDN proxy must not follow a redirect outside its allowlist."""
import httpx
import pytest
from fastapi import HTTPException

from hermes_cli.web_routers import files


@pytest.mark.asyncio
async def test_media_proxy_refuses_redirect_before_contacting_other_host(monkeypatch):
    contacted = []

    def respond(request):
        contacted.append(request.url.host)
        if request.url.host == "fal.media":
            return httpx.Response(302, headers={"location": "http://127.0.0.1/internal.png"})
        return httpx.Response(200, headers={"content-type": "image/png"}, content=b"private")

    original = httpx.AsyncClient

    def client(**kwargs):
        return original(transport=httpx.MockTransport(respond), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", client)
    monkeypatch.setattr(files, "_require_token", lambda request: None)
    with pytest.raises(HTTPException) as error:
        await files.proxy_remote_media("https://fal.media/image.png", object())
    assert error.value.status_code == 403
    assert contacted == ["fal.media"]


@pytest.mark.asyncio
async def test_media_proxy_keeps_allowlisted_redirects_working(monkeypatch):
    def respond(request):
        if request.url.host == "fal.media":
            return httpx.Response(302, headers={"location": "https://v3.fal.media/image.png"})
        return httpx.Response(200, headers={"content-type": "image/png"}, content=b"image")

    original = httpx.AsyncClient
    monkeypatch.setattr(
        httpx, "AsyncClient",
        lambda **kwargs: original(transport=httpx.MockTransport(respond), **kwargs),
    )
    monkeypatch.setattr(files, "_require_token", lambda request: None)
    result = await files.proxy_remote_media("https://fal.media/image.png", object())
    assert result == {"data_url": "data:image/png;base64,aW1hZ2U="}
