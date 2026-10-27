"""``/api/media/proxy`` re-checks the host allowlist on every redirect hop."""

import httpx
import pytest
from fastapi import HTTPException

from hermes_cli.web_routers import files

_PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 8


def _route_through(monkeypatch, handler):
    """Serve every outbound request of the route from ``handler`` (real httpx client)."""
    real_client = httpx.AsyncClient

    class _Client(real_client):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    monkeypatch.setattr(files, "_require_token", lambda request: None)


def _image(request):
    return httpx.Response(200, headers={"content-type": "image/png"}, content=_PNG)


@pytest.mark.asyncio
async def test_redirect_off_the_allowlist_is_not_followed(monkeypatch):
    fetched = []

    def handler(request):
        fetched.append(request.url.host)
        if request.url.host == "v3.fal.media":
            return httpx.Response(302, headers={"location": "http://169.254.169.254/latest/x.png"})
        return _image(request)

    _route_through(monkeypatch, handler)
    with pytest.raises(HTTPException) as exc:
        await files.proxy_remote_media("https://v3.fal.media/media/abc", request=None)
    assert exc.value.status_code == 403
    assert fetched == ["v3.fal.media"]


@pytest.mark.asyncio
async def test_redirect_between_allowlisted_hosts_is_followed(monkeypatch):
    def handler(request):
        if request.url.host == "fal.run":
            return httpx.Response(302, headers={"location": "https://v3.fal.media/media/abc"})
        return _image(request)

    _route_through(monkeypatch, handler)
    result = await files.proxy_remote_media("https://fal.run/files/abc", request=None)
    assert result["data_url"].startswith("data:image/png;base64,")
