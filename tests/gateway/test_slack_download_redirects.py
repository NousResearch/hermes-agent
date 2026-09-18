"""Authenticated Slack downloads retain the token only on validated CDN hops."""
import asyncio
import socket
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest


if "slack_bolt" not in sys.modules:
    for name in (
        "slack_bolt", "slack_bolt.adapter", "slack_bolt.adapter.socket_mode",
        "slack_bolt.adapter.socket_mode.async_handler", "slack_bolt.async_app",
        "slack_sdk", "slack_sdk.web", "slack_sdk.web.async_client", "slack_sdk.errors",
    ):
        sys.modules.setdefault(name, MagicMock())
if "aiohttp" not in sys.modules:
    sys.modules.setdefault("aiohttp", MagicMock())

from gateway.config import PlatformConfig  # noqa: E402
from plugins.platforms.slack.adapter import SlackAdapter  # noqa: E402


START = "https://files.slack.com/files-pri/TSECOND-F123/image.png"
ORIGIN = "https://files-origin.slack.com/files-pri/TSECOND-F123/image.png"
IMAGE = b"\x89PNG\r\n\x1a\nimage bytes"
TOKEN = "test-download-token"


@pytest.fixture
def adapter(monkeypatch):
    adapter = SlackAdapter.__new__(SlackAdapter)
    adapter.config = PlatformConfig(token="primary-test-token")
    adapter._team_clients = {"TSECOND": SimpleNamespace(token=TOKEN)}
    # Exercise the real URL validators without external DNS or network access.
    monkeypatch.setattr(socket, "getaddrinfo", lambda host, *a, **kw: [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", (
            "169.254.169.254" if host == "169.254.169.254" else "93.184.216.34", 443)),
    ])
    return adapter


@pytest.fixture
def install_transport(monkeypatch):
    def install(handler):
        def client(**kwargs):
            return httpx.AsyncClient(**kwargs, transport=httpx.MockTransport(handler), trust_env=False)
        # MockTransport has no TCP backend to DNS-pin. Replace only the factory;
        # httpx redirect/header handling and the production response hook stay real.
        monkeypatch.setattr("tools.url_safety.create_ssrf_safe_async_client", client)
    return install


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
def test_enterprise_grid_redirect_retains_bearer(adapter, install_transport, status):
    requests = []

    def serve(request):
        requests.append(request)
        if str(request.url) == START:
            return httpx.Response(status, headers={"location": ORIGIN})
        assert str(request.url) == ORIGIN
        if request.headers.get("Authorization") != f"Bearer {TOKEN}":
            return httpx.Response(200, headers={"content-type": "text/html"}, content=b"sign in")
        return httpx.Response(200, headers={"content-type": "image/png"}, content=IMAGE)

    install_transport(serve)
    assert asyncio.run(adapter._download_slack_file_bytes(START)) == IMAGE
    assert [request.headers.get("Authorization") for request in requests] == [
        f"Bearer {TOKEN}", f"Bearer {TOKEN}",
    ]


@pytest.mark.parametrize("target", [
    "https://evil.example.com/file", "https://files.slack.com.evil.example.com/file",
    "http://files-origin.slack.com/file", "https://169.254.169.254/latest/meta-data/",
])
def test_redirect_refuses_untrusted_target(adapter, install_transport, target):
    requests = []

    def serve(request):
        requests.append(request)
        if str(request.url) != START:
            assert "Authorization" not in request.headers, "bearer reached an untrusted target"
            return httpx.Response(200, content=b"not Slack")
        return httpx.Response(302, headers={"location": target})

    install_transport(serve)
    with pytest.raises(ValueError, match="Blocked"):
        asyncio.run(adapter._download_slack_file_bytes(START))
    assert [str(request.url) for request in requests] == [START]


def test_redirect_rejects_slack_host_resolving_private(adapter, install_transport, monkeypatch):
    requests = []

    def resolve(host, *args, **kwargs):
        ip = "10.0.0.1" if host == "files-origin.slack.com" else "93.184.216.34"
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 443))]

    monkeypatch.setattr(socket, "getaddrinfo", resolve)

    def serve(request):
        requests.append(request)
        return httpx.Response(302, headers={"location": ORIGIN})

    install_transport(serve)
    with pytest.raises(ValueError, match="Blocked"):
        asyncio.run(adapter._download_slack_file_bytes(START))
    assert len(requests) == 1


def test_every_redirect_checks_cdn_allowlist(adapter, install_transport):
    requests = []

    def serve(request):
        requests.append(request)
        assert request.url.host in {"files.slack.com", "files-origin.slack.com"}
        target = ORIGIN if str(request.url) == START else "https://evil.example.com/file"
        return httpx.Response(302, headers={"location": target})

    install_transport(serve)
    with pytest.raises(ValueError, match="Blocked non-Slack-CDN"):
        asyncio.run(adapter._download_slack_file_bytes(START))
    assert [str(request.url) for request in requests] == [START, ORIGIN]


def test_relative_redirects_allow_three_hops(adapter, install_transport):
    requests = []

    def serve(request):
        requests.append(request)
        assert request.headers["Authorization"] == f"Bearer {TOKEN}"
        if len(requests) <= 3:
            return httpx.Response(302, headers={"location": f"/hop-{len(requests)}"})
        return httpx.Response(200, content=IMAGE)

    install_transport(serve)
    assert asyncio.run(adapter._download_slack_file_bytes(START)) == IMAGE
    assert [request.url.path for request in requests[1:]] == ["/hop-1", "/hop-2", "/hop-3"]


def test_redirect_loop_is_bounded(adapter, install_transport):
    requests = []

    def serve(request):
        requests.append(request)
        return httpx.Response(302, headers={"location": ORIGIN})

    install_transport(serve)
    with pytest.raises(ValueError, match="Too many Slack file redirects"):
        asyncio.run(adapter._download_slack_file_bytes(START))
    assert len(requests) == 4


@pytest.mark.parametrize("failure", [429, 503, "timeout"])
def test_redirected_download_retries_from_original_url(adapter, install_transport, monkeypatch, failure):
    requests = []
    sleep = AsyncMock()
    monkeypatch.setattr(asyncio, "sleep", sleep)

    def serve(request):
        requests.append(request)
        assert request.headers["Authorization"] == f"Bearer {TOKEN}"
        if str(request.url) == START:
            return httpx.Response(302, headers={"location": ORIGIN})
        if len(requests) < 6:
            if failure == "timeout":
                raise httpx.ReadTimeout("test timeout", request=request)
            return httpx.Response(failure)
        return httpx.Response(200, content=IMAGE)

    install_transport(serve)
    assert asyncio.run(adapter._download_slack_file_bytes(START)) == IMAGE
    assert [str(request.url) for request in requests] == [START, ORIGIN] * 3
    assert [call.args for call in sleep.await_args_list] == [(1.5,), (3.0,)]


def test_html_reject_explains_enterprise_grid_redirect(adapter, install_transport):
    install_transport(lambda request: httpx.Response(
        200, headers={"content-type": "text/html"}, content=b"sign in"))
    with pytest.raises(ValueError, match="cross-origin redirect dropped the token.*Enterprise Grid files-origin"):
        asyncio.run(adapter._download_slack_file_bytes(START))
