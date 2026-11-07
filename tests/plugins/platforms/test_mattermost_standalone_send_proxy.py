"""``hermes send --to mattermost`` must honour NO_PROXY for an internal server.

The plugin's standalone send path (out-of-process cron delivery, ``hermes send``) builds its own
``aiohttp`` session instead of reusing the live adapter's ``trust_env`` session, and it resolved the
proxy with ``resolve_proxy_url(platform_env_var="MATTERMOST_PROXY")`` — no ``target_hosts``. With
``None`` the resolver's ``should_bypass_proxy()`` can never match, so a self-hosted Mattermost on a
NO_PROXY-covered address was pushed through the egress proxy and answered ``403 ERR_ACCESS_DENIED``
while the live gateway delivered the same channel fine. Telegram already passes its API hosts; the
REST host we are about to post to is the same information.
"""

from __future__ import annotations

import asyncio
import sys
import types

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.mattermost.adapter import _standalone_send

_SQUID = "http://squid.example:3128"
_INTERNAL_HOST = "mm.internal.example"
_INTERNAL_URL = f"http://{_INTERNAL_HOST}:8065"
_PUBLIC_URL = "https://mm.example.com"


class _FakeResponse:
    status = 201

    async def json(self):
        return {"id": "post-1"}

    async def text(self):
        return ""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeSession:
    """Records the kwargs the adapter passes to ``aiohttp.ClientSession`` / ``session.post()``."""

    def __init__(self, captured, **kwargs):
        self._captured = captured
        captured["session_kwargs"] = kwargs

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def post(self, url, **kwargs):
        self._captured.setdefault("post_kwargs", []).append(kwargs)
        return _FakeResponse()


@pytest.fixture
def proxy_env(monkeypatch):
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy",
                "NO_PROXY", "no_proxy", "MATTERMOST_PROXY", "mattermost_proxy"):
        monkeypatch.delenv(key, raising=False)
    for key in ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy"):
        monkeypatch.setenv(key, _SQUID)


def _install_fake_aiohttp(monkeypatch, captured):
    fake = types.SimpleNamespace(
        ClientSession=lambda **kwargs: _FakeSession(captured, **kwargs),
        ClientTimeout=lambda **kwargs: object(),
        ClientError=Exception,
        FormData=lambda: object(),
    )
    monkeypatch.setitem(sys.modules, "aiohttp", fake)


def _send(monkeypatch, url: str) -> dict:
    """Run the standalone send against a fake transport; return what reached the network layer."""
    captured: dict = {}
    _install_fake_aiohttp(monkeypatch, captured)
    pconfig = PlatformConfig(enabled=True, token="tok", extra={"url": url})
    result = asyncio.run(_standalone_send(pconfig, "chan-1", "hello"))
    assert result.get("success") is True, result
    assert captured["post_kwargs"], "expected a post through the fake session"
    return captured


def test_no_proxy_covered_server_is_not_pushed_through_the_proxy(proxy_env, monkeypatch):
    monkeypatch.setenv("NO_PROXY", f"localhost,127.0.0.1,{_INTERNAL_HOST}")
    monkeypatch.setenv("no_proxy", f"localhost,127.0.0.1,{_INTERNAL_HOST}")

    captured = _send(monkeypatch, _INTERNAL_URL)

    for post_kwargs in captured["post_kwargs"]:
        assert "proxy" not in post_kwargs
    assert "connector" not in captured["session_kwargs"]


def test_public_server_still_uses_the_proxy(proxy_env, monkeypatch):
    monkeypatch.setenv("NO_PROXY", "localhost,127.0.0.1")
    monkeypatch.setenv("no_proxy", "localhost,127.0.0.1")

    captured = _send(monkeypatch, _PUBLIC_URL)

    post_kwargs = captured["post_kwargs"][-1]
    assert post_kwargs.get("proxy") == _SQUID or "connector" in captured["session_kwargs"]
