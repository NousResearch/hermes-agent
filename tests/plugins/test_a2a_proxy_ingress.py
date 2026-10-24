"""A2A behind a reverse proxy: localhost-only trust never extends to proxied requests, and an
IPv6 bind (``A2A_HOST=::``) serves both families so IPv6-only private networks can reach it."""

from __future__ import annotations

import asyncio
import json
import socket
import urllib.error
import urllib.request

import pytest

from plugins.platforms.a2a import security


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _live_adapter(monkeypatch, **env):
    from gateway.config import PlatformConfig
    from plugins.platforms.a2a.adapter import A2AAdapter

    for var in ("A2A_BEARER_TOKEN", "A2A_PEER_TOKENS", "A2A_HOST", "A2A_TRUSTED_PEERS", "A2A_ALLOW_ALL_USERS"):
        monkeypatch.delenv(var, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    port = _free_port()
    monkeypatch.setenv("A2A_PORT", str(port))
    return A2AAdapter(PlatformConfig(enabled=True, extra={"port": port})), port


def _post(url: str, headers: dict) -> int:
    body = {"jsonrpc": "2.0", "id": "1", "method": "tasks/list", "params": {}}
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code


def _get_json(url: str, headers: dict | None = None) -> dict:
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers or {}), timeout=10) as r:
        return json.loads(r.read().decode())


def test_localhost_only_refuses_forwarded_identity(monkeypatch):
    monkeypatch.delenv("A2A_BEARER_TOKEN", raising=False)
    monkeypatch.delenv("A2A_PEER_TOKENS", raising=False)
    ctx = security.A2ASecurityContext.capture()
    assert ctx.authenticate(None, "127.0.0.1") == "ip:127.0.0.1"
    assert ctx.authenticate("Bearer anything", "127.0.0.1", forwarded=True) is None


def test_token_identity_unaffected_by_forwarding(monkeypatch):
    monkeypatch.setenv("A2A_PEER_TOKENS", "alice:tok-a")
    ctx = security.A2ASecurityContext.capture()
    assert ctx.authenticate("Bearer tok-a", "10.0.0.1", forwarded=True) == "alice"


@pytest.mark.integration
def test_proxied_request_to_localhost_only_server_is_unauthorized(monkeypatch):
    """A same-host reverse proxy must not turn the no-token loopback server into an open endpoint."""
    adapter, port = _live_adapter(monkeypatch)
    base = f"http://127.0.0.1:{port}/"

    async def run():
        assert await adapter.connect() is True
        try:
            assert await asyncio.to_thread(_post, base, {}) == 200
            for header in ("X-Forwarded-For", "Forwarded", "X-Real-IP"):
                assert await asyncio.to_thread(_post, base, {header: "203.0.113.7"}) == 401
            proxied = await asyncio.to_thread(_get_json, base, {"X-Forwarded-For": "203.0.113.7"})
            assert "served_agents" not in proxied
        finally:
            await adapter.disconnect()

    asyncio.run(run())


@pytest.mark.integration
def test_ipv6_any_bind_serves_both_families(monkeypatch):
    adapter, port = _live_adapter(monkeypatch, A2A_PEER_TOKENS="peer:tok", A2A_TRUSTED_PEERS="peer", A2A_HOST="::")

    async def run():
        assert await adapter.connect() is True
        try:
            for host in ("[::1]", "127.0.0.1"):
                card = await asyncio.to_thread(_get_json, f"http://{host}:{port}/.well-known/agent-card.json")
                assert card["name"]
                assert await asyncio.to_thread(_post, f"http://{host}:{port}/", {"Authorization": "Bearer tok"}) == 200
                assert await asyncio.to_thread(_post, f"http://{host}:{port}/", {}) == 401
        finally:
            await adapter.disconnect()

    asyncio.run(run())


@pytest.mark.integration
def test_card_advertises_the_proxy_prefix(monkeypatch):
    adapter, port = _live_adapter(monkeypatch)
    headers = {"X-Forwarded-Host": "agent.example.com", "X-Forwarded-Proto": "https", "X-Forwarded-Prefix": "/a2a"}

    async def run():
        assert await adapter.connect() is True
        try:
            card = await asyncio.to_thread(_get_json, f"http://127.0.0.1:{port}/.well-known/agent-card.json", headers)
            assert card["supportedInterfaces"][0]["url"] == "https://agent.example.com/a2a/"
        finally:
            await adapter.disconnect()

    asyncio.run(run())
