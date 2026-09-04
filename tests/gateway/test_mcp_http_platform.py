"""Behaviour contracts for the ``mcp_http`` platform plugin.

Security primitives are exercised directly; the chat/wait_reply loop runs end-to-end over a
real Streamable HTTP MCP client against an ephemeral port with a fake gateway handler that
reproduces the gateway's delivery contract (progress bubbles without ``notify``, final reply
with ``notify=True``).
"""

from __future__ import annotations

import asyncio
import socket
from concurrent.futures import Future

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import ProcessingOutcome
from plugins.platforms.mcp_http import history, security
from plugins.platforms.mcp_http.adapter import McpHttpAdapter, _current_peer

pytestmark = pytest.mark.skipif(
    not __import__("plugins.platforms.mcp_http", fromlist=["check_requirements"]).check_requirements(),
    reason="mcp >= 2.0 SDK / uvicorn not installed",
)


@pytest.fixture(autouse=True)
def _no_tokens(monkeypatch, tmp_path):
    """Tokens are read from the temp HERMES_HOME's .env (redirected by conftest) and the env;
    start every test from a clean slate."""
    for key in ("MCP_HTTP_PEER_TOKENS", "MCP_HTTP_BEARER_TOKEN", "MCP_HTTP_HOST", "MCP_HTTP_PORT",
                "MCP_HTTP_PUBLIC_URL", "MCP_HTTP_TRUSTED_PEERS", "MCP_HTTP_ALLOW_ALL_USERS"):
        monkeypatch.delenv(key, raising=False)


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


# --------------------------------------------------------------------------- security


def test_no_token_forces_loopback_even_when_wider_host_requested(monkeypatch):
    monkeypatch.setenv("MCP_HTTP_HOST", "0.0.0.0")
    assert security.localhost_only() is True
    assert security.Settings.from_extra({}).bind_host() == "127.0.0.1"
    # config.yaml `extra.host` is subject to the same rule.
    assert security.Settings.from_extra({"host": "0.0.0.0"}).bind_host() == "127.0.0.1"
    monkeypatch.setenv("MCP_HTTP_PEER_TOKENS", "alice:tok-alice")
    assert security.Settings.from_extra({}).bind_host() == "0.0.0.0"


def test_authenticate_maps_tokens_to_identity(monkeypatch):
    monkeypatch.setenv("MCP_HTTP_PEER_TOKENS", "alice:tok-alice,bob:tok-bob")
    monkeypatch.setenv("MCP_HTTP_BEARER_TOKEN", "shared-secret")
    assert security.authenticate("Bearer tok-alice", "10.0.0.5") == "alice"
    assert security.authenticate("Bearer shared-secret", "10.0.0.5") == "ip:10.0.0.5"
    assert security.authenticate("Bearer wrong", "10.0.0.5") is None
    assert security.authenticate(None, "10.0.0.5") is None
    assert security.authenticate("Basic tok-alice", "10.0.0.5") is None


def test_env_file_tokens_read_from_hermes_home(tmp_path, monkeypatch):
    """Token rotation in the profile's .env must work without a restart — and the file read
    must follow HERMES_HOME, never the real ~/.hermes."""
    from hermes_constants import get_hermes_home

    (get_hermes_home() / ".env").write_text('MCP_HTTP_PEER_TOKENS="carol:tok-carol"\n', encoding="utf-8")
    assert security.authenticate("Bearer tok-carol", "1.2.3.4") == "carol"
    assert security.localhost_only() is False


def test_transport_security_hosts_derive_only_from_config():
    settings = security.Settings.from_extra({"public_url": "https://hermes.example.net/mcp",
                                             "allowed_hosts": ["tunnel.example.org"]})
    ts = security.transport_security(settings)
    assert ts.enable_dns_rebinding_protection is True
    assert {"127.0.0.1", "localhost", "hermes.example.net", "tunnel.example.org"} <= set(ts.allowed_hosts)
    assert "https://hermes.example.net" in ts.allowed_origins
    # Exactly loopback + configured names — nothing baked in from any particular install.
    expected = {"127.0.0.1", "localhost", "hermes.example.net", "tunnel.example.org"}
    assert {h.split(":")[0] for h in ts.allowed_hosts} == expected


def test_settings_env_overrides_extra(monkeypatch):
    monkeypatch.setenv("MCP_HTTP_PORT", "9111")
    settings = security.Settings.from_extra({"port": 8000, "rate_limit": 5, "trusted_peers": ["alice"]})
    assert settings.port == 9111
    assert settings.rate_limit == 5
    assert settings.trusted_peers == frozenset({"alice"})


# --------------------------------------------------------------------------- adapter


def _adapter(port: int = 0) -> McpHttpAdapter:
    return McpHttpAdapter(PlatformConfig(enabled=True, extra={"port": port or _free_port()}))


def test_cross_peer_conversation_access_refused():
    adapter = _adapter()
    assert adapter._own_conv("alice", "alice") is None
    assert adapter._own_conv("alice", "alice-abc123") is None
    assert adapter._own_conv("alice", "bob-abc123")
    assert adapter._own_conv("alice", "alicex")


def test_success_without_notify_send_resolves_non_empty():
    adapter = _adapter()
    conv = adapter._conv("alice", "alice")
    conv.future = Future()
    conv.started_at = 1.0
    asyncio.run(adapter.send("alice", "🔧 terminal: ls", metadata={}))
    event = type("E", (), {"source": adapter._source_for("alice", "alice")})()
    asyncio.run(adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS))
    reply = adapter._wait_reply("alice", 1)
    assert reply.startswith("done conversation_id=alice")
    body = reply.split("\n\n", 1)[1]
    assert body.strip() and "terminal: ls" in body


def test_history_persists_across_adapter_instances():
    history.append("alice-t1", "alice", "hello")
    history.append("alice-t1", "hermes", "hi alice")
    fresh = _adapter()
    rendered = history.render("alice-t1", 10)
    assert "alice: hello" in rendered and "hermes: hi alice" in rendered
    assert fresh._conv("alice-t1").last_reply == ""  # live state is per-instance; transcript is not


# --------------------------------------------------------------------------- end to end


@pytest.mark.asyncio
async def test_chat_wait_reply_loop_over_real_streamable_http(monkeypatch):
    monkeypatch.setenv("MCP_HTTP_PEER_TOKENS", "tester:tok-tester")
    port = _free_port()
    adapter = _adapter(port)

    async def fake_handler(event):
        cid = event.source.chat_id
        first = await adapter.send(cid, "🔧 terminal: git status", metadata={})
        await asyncio.sleep(0.3)
        await adapter.edit_message(cid, first.message_id, "🔧 terminal: git status\n🔧 read_file: README.md")
        await asyncio.sleep(2.5)
        await adapter.send(cid, f"Hello {event.user_id}! You said: {event.text.splitlines()[-1]}",
                           metadata={"notify": True})
        await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)

    adapter._message_handler = fake_handler
    assert await adapter.connect() is True
    try:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client
        import httpx2 as httpx

        def text(res) -> str:
            return res.content[0].text

        async with httpx.AsyncClient(headers={"Authorization": "Bearer tok-tester"},
                                     timeout=httpx.Timeout(10, read=120)) as hc:
            async with streamable_http_client(f"http://127.0.0.1:{port}/mcp", http_client=hc) as streams:
                async with ClientSession(streams[0], streams[1]) as s:
                    await s.initialize()
                    names = {t.name for t in (await s.list_tools()).tools}
                    assert {"whoami", "new_conversation", "chat", "wait_reply", "status", "cancel", "history"} <= names
                    assert text(await s.call_tool("whoami", {})) == "tester"
                    cid = text(await s.call_tool("new_conversation", {}))
                    assert cid.startswith("tester-")
                    accepted = text(await s.call_tool("chat", {"message": "ping", "conversation_id": cid}))
                    assert accepted.startswith(f"accepted conversation_id={cid}")
                    working = text(await s.call_tool("wait_reply", {"conversation_id": cid, "timeout_s": 1.5}))
                    assert working.startswith("working") and "last activity" in working
                    assert "read_file: README.md" in working
                    done = text(await s.call_tool("wait_reply", {"conversation_id": cid, "timeout_s": 30}))
                    assert done.startswith(f"done conversation_id={cid}") and "Hello tester! You said: ping" in done
                    foreign = text(await s.call_tool("wait_reply", {"conversation_id": "someone-else"}))
                    assert "does not belong to tester" in foreign
                    hist = text(await s.call_tool("history", {"conversation_id": cid}))
                    assert "tester: ping" in hist and "Hello tester" in hist

        # Unauthenticated requests are rejected before reaching the MCP app; /health is open.
        async with httpx.AsyncClient(timeout=10) as anon:
            assert (await anon.get(f"http://127.0.0.1:{port}/health")).json()["ok"] is True
            assert (await anon.post(f"http://127.0.0.1:{port}/mcp", json={})).status_code == 401
    finally:
        await adapter.disconnect()
    assert _current_peer.get() == ""
