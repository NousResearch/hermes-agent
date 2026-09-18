"""A2A peer identity refinement — port-less ip: identities get a routable peer.

Regression tests: a same-host gateway recorded the inbound caller as
``ip:127.0.0.1`` (localhost-only mode authenticates by client IP only — no
port), so the completion push fell back to the receiving gateway's own
loopback endpoint and the calling gateway received nothing.

The fix has two halves:

- **Outbound** — every message this gateway sends (``a2a_call`` and
  out-of-band pushes) stamps an A2A v1.0 ``sender`` AgentName carrying the
  gateway's real endpoint URL (host + port), so the receiving gateway can
  learn where to push back.
- **Inbound** — ``_refine_peer_identity`` resolves a port-less ``ip:``
  identity to a configured ``a2a_agents`` key (via sender agentId/name) or a
  validated sender URL before the context→peer registration, so
  ``_push_out_of_band`` can route the push to the peer's real gateway.
"""

from __future__ import annotations

import asyncio
import json
import threading

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.a2a import protocol, tools
from plugins.platforms.a2a.adapter import A2AAdapter


def _bare_adapter() -> A2AAdapter:
    return A2AAdapter(PlatformConfig(enabled=True))


def _sender_message(text: str = "hello", **sender) -> dict:
    msg = protocol.text_message(protocol.ROLE_USER, text, context_id="ctx-peer-1", sender=sender if sender else None)
    return msg


# ---------------------------------------------------------------------------
# text_message / sender identity
# ---------------------------------------------------------------------------


def test_text_message_stamps_sender():
    msg = protocol.text_message(
        protocol.ROLE_USER, "hi", context_id="ctx-1",
        sender={"agentId": "peer-a", "name": "peer-a", "url": "http://127.0.0.1:8802"},
    )
    assert msg["metadata"]["a2a.sender"] == {
        "agentId": "peer-a", "name": "peer-a", "url": "http://127.0.0.1:8802",
    }
    # Without sender, the message shape is unchanged (v0.3 peers etc.).
    bare = protocol.text_message(protocol.ROLE_USER, "hi", context_id="ctx-1")
    assert "metadata" not in bare


def test_sender_identity_includes_port():
    adapter = _bare_adapter()
    ident = adapter._sender_identity()
    assert ident["agentId"]
    assert ident["name"] == ident["agentId"]
    assert ident["url"].startswith("http://")
    assert ":" in ident["url"].rsplit("/", 1)[-1]  # host:port, not host alone


def test_own_sender_config_fallback_without_live_adapter(monkeypatch):
    """No live adapter (CLI/helper process) must still yield a
    sender identity derived from config/env — helper-sent messages that
    carried no sender refined to ip:127.0.0.1 and their replies were
    silently dropped."""
    monkeypatch.setenv("A2A_AGENT_NAME", "helper-agent")
    monkeypatch.setenv("A2A_PORT", "12345")
    with _ADAPTERS_CLEARED():
        ident = A2AAdapter._own_sender()
        assert ident == {
            "agentId": "helper-agent",
            "name": "helper-agent",
            "url": "http://127.0.0.1:12345",
        }


def test_own_sender_reads_live_adapter():
    adapter = _bare_adapter()
    try:
        ident = A2AAdapter._own_sender()
        assert ident.get("url") == adapter._sender_identity()["url"]
    finally:
        adapter._unregister_adapter()


# ---------------------------------------------------------------------------
# _refine_peer_identity
# ---------------------------------------------------------------------------


def test_refine_keeps_bearer_identity(monkeypatch):
    adapter = _bare_adapter()
    # Bearer-authenticated identities are token-derived names — already
    # resolvable, never rewritten from the body.
    assert adapter._refine_peer_identity("bearer-peer-name", {"message": _sender_message()}, "ctx-1") == "bearer-peer-name"


def test_refine_unchanged_without_sender(monkeypatch):
    adapter = _bare_adapter()
    params = {"message": protocol.text_message(protocol.ROLE_USER, "hi", context_id="ctx-1")}
    assert adapter._refine_peer_identity("ip:127.0.0.1", params, "ctx-1") == "ip:127.0.0.1"


def test_refine_matches_configured_agent_id(monkeypatch):
    monkeypatch.setattr(
        tools, "_load_config",
        lambda: {"a2a_agents": {"peer-a": {"url": "http://127.0.0.1:8802"}}},
    )
    adapter = _bare_adapter()
    params = {"message": _sender_message(agentId="peer-a", name="peer-a", url="http://127.0.0.1:8802")}
    assert adapter._refine_peer_identity("ip:127.0.0.1", params, "ctx-1") == "peer-a"


def test_refine_matches_configured_name(monkeypatch):
    # Security: agentId alone must not promote ip: identity without URL/origin validation
    monkeypatch.setattr(
        tools, "_load_config",
        lambda: {"a2a_agents": {"peer-a": {"url": "http://127.0.0.1:8802"}}},
    )
    adapter = _bare_adapter()
    params = {"message": _sender_message(name="peer-a")}
    assert adapter._refine_peer_identity("ip:127.0.0.1", params, "ctx-1") == "ip:127.0.0.1"


def test_refine_uses_loopback_sender_url(monkeypatch):
    monkeypatch.setattr(tools, "_load_config", lambda: {})
    adapter = _bare_adapter()
    params = {"message": _sender_message(url="http://127.0.0.1:8802")}
    assert adapter._refine_peer_identity("ip:127.0.0.1", params, "ctx-1") == "http://127.0.0.1:8802"


def test_refine_accepts_configured_host_url(monkeypatch):
    # A remote/tailscale peer's URL host must appear in a2a_agents to be
    # honored (defense in depth against body-supplied external targets).
    monkeypatch.setattr(
        tools, "_load_config",
        lambda: {"a2a_agents": {"remote-peer": {"url": "http://100.64.0.5:8803"}}},
    )
    adapter = _bare_adapter()
    params = {"message": _sender_message(url="http://100.64.0.5:8803")}
    assert adapter._refine_peer_identity("ip:127.0.0.1", params, "ctx-1") == "remote-peer"


def test_refine_rejects_unconfigured_external_url(monkeypatch):
    monkeypatch.setattr(tools, "_load_config", lambda: {})
    adapter = _bare_adapter()
    params = {"message": _sender_message(url="http://evil.example:8804")}
    assert adapter._refine_peer_identity("ip:127.0.0.1", params, "ctx-1") == "ip:127.0.0.1"


def test_refine_rejects_non_http_url(monkeypatch):
    monkeypatch.setattr(tools, "_load_config", lambda: {})
    adapter = _bare_adapter()
    params = {"message": _sender_message(url="file:///etc/passwd")}
    assert adapter._refine_peer_identity("ip:127.0.0.1", params, "ctx-1") == "ip:127.0.0.1"


# ---------------------------------------------------------------------------
# End-to-end: _prepare_task registers the REFINED peer
# ---------------------------------------------------------------------------


def test_prepare_task_registers_refined_peer(monkeypatch, tmp_path):
    """The regression: an inbound from a same-host gateway must register the
    peer's real endpoint (config key / URL), NOT the port-less ip: identity —
    otherwise the completion push falls back to this gateway's own endpoint
    and the caller never receives it."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(
        tools, "_load_config",
        lambda: {"a2a_agents": {"peer-a": {"url": "http://127.0.0.1:8802"}}},
    )
    adapter = _bare_adapter()
    loop = asyncio.new_event_loop()
    adapter._loop = loop
    try:
        async def fake_handle_message(event):
            pass  # do not resolve — the registration happens before dispatch

        adapter._message_handler = fake_handle_message  # type: ignore
        # Mock dispatch to run synchronously without needing a running loop thread
        from concurrent.futures import Future as CFuture

        def fake_run(coro, target_loop):
            try:
                asyncio.run(coro)
            except RuntimeError:
                new_loop = asyncio.new_event_loop()
                try:
                    new_loop.run_until_complete(coro)
                finally:
                    try:
                        new_loop.close()
                    except Exception:
                        pass
            fut = CFuture()
            fut.set_result(None)
            return fut

        monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", fake_run)
        params = {
            "message": protocol.text_message(
                protocol.ROLE_USER, "roundtrip test", context_id="ctx-roundtrip-1",
                sender={"agentId": "peer-a", "name": "peer-a", "url": "http://127.0.0.1:8802"},
            ),
        }
        terminal, pending = adapter._prepare_task(params, "ip:127.0.0.1")
        assert terminal is None
        assert pending is not None
        with adapter._context_peers_lock:
            assert adapter._context_peers.get("ctx-roundtrip-1") == "peer-a"
        # Persisted too — survives gateway restarts.
        peers_file = tmp_path / "a2a_context_peers.json"
        assert json.loads(peers_file.read_text())["ctx-roundtrip-1"] == "peer-a"
    finally:
        try:
            if not loop.is_closed():
                loop.close()
        except Exception:
            pass
        try:
            adapter._unregister_adapter()
        except Exception:
            pass
        try:
            pol = asyncio.get_event_loop_policy()
            cur = None
            try:
                cur = pol.get_event_loop()
            except RuntimeError:
                cur = None
            if cur is loop:
                pol.set_event_loop(None)
            elif cur is not None and cur.is_closed():
                pol.set_event_loop(None)
        except Exception:
            try:
                asyncio.set_event_loop(None)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Push path: out-of-band completion goes to the REFINED peer's real URL
# ---------------------------------------------------------------------------


def test_push_out_of_band_hits_refined_peer_url(monkeypatch, tmp_path):
    """_push_out_of_band must POST to the refined peer's gateway (with port),
    not fall back to this gateway's own endpoint when the loopback identity
    refines to a configured peer URL — that fallback is exactly the failure
    this fixes."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(
        tools, "_load_config",
        lambda: {"a2a_agents": {"peer-a": {"url": "http://127.0.0.1:8802"}}},
    )
    monkeypatch.setattr(tools, "_fetch_card", lambda *a, **k: (_ for _ in ()).throw(ConnectionError("no card")))
    posted: dict = {}

    def fake_post(url, body, headers, timeout, **kw):
        posted["url"] = url
        posted["body"] = body
        return {"jsonrpc": "2.0", "id": body["id"], "result": {"task": protocol.build_task("t-valid", "ctx-roundtrip-1", protocol.STATE_COMPLETED, "ok")}}

    monkeypatch.setattr(tools, "_http_post_json", fake_post)

    adapter = _bare_adapter()
    try:
        adapter._register_context_peer("ctx-roundtrip-1", "peer-a")
        adapter._push_out_of_band("ctx-roundtrip-1", "✔ task done")
        assert posted["url"] == "http://127.0.0.1:8802"
        sent_msg = posted["body"]["params"]["message"]
        assert sent_msg["contextId"] == "ctx-roundtrip-1"
        # The pushed message itself carries the sender so the receiving
        # gateway can register OUR real endpoint for any follow-up push.
        assert sent_msg["metadata"]["a2a.sender"]["url"] == adapter._sender_identity()["url"]
    finally:
        adapter._unregister_adapter()


def test_push_out_of_band_own_endpoint_delivers_in_process(monkeypatch, tmp_path):
    """A context→peer map refined to THIS gateway's own URL must deliver
    in-process (via _prepare_task), never via a synchronous HTTP self-call
    that times out while the session processes the message."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(tools, "_load_config", lambda: {})
    posted: dict = {}

    def fake_post(url, body, headers, timeout, **kw):
        posted["url"] = url
        return {"jsonrpc": "2.0", "id": body["id"], "result": {}}

    monkeypatch.setattr(tools, "_http_post_json", fake_post)

    adapter = _bare_adapter()
    prepared: dict = {}

    def fake_prepare_task(params, peer):
        prepared["peer"] = peer
        prepared["params"] = params
        return None, {"task_id": "task-inproc", "context_id": "ctx-own-1",
                       "peer": peer, "created_iso": "2026-01-01T00:00:00Z",
                       "started": __import__('time').time()}

    monkeypatch.setattr(adapter, "_prepare_task", fake_prepare_task)
    try:
        own_url = adapter._sender_identity()["url"]
        adapter._register_context_peer("ctx-own-1", own_url)
        adapter._push_out_of_band("ctx-own-1", "✔ done")
        # No HTTP round-trip to ourselves; the in-process path ran instead.
        assert posted == {}
        assert prepared["peer"] == own_url
        assert prepared["params"]["message"]["contextId"] == "ctx-own-1"
    finally:
        adapter._unregister_adapter()


def test_is_own_endpoint():
    from plugins.platforms.a2a.adapter import _is_own_endpoint

    assert _is_own_endpoint("http://127.0.0.1:8801", "127.0.0.1", 8801) is True
    assert _is_own_endpoint("http://127.0.0.1:8802", "127.0.0.1", 8801) is False
    assert _is_own_endpoint("http://100.64.0.5:8801", "127.0.0.1", 8801) is False
    assert _is_own_endpoint("ftp://127.0.0.1:8801", "127.0.0.1", 8801) is False


def test_send_task_stamps_sender_on_outbound(monkeypatch, tmp_path):
    """a2a_call messages must carry the sender AgentName so the receiving
    gateway can learn this gateway's real endpoint (port included)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(tools, "_fetch_card", lambda *a, **k: None)
    posted: dict = {}

    def fake_post(url, body, headers, timeout, **kw):
        posted["url"] = url
        posted["body"] = body
        return {"jsonrpc": "2.0", "id": body["id"], "result": {"task": protocol.build_task(
            "task-1", "ctx-out-1", protocol.STATE_COMPLETED, "ok",
        )}}

    monkeypatch.setattr(tools, "_http_post_json", fake_post)

    adapter = _bare_adapter()
    try:
        reply, _ctx, _state = tools._send_task(
            "peer-agent", {"url": "http://127.0.0.1:8801", "auth": {}, "timeout": 5},
            "hello peer-agent", "ctx-out-1",
        )
        assert reply == "ok"
        sent = posted["body"]["params"]["message"]
        assert sent["metadata"]["a2a.sender"]["agentId"] == adapter.agent_name
        assert sent["metadata"]["a2a.sender"]["url"] == adapter._sender_identity()["url"]
    finally:
        adapter._unregister_adapter()


# ---------------------------------------------------------------------------
# Helper: clear the module-level live-adapter registry around a test
# ---------------------------------------------------------------------------


class _ADAPTERS_CLEARED:
    def __enter__(self):
        import plugins.platforms.a2a.adapter as mod

        self._mod = mod
        with mod._ADAPTERS_GUARD:
            self._saved = dict(mod._ADAPTERS)
            mod._ADAPTERS.clear()
        return self

    def __exit__(self, *exc):
        with self._mod._ADAPTERS_GUARD:
            self._mod._ADAPTERS.update(self._saved)
        return False
