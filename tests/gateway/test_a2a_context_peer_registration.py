"""Cross-platform A2A context-peer registration.

The A2A adapter's ``_context_peers`` map only learned peers from inbound
A2A tasks. A context born on another platform (discord, telegram, CLI/ACP,
api_server) had no peer entry, so the task notifier's out-of-band
completion push found no peer and dropped the message. The outbound client
tools now register the context→peer mapping on every live local adapter
before the A2A call, so the completion push has a target regardless of
where the context originated.
"""

from __future__ import annotations

import asyncio

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.a2a import protocol, tools
from plugins.platforms.a2a.adapter import A2AAdapter


def _bare_adapter():
    return A2AAdapter(PlatformConfig(enabled=True))


def test_outbound_call_registers_context_peer_on_local_adapter(monkeypatch):
    """An outbound a2a_call from ANY origin registers the context→peer
    mapping on the local gateway adapter, so a later completion push can
    find the peer."""
    adapter = _bare_adapter()
    try:
        monkeypatch.setattr(
            tools, "_load_config",
            lambda: {"a2a_agents": {"peer-agent": {"url": "http://127.0.0.1:8801"}}},
        )

        def fake_post(url, body, headers, timeout):
            return protocol.jsonrpc_result(
                body["id"],
                protocol.build_task("t2", "ctx-discord-born", protocol.STATE_COMPLETED, "ok"),
            )

        monkeypatch.setattr(tools, "_http_post_json", fake_post)

        out = tools.a2a_call(
            {"agent": "peer-agent", "message": "ping", "context_id": "ctx-discord-born"}
        )
        assert "ok" in out

        # The context was born on discord — no A2A inbound ever touched the
        # adapter — yet the outbound call must have registered the peer.
        with adapter._context_peers_lock:
            assert adapter._context_peers.get("ctx-discord-born") == "peer-agent"
    finally:
        adapter._unregister_adapter()


def test_outbound_registration_is_best_effort(monkeypatch):
    """Registration must never fail the call, even with no live adapter."""
    monkeypatch.setattr(
        tools, "_load_config",
        lambda: {"a2a_agents": {"peer-agent": {"url": "http://127.0.0.1:8801"}}},
    )

    def fake_post(url, body, headers, timeout):
        return protocol.jsonrpc_result(
            body["id"],
            protocol.build_task("t2", "ctx-x", protocol.STATE_COMPLETED, "ok"),
        )

    monkeypatch.setattr(tools, "_http_post_json", fake_post)

    out = tools.a2a_call({"agent": "peer-agent", "message": "ping", "context_id": "ctx-x"})
    assert "ok" in out


def test_push_out_of_band_writes_push_audit(monkeypatch, tmp_path):
    """The out-of-band push now writes the documented 'push' audit direction
    (the push audit row was previously never written)."""
    audit_file = tmp_path / "a2a_audit.jsonl"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    # security._audit_path resolves via hermes_constants; force the file by
    # monkeypatching the audit function to record into our tmp file.
    import plugins.platforms.a2a.security as security

    records = []

    def fake_audit(direction, peer, task_id, summary, context_id=None):
        records.append((direction, peer, task_id, summary, context_id))

    monkeypatch.setattr(security, "audit", fake_audit)

    adapter = _bare_adapter()
    try:
        adapter._context_peers["ctx-audit"] = "peer-agent"
        monkeypatch.setattr(
            tools, "_load_config",
            lambda: {"a2a_agents": {"peer-agent": {"url": "http://127.0.0.1:8801"}}},
        )

        def fake_post(url, body, headers, timeout):
            return protocol.jsonrpc_result(
                body["id"],
                protocol.build_task("t2", "ctx-audit", protocol.STATE_COMPLETED, "ok"),
            )

        monkeypatch.setattr(tools, "_http_post_json", fake_post)

        async def run():
            return await adapter.send("ctx-audit", "late reply", metadata={"notify": True})

        res = asyncio.run(run())
        assert res.success is True
        assert any(direction == "push" for direction, _, _, _, _ in records)
    finally:
        adapter._unregister_adapter()
