"""Tests for optional-skills/mcp/web3-chain-tools (queue, limits, MCP wiring)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _scripts_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "optional-skills/mcp/web3-chain-tools/scripts"


def _prep_path() -> None:
    sd = str(_scripts_dir())
    if sd not in sys.path:
        sys.path.insert(0, sd)


def test_token_bucket_capacity():
    _prep_path()
    from rate_limit import TokenBucket

    b = TokenBucket(capacity=2.0, refill_per_sec=100.0)
    assert b.acquire() and b.acquire() and not b.acquire()


def test_event_queue_roundtrip(tmp_path):
    _prep_path()
    from event_queue import EventQueue

    db = tmp_path / "q.sqlite3"
    q = EventQueue(db)
    eid = q.enqueue("evm", {"hello": "world"})
    assert eid >= 1
    rows = q.dequeue(10)
    assert len(rows) == 1
    assert rows[0]["payload"]["hello"] == "world"
    q.close()


def test_sync_preview_redact():
    _prep_path()
    from approval_hooks import sync_preview_redact

    out = sync_preview_redact({"raw_transaction": "0x" + "ab" * 40})
    assert "redacted" in out["raw_transaction"]


def test_reject_secrets_in_args():
    pytest.importorskip("mcp")
    _prep_path()
    from tool_handlers import handle_tool, reject_secrets_in_args

    try:
        reject_secrets_in_args({"private_key": "0xabc"})
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


@pytest.mark.asyncio
async def test_monitor_event_enqueue(monkeypatch, tmp_path):
    pytest.importorskip("mcp")
    pytest.importorskip("web3", reason="optional hermes-agent[web3-mcp] extra")
    db = tmp_path / "q.sqlite3"
    monkeypatch.setenv("WEB3_MCP_QUEUE_DB", str(db))
    for name in ("tool_handlers", "event_queue", "tools_schema"):
        sys.modules.pop(name, None)
    _prep_path()
    import evm_tools
    import tool_handlers as th

    monkeypatch.setattr(evm_tools, "evm_get_logs", lambda *_a, **_k: [{"stub": True}])
    th.RT._w3 = object()
    out = await th.handle_tool("monitor_event", {})
    assert out[0].type == "text"
    assert "enqueued_id" in out[0].text


def test_build_server_smoke():
    pytest.importorskip("mcp")
    pytest.importorskip("web3", reason="optional hermes-agent[web3-mcp] extra")
    script = _scripts_dir() / "web3_mcp_server.py"
    spec = importlib.util.spec_from_file_location("web3_mcp_server_smoke", script)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    _prep_path()
    spec.loader.exec_module(mod)
    srv = mod.build_server()
    assert srv.name == "hermes-web3-chain-tools"


@pytest.mark.asyncio
async def test_approval_gateway_unreachable_denies(monkeypatch):
    pytest.importorskip("mcp")
    _prep_path()
    from approval_hooks import approval_gateway_allow

    monkeypatch.setenv("WEB3_APPROVAL_GATEWAY_URL", "http://127.0.0.1:9/nope")
    monkeypatch.setenv("WEB3_APPROVAL_DENY_ON_ERROR", "1")
    ok = await approval_gateway_allow(tool_name="t", preview={})
    assert ok is False
