"""Tests for optional Web3 MCP governance policy layer."""

import json
from pathlib import Path

import pytest

from agent import web3_mcp_governance as w3


def _policy(**kwargs):
    base = {
        "enabled": True,
        "server_names": ["chain_mcp"],
        "blocked_tool_substrings": ["send_raw", "submit_transaction"],
        "allowed_tool_substrings": [],
        "simulate_on_block": False,
        "mcp_call_extra_retries": 0,
        "retry_backoff_seconds": 1.5,
        "persist_chain_hints": True,
        "inject_wallet_context_in_prompt": False,
    }
    base.update(kwargs)
    return base


def test_disabled_by_default(monkeypatch):
    monkeypatch.setattr(w3, "_effective_policy", w3._defaults)
    out = w3.evaluate_mcp_tool_call("chain_mcp", "send_raw_transaction", {})
    assert out.mode == "allow"


def test_blocks_destructive_tool_name(monkeypatch):
    monkeypatch.setattr(w3, "_effective_policy", lambda: _policy())
    out = w3.evaluate_mcp_tool_call("chain_mcp", "evm_send_raw_transaction", {})
    assert out.mode == "block"
    assert "send_raw" in out.message.lower() or "blocked" in out.message.lower()


def test_wrong_server_not_governed(monkeypatch):
    monkeypatch.setattr(w3, "_effective_policy", lambda: _policy())
    out = w3.evaluate_mcp_tool_call("filesystem", "send_raw_transaction", {})
    assert out.mode == "allow"


def test_allowlist_restricts(monkeypatch):
    monkeypatch.setattr(
        w3,
        "_effective_policy",
        lambda: _policy(
            allowed_tool_substrings=["estimate"],
            blocked_tool_substrings=[],
        ),
    )
    out = w3.evaluate_mcp_tool_call("chain_mcp", "some_random_tool", {})
    assert out.mode == "block"


def test_simulate_stub(monkeypatch):
    monkeypatch.setattr(
        w3,
        "_effective_policy",
        lambda: _policy(simulate_on_block=True),
    )
    out = w3.evaluate_mcp_tool_call("chain_mcp", "send_raw_tx", {})
    assert out.mode == "simulate"
    assert out.simulated_payload is not None
    assert out.simulated_payload.get("simulated") is True


def test_persist_chain_hints(tmp_path, monkeypatch):
    monkeypatch.setattr(w3, "_effective_policy", lambda: _policy())
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    w3.maybe_persist_chain_hints(
        "chain_mcp",
        "route",
        {"chainId": "1", "from": "0x" + "a" * 40},
    )
    snap = w3.load_wallet_context_snapshot()
    assert "recent_calls" in snap
    assert snap["recent_calls"][-1]["chains"]["chainId"] == "1"


def test_load_snapshot_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    assert w3.load_wallet_context_snapshot() == {}


def test_wallet_context_prompt_block_disabled(monkeypatch):
    monkeypatch.setattr(w3, "_effective_policy", lambda: _policy(inject_wallet_context_in_prompt=False))
    assert w3.wallet_context_prompt_block() == ""


def test_wallet_context_prompt_block_formats(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setattr(
        w3,
        "_effective_policy",
        lambda: _policy(inject_wallet_context_in_prompt=True),
    )
    hist = [
        {
            "server": "srv",
            "tool": "route",
            "chains": {"chainId": "42161"},
            "address_hints": {},
        }
    ]
    monkeypatch.setattr(w3, "load_wallet_context_snapshot", lambda: {"recent_calls": hist})
    text = w3.wallet_context_prompt_block()
    assert "42161" in text
    assert "srv" in text and "route" in text


def test_build_web3_wallet_prompt_import(monkeypatch):
    from agent import prompt_builder as pb

    monkeypatch.setattr(w3, "_effective_policy", lambda: _policy(inject_wallet_context_in_prompt=False))
    assert pb.build_web3_mcp_wallet_context_prompt() == ""
