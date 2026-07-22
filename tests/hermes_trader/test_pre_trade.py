"""Tests for hermes_trader.hooks.pre_trade MCP interception."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from hermes_trader.hooks.pre_trade import intercept_mcp_tool_call
from hermes_trader.risk.mandate import save_mandate, sign_mandate

TEST_KEY = b"test-mandate-secret-key"
WALLET = "0xabcdef1234567890abcdef1234567890abcdef12"


@pytest.fixture
def trader_home(tmp_path, monkeypatch):
    hh = tmp_path / "hermes-home"
    trader = hh / "trader"
    trader.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(hh))
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: hh)

    config = trader / "hermes_trader.yaml"
    config.write_text(
        "mode: live\nmcp_server_name: defi-trading\nprimary_chain: base\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_TRADER_CONFIG", str(config))
    return trader


def test_read_tool_not_blocked():
    assert intercept_mcp_tool_call("defi-trading", "get_swap_quote", {}) is None


def test_write_tool_blocked_paper_mode(trader_home, monkeypatch):
    config = trader_home / "hermes_trader.yaml"
    config.write_text("mode: paper\nmcp_server_name: defi-trading\n", encoding="utf-8")
    err = intercept_mcp_tool_call("defi-trading", "execute_swap", {})
    assert err is not None
    assert "Paper mode" in err


def test_write_tool_blocked_kill_switch(trader_home, monkeypatch):
    monkeypatch.setenv("HERMES_TRADER_KILL_SWITCH", "true")
    err = intercept_mcp_tool_call("defi-trading", "execute_swap", {})
    assert err is not None
    assert "KILL_SWITCH" in err


def test_write_tool_blocked_missing_mandate(trader_home, monkeypatch):
    monkeypatch.delenv("HERMES_TRADER_KILL_SWITCH", raising=False)
    err = intercept_mcp_tool_call("defi-trading", "submit_gasless_swap", {})
    assert err is not None
    assert "mandate.json missing" in err


def test_write_tool_allowed_with_valid_mandate(trader_home, monkeypatch):
    monkeypatch.setenv("USER_ADDRESS", WALLET)
    monkeypatch.setenv("HERMES_TRADER_MANDATE_SECRET", TEST_KEY.decode())
    mandate = sign_mandate(WALLET, signing_key=TEST_KEY)
    save_mandate(mandate, trader_home / "mandate.json")
    assert intercept_mcp_tool_call("defi-trading", "execute_swap", {}) is None


def test_other_server_write_tools_not_intercepted(trader_home):
    assert intercept_mcp_tool_call("other-mcp", "execute_swap", {}) is None


def test_write_tool_blocked_rate_limit(trader_home, monkeypatch):
    import time

    monkeypatch.setenv("USER_ADDRESS", WALLET)
    monkeypatch.setenv("HERMES_TRADER_MANDATE_SECRET", TEST_KEY.decode())
    mandate = sign_mandate(WALLET, signing_key=TEST_KEY)
    save_mandate(mandate, trader_home / "mandate.json")

    state = trader_home / "write_rate_limit.json"
    state.write_text(json.dumps([time.time(), time.time() + 1]), encoding="utf-8")
    config = trader_home / "hermes_trader.yaml"
    config.write_text(
        "mode: live\nmcp_server_name: defi-trading\nmax_write_tools_per_hour: 2\n",
        encoding="utf-8",
    )
    err = intercept_mcp_tool_call("defi-trading", "execute_swap", {})
    assert err is not None
    assert "rate limit" in err.lower()


def test_mcp_handler_integration_blocks_write_tool(trader_home, monkeypatch):
    """_make_tool_handler returns JSON error when pre_trade blocks."""
    monkeypatch.delenv("HERMES_TRADER_KILL_SWITCH", raising=False)
    from tools.mcp_tool import _make_tool_handler

    handler = _make_tool_handler("defi-trading", "execute_swap", 30.0)
    with patch("tools.mcp_tool._servers", {}):
        result = handler({})
    payload = json.loads(result)
    assert "error" in payload
    assert "mandate" in payload["error"].lower() or "Paper mode" in payload["error"]