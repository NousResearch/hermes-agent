"""Regression coverage for the trading-risk and write-rate review fixes."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

from hermes_trader.audit.rate_limit import WriteToolRateLimiter
from hermes_trader.hooks.pre_trade import prepare_mcp_tool_call
from hermes_trader.risk.mandate import save_mandate, sign_mandate

TEST_KEY = b"test-mandate-secret-key"
WALLET = "0xabcdef1234567890abcdef1234567890abcdef12"
TOKEN = "0x1234567890123456789012345678901234567890"


def _live_home(tmp_path, monkeypatch, *, max_writes: int = 10):
    home = tmp_path / "hermes-home"
    trader = home / "trader"
    trader.mkdir(parents=True)
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: home)
    monkeypatch.setenv("USER_ADDRESS", WALLET)
    monkeypatch.setenv("HERMES_TRADER_MANDATE_SECRET", TEST_KEY.decode())
    save_mandate(sign_mandate(WALLET, signing_key=TEST_KEY), trader / "mandate.json")
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {
            "trader": {
                "mode": "live",
                "allowed_chains": ["base"],
                "max_slippage_bps": 100,
                "max_write_tools_per_hour": max_writes,
            }
        },
    )
    return trader


def _args(**overrides):
    values = {
        "action": "buy",
        "chain": "base",
        "token_address": TOKEN,
        "amount_usd": 25,
        "slippage_bps": 50,
    }
    values.update(overrides)
    return values


def test_direct_live_mcp_write_runs_argument_aware_risk_gate(tmp_path, monkeypatch):
    _live_home(tmp_path, monkeypatch)
    denied = prepare_mcp_tool_call(
        "defi-trading", "execute_swap", _args(chain="ethereum", slippage_bps=250)
    )
    assert denied.error is not None
    assert "Chain 'ethereum'" in denied.error
    assert denied.reservation_id is None


def test_direct_live_mcp_write_rejects_missing_order_arguments(tmp_path, monkeypatch):
    _live_home(tmp_path, monkeypatch)
    denied = prepare_mcp_tool_call("defi-trading", "execute_swap", {})
    assert denied.error is not None
    assert "missing chain" in denied.error


def test_concurrent_write_reservations_cannot_oversubscribe(tmp_path, monkeypatch):
    trader = _live_home(tmp_path, monkeypatch, max_writes=1)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(
            pool.map(
                lambda _: prepare_mcp_tool_call("defi-trading", "execute_swap", _args()),
                range(8),
            )
        )
    reservations = [result.reservation_id for result in results if result.reservation_id]
    assert len(reservations) == 1
    state = json.loads((trader / "write_rate_limit.json").read_text(encoding="utf-8"))
    assert len(state) == 1
    assert state[0]["status"] == "pending"


def test_failed_dispatch_releases_reserved_capacity(tmp_path, monkeypatch):
    trader = _live_home(tmp_path, monkeypatch, max_writes=1)
    limiter = WriteToolRateLimiter(max_per_hour=1, state_path=trader / "write_rate_limit.json")
    reservation_id = limiter.reserve(now=1_700_000_000)
    assert reservation_id is not None
    limiter.reconcile(reservation_id, succeeded=False)
    assert limiter.reserve(now=1_700_000_001) is not None


def test_default_loader_ignores_behavior_environment_override(tmp_path, monkeypatch):
    from hermes_trader.config import load_trader_config

    override = tmp_path / "override.yaml"
    override.write_text("mode: live\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_TRADER_CONFIG", str(override))
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"trader": {"mode": "paper"}})
    assert load_trader_config().mode == "paper"
