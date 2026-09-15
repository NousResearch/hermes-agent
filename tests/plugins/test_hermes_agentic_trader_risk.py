"""Deterministic live-risk tests for the trader repair."""

from __future__ import annotations

import importlib.util
import sys
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_package_module(name: str):
    plugin_dir = _repo_root() / "plugins" / "hermes-agentic-trader"
    if "hermes_plugins" not in sys.modules:
        ns = types.ModuleType("hermes_plugins")
        ns.__path__ = []
        sys.modules["hermes_plugins"] = ns
    package_name = "hermes_plugins.hermes_agentic_trader"
    if package_name not in sys.modules:
        pkg = types.ModuleType(package_name)
        pkg.__path__ = [str(plugin_dir)]
        sys.modules[package_name] = pkg
    full_name = package_name + "." + name
    spec = importlib.util.spec_from_file_location(
        full_name,
        plugin_dir / f"{name}.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def modules():
    policy = _load_package_module("policy")
    quote_cache = _load_package_module("quote_cache")
    mandate = _load_package_module("mandate")
    risk = _load_package_module("risk")
    quote_cache.clear_quote_cache()
    return policy, quote_cache, mandate, risk


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv(
        "USER_ADDRESS",
        "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    )
    monkeypatch.setenv("HERMES_TRADER_MANDATE_SECRET", "test-secret")
    return home


def _record(quote_cache, *, slippage=50, chain_id=8453):
    quote = {
        "chainId": chain_id,
        "transaction": {
            "to": "0xdddddddddddddddddddddddddddddddddddddddd",
            "data": "0xdeadbeef",
        },
    }
    args = {
        "chainId": chain_id,
        "sellToken": "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        "buyToken": "0xcccccccccccccccccccccccccccccccccccccccc",
        "sellAmount": "1000000",
        "slippageBps": slippage,
    }
    assert quote_cache.capture_quote(
        tool_name="get_swap_quote",
        args=args,
        result={"data": quote},
        session_id="s1",
        now=10.0,
    )
    record = quote_cache.resolve_execution_quote(
        tool_name="execute_swap",
        args={"quoteData": quote},
        session_id="s1",
        now=10.0,
    )
    assert record is not None
    return record


def _mandate(mandate):
    return mandate.sign_mandate(
        "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        allowed_chain_ids=(8453,),
        max_trade_usd=50,
        max_slippage_bps=100,
        max_writes_per_hour=10,
        signing_key=b"test-secret",
    )


def test_live_policy_defaults_to_canary(modules):
    policy, _quote_cache, _mandate_mod, _risk = modules
    cfg = policy.TraderPolicy.from_mapping({"mode": "live"})
    assert cfg.rollout_stage == "canary"
    assert cfg.allowed_chain_ids == (8453,)
    assert cfg.rollout_cap_usd == 50.0


def test_paper_policy_has_zero_cap(modules):
    policy, _quote_cache, _mandate_mod, _risk = modules
    cfg = policy.TraderPolicy.from_mapping({"mode": "paper"})
    assert cfg.rollout_stage == "paper"
    assert cfg.rollout_cap_usd == 0.0


def test_mandate_requires_independent_secret(modules, monkeypatch):
    _policy, _quote_cache, mandate, _risk = modules
    monkeypatch.delenv("HERMES_TRADER_MANDATE_SECRET", raising=False)
    with pytest.raises(ValueError, match="MANDATE_SECRET"):
        mandate.sign_mandate(
            "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        )


def test_expired_mandate_is_rejected(modules):
    _policy, _quote_cache, mandate, _risk = modules
    now = datetime.now(timezone.utc)
    m = mandate.sign_mandate(
        "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        signed_at=(now - timedelta(hours=2)).isoformat(),
        expires_at=(now - timedelta(hours=1)).isoformat(),
        signing_key=b"test-secret",
    )
    ok, message = mandate.validate_mandate(
        m,
        signing_key=b"test-secret",
        now=now,
    )
    assert not ok
    assert "expired" in message


def test_valid_mandate_detects_tampering(modules):
    _policy, _quote_cache, mandate, _risk = modules
    m = _mandate(mandate)
    tampered = mandate.Mandate(
        **{**m.__dict__, "max_trade_usd": 500.0}
    )
    ok, message = mandate.validate_mandate(
        tampered,
        signing_key=b"test-secret",
    )
    assert not ok
    assert "signature invalid" in message


def test_bound_quote_reaches_notional_fail_closed_gate(modules):
    policy, quote_cache, mandate, risk = modules
    record = _record(quote_cache)
    decision = risk.evaluate_bound_quote(
        policy=policy.TraderPolicy.from_mapping({"mode": "live"}),
        record=record,
        mandate=_mandate(mandate),
        notional_usd=None,
        portfolio_value_usd=None,
        daily_loss_pct=None,
        pool_liquidity_usd=None,
    )
    assert not decision.approved
    assert decision.reason == risk.RiskReason.NOTIONAL_UNAVAILABLE


def test_slippage_rejected_before_valuation(modules):
    policy, quote_cache, mandate, risk = modules
    record = _record(quote_cache, slippage=250)
    decision = risk.evaluate_bound_quote(
        policy=policy.TraderPolicy.from_mapping(
            {"mode": "live", "max_slippage_bps": 100}
        ),
        record=record,
        mandate=_mandate(mandate),
        notional_usd=None,
        portfolio_value_usd=None,
        daily_loss_pct=None,
        pool_liquidity_usd=None,
    )
    assert decision.reason == risk.RiskReason.SLIPPAGE


def test_chain_rejected_before_valuation(modules):
    policy, quote_cache, mandate, risk = modules
    record = _record(quote_cache, chain_id=1)
    decision = risk.evaluate_bound_quote(
        policy=policy.TraderPolicy.from_mapping({"mode": "live"}),
        record=record,
        mandate=_mandate(mandate),
        notional_usd=None,
        portfolio_value_usd=None,
        daily_loss_pct=None,
        pool_liquidity_usd=None,
    )
    assert decision.reason == risk.RiskReason.CHAIN_DENIED


def test_complete_safe_context_can_approve(modules):
    policy, quote_cache, mandate, risk = modules
    record = _record(quote_cache)
    decision = risk.evaluate_bound_quote(
        policy=policy.TraderPolicy.from_mapping({"mode": "live"}),
        record=record,
        mandate=_mandate(mandate),
        notional_usd=25.0,
        portfolio_value_usd=1000.0,
        daily_loss_pct=0.5,
        pool_liquidity_usd=250000.0,
    )
    assert decision.approved
    assert decision.reason is None


def test_position_limit_blocks_oversize_context(modules):
    policy, quote_cache, mandate, risk = modules
    record = _record(quote_cache)
    decision = risk.evaluate_bound_quote(
        policy=policy.TraderPolicy.from_mapping(
            {"mode": "live", "max_position_pct": 2}
        ),
        record=record,
        mandate=_mandate(mandate),
        notional_usd=25.0,
        portfolio_value_usd=1000.0,
        daily_loss_pct=0.5,
        pool_liquidity_usd=250000.0,
    )
    assert decision.reason == risk.RiskReason.POSITION_LIMIT
