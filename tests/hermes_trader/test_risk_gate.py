"""Tests for hermes_trader.risk.gate — every REJECT reason code."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from hermes_trader.config import TraderConfig
from hermes_trader.market_state import MarketState, PortfolioToken, build_market_state
from hermes_trader.risk.gate import (
    RejectReason,
    RiskGate,
    TradeIntent,
    is_kill_switch_active,
)
from hermes_trader.risk.mandate import sign_mandate

TEST_KEY = b"test-mandate-secret-key"
WALLET = "0xabcdef1234567890abcdef1234567890abcdef12"


@pytest.fixture(autouse=True)
def _mandate_secret(monkeypatch):
    monkeypatch.setenv("HERMES_TRADER_MANDATE_SECRET", TEST_KEY.decode())


@pytest.fixture
def live_config():
    return TraderConfig(mode="live", min_confidence=0.6, max_position_pct=5.0)


@pytest.fixture
def signed_mandate():
    return sign_mandate(WALLET, signing_key=TEST_KEY)


@pytest.fixture
def portfolio_state():
    return build_market_state(
        chain="base",
        captured_at="2026-07-07T12:00:00+00:00",
        portfolio_payload={
            "tokens": [
                {
                    "symbol": "USDC",
                    "address": "0x1",
                    "balance": 10000,
                    "balance_usd": 10000,
                }
            ]
        },
    )


def _intent(**overrides) -> TradeIntent:
    base = {
        "action": "buy",
        "chain": "base",
        "token_address": "0xtoken",
        "size_usd": 100.0,
        "confidence": 0.8,
        "reasoning": "test",
        "pool_liquidity_usd": 200_000.0,
        "slippage_bps": 50,
    }
    base.update(overrides)
    return TradeIntent.from_mapping(base)


def _gate(config: TraderConfig | None = None) -> RiskGate:
    return RiskGate(config=config or TraderConfig(mode="live"))


def test_approve_live_intent(live_config, signed_mandate, portfolio_state, monkeypatch):
    monkeypatch.setenv("USER_ADDRESS", WALLET)
    monkeypatch.setenv("HERMES_TRADER_MANDATE_SECRET", TEST_KEY.decode())
    decision = _gate(live_config).evaluate(
        _intent(),
        market_state=portfolio_state,
        mandate=signed_mandate,
    )
    assert decision.approved
    assert decision.order is not None
    assert decision.order.tool == "submit_gasless_swap"
    assert decision.order.size_usd == 100.0


def test_reject_kill_switch(live_config, signed_mandate, monkeypatch):
    monkeypatch.setenv("HERMES_TRADER_KILL_SWITCH", "1")
    decision = _gate(live_config).evaluate(_intent(), mandate=signed_mandate)
    assert not decision.approved
    assert decision.reason_code == RejectReason.KILL_SWITCH


def test_reject_kill_switch_file_sentinel(tmp_path, live_config, signed_mandate, monkeypatch):
    monkeypatch.delenv("HERMES_TRADER_KILL_SWITCH", raising=False)
    hh = tmp_path / "hermes-home"
    (hh / "trader").mkdir(parents=True)
    (hh / "trader" / "KILL_SWITCH").write_text("", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(hh))
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: hh)
    assert is_kill_switch_active()
    decision = _gate(live_config).evaluate(_intent(), mandate=signed_mandate)
    assert decision.reason_code == RejectReason.KILL_SWITCH


def test_reject_paper_mode(signed_mandate):
    cfg = TraderConfig(mode="paper")
    decision = _gate(cfg).evaluate(_intent(), mandate=signed_mandate)
    assert decision.reason_code == RejectReason.PAPER_MODE


def test_reject_chain_denied(live_config, signed_mandate):
    decision = _gate(live_config).evaluate(
        _intent(chain="polygon"),
        mandate=signed_mandate,
    )
    assert decision.reason_code == RejectReason.CHAIN_DENIED


def test_reject_oversize(live_config, signed_mandate, portfolio_state):
    decision = _gate(live_config).evaluate(
        _intent(size_usd=1000.0),
        market_state=portfolio_state,
        mandate=signed_mandate,
    )
    assert decision.reason_code == RejectReason.OVERSIZE


def test_reject_daily_limit(live_config, signed_mandate):
    decision = _gate(live_config).evaluate(
        _intent(),
        mandate=signed_mandate,
        daily_loss_pct=5.0,
    )
    assert decision.reason_code == RejectReason.DAILY_LIMIT


def test_reject_low_liquidity(live_config, signed_mandate):
    decision = _gate(live_config).evaluate(
        _intent(pool_liquidity_usd=50_000.0),
        mandate=signed_mandate,
    )
    assert decision.reason_code == RejectReason.LOW_LIQUIDITY


def test_reject_slippage(live_config, signed_mandate):
    decision = _gate(live_config).evaluate(
        _intent(slippage_bps=250),
        mandate=signed_mandate,
    )
    assert decision.reason_code == RejectReason.SLIPPAGE


def test_reject_no_mandate_missing(live_config):
    decision = _gate(live_config).evaluate(_intent(), mandate=None, mandate_path=None)
    assert decision.reason_code == RejectReason.NO_MANDATE


def test_reject_no_mandate_invalid_signature(live_config, monkeypatch):
    monkeypatch.setenv("HERMES_TRADER_MANDATE_SECRET", TEST_KEY.decode())
    bad = sign_mandate(WALLET, signing_key=TEST_KEY)
    bad = type(bad)(
        version=bad.version,
        wallet_address=bad.wallet_address,
        signed_at=bad.signed_at,
        expires_at=bad.expires_at,
        signature="deadbeef" * 8,
    )
    decision = _gate(live_config).evaluate(_intent(), mandate=bad)
    assert decision.reason_code == RejectReason.NO_MANDATE


def test_reject_low_confidence(live_config, signed_mandate):
    decision = _gate(live_config).evaluate(
        _intent(confidence=0.2),
        mandate=signed_mandate,
    )
    assert decision.reason_code == RejectReason.LOW_CONFIDENCE


def test_reject_hold_action(live_config, signed_mandate):
    decision = _gate(live_config).evaluate(
        _intent(action="hold"),
        mandate=signed_mandate,
    )
    assert decision.reason_code == RejectReason.HOLD


def test_reject_watch_action(live_config, signed_mandate):
    decision = _gate(live_config).evaluate(
        _intent(action="watch"),
        mandate=signed_mandate,
    )
    assert decision.reason_code == RejectReason.HOLD


def test_reject_invalid_intent_missing_chain(live_config, signed_mandate):
    decision = _gate(live_config).evaluate(
        _intent(chain=""),
        mandate=signed_mandate,
    )
    assert decision.reason_code == RejectReason.INVALID_INTENT


def test_reject_invalid_intent_missing_token(live_config, signed_mandate):
    decision = _gate(live_config).evaluate(
        _intent(token_address=""),
        mandate=signed_mandate,
    )
    assert decision.reason_code == RejectReason.INVALID_INTENT


def test_approve_ethereum_uses_execute_swap(live_config, signed_mandate):
    decision = _gate(live_config).evaluate(
        _intent(chain="ethereum"),
        mandate=signed_mandate,
        portfolio_value_usd_override=10_000,
    )
    assert decision.approved
    assert decision.order.tool == "execute_swap"


def test_portfolio_value_from_market_state():
    state = MarketState(
        chain="base",
        captured_at="2026-07-07T12:00:00+00:00",
        portfolio_tokens=[
            PortfolioToken("base", "USDC", "0x1", 1.0, balance_usd=500.0),
            PortfolioToken("base", "ETH", "0x2", 1.0, balance_usd=1500.0),
        ],
    )
    from hermes_trader.risk.gate import portfolio_value_usd

    assert portfolio_value_usd(state) == 2000.0


def test_gate_decision_to_dict(live_config, signed_mandate):
    decision = _gate(live_config).evaluate(
        _intent(),
        mandate=signed_mandate,
        portfolio_value_usd_override=10_000,
    )
    data = decision.to_dict()
    assert data["approved"] is True
    assert data["order"]["tool"] == "submit_gasless_swap"


def test_reject_expired_mandate(live_config):
    past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    expired = sign_mandate(WALLET, signing_key=TEST_KEY, expires_at=past)
    decision = _gate(live_config).evaluate(
        _intent(),
        mandate=expired,
        now=datetime.now(timezone.utc),
    )
    assert decision.reason_code == RejectReason.NO_MANDATE