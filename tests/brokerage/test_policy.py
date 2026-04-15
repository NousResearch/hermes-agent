"""Tests for deterministic brokerage policy checks."""

from datetime import datetime, timedelta, timezone

from brokerage.config import BrokerageSettings
from brokerage.models import TradeIntent
from brokerage.policy import BrokeragePolicy


def _make_intent(**overrides) -> TradeIntent:
    data = {
        "request_id": "req-1",
        "account_mode": "paper",
        "symbol": "AAPL",
        "side": "BUY",
        "quantity": 10,
        "order_type": "MARKET",
        "asset_class": "stock",
    }
    data.update(overrides)
    return TradeIntent(**data)


def test_validate_new_intent_allows_valid_paper_trade():
    settings = BrokerageSettings()
    policy = BrokeragePolicy(settings)

    decision = policy.validate_new_intent(_make_intent(), market_snapshot={"last_price": 100.0})

    assert decision.allowed is True
    assert decision.reason is None


def test_validate_new_intent_rejects_trade_over_paper_share_cap():
    settings = BrokerageSettings(paper_max_shares=5)
    policy = BrokeragePolicy(settings)

    decision = policy.validate_new_intent(_make_intent(quantity=10), market_snapshot={"last_price": 100.0})

    assert decision.allowed is False
    assert "paper_max_shares" in decision.reason


def test_validate_new_intent_rejects_trade_over_paper_notional_cap():
    settings = BrokerageSettings(paper_max_notional=500.0)
    policy = BrokeragePolicy(settings)

    decision = policy.validate_new_intent(_make_intent(quantity=10), market_snapshot={"last_price": 100.0})

    assert decision.allowed is False
    assert "paper_max_notional" in decision.reason


def test_validate_new_intent_blocks_live_when_live_disabled():
    settings = BrokerageSettings(live_enabled=False)
    policy = BrokeragePolicy(settings)

    decision = policy.validate_new_intent(
        _make_intent(account_mode="live", quantity=1),
        market_snapshot={"last_price": 100.0},
    )

    assert decision.allowed is False
    assert "live trading is disabled" in decision.reason.lower()


def test_validate_confirmation_accepts_exact_paper_confirmation_token():
    policy = BrokeragePolicy(BrokerageSettings())
    decision = policy.validate_confirmation(
        _make_intent(),
        confirmation_text="CONFIRM T-82K4",
        confirmation_code="T-82K4",
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=1),
    )

    assert decision.allowed is True
    assert decision.reason is None


def test_validate_confirmation_rejects_expired_token():
    policy = BrokeragePolicy(BrokerageSettings())
    decision = policy.validate_confirmation(
        _make_intent(),
        confirmation_text="CONFIRM T-82K4",
        confirmation_code="T-82K4",
        expires_at=datetime.now(timezone.utc) - timedelta(seconds=1),
    )

    assert decision.allowed is False
    assert "expired" in decision.reason.lower()


def test_validate_confirmation_requires_stronger_live_phrase():
    policy = BrokeragePolicy(BrokerageSettings(live_enabled=True))
    decision = policy.validate_confirmation(
        _make_intent(account_mode="live", quantity=10),
        confirmation_text="CONFIRM LIVE BUY 10 AAPL T-82K4",
        confirmation_code="T-82K4",
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=1),
    )

    assert decision.allowed is True
    assert decision.reason is None
