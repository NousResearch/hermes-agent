"""Tests for deterministic brokerage policy checks."""

from datetime import datetime, timedelta, timezone

import pytest

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


# --- Paper mode validation ---

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


def test_validate_new_intent_uses_stop_price_for_paper_stop_notional_estimate():
    settings = BrokerageSettings(paper_max_notional=500.0)
    policy = BrokeragePolicy(settings)

    decision = policy.validate_new_intent(
        _make_intent(side="SELL", quantity=5, order_type="STOP", stop_price=150.0),
        market_snapshot={"last_price": 50.0},
    )

    assert decision.allowed is False
    assert "paper_max_notional" in decision.reason


# --- Live mode validation ---

def test_validate_new_intent_blocks_live_when_live_disabled():
    settings = BrokerageSettings(live_enabled=False)
    policy = BrokeragePolicy(settings)

    decision = policy.validate_new_intent(
        _make_intent(account_mode="live", quantity=1),
        market_snapshot={"last_price": 100.0},
    )

    assert decision.allowed is False
    assert "live trading is disabled" in decision.reason.lower()


def test_validate_new_intent_allows_live_when_live_enabled():
    settings = BrokerageSettings(live_enabled=True)
    policy = BrokeragePolicy(settings)

    decision = policy.validate_new_intent(
        _make_intent(account_mode="live", quantity=1),
        market_snapshot={"last_price": 100.0},
    )

    assert decision.allowed is True


def test_validate_new_intent_rejects_live_trade_over_live_share_cap():
    settings = BrokerageSettings(live_enabled=True, live_max_shares=3)
    policy = BrokeragePolicy(settings)

    decision = policy.validate_new_intent(
        _make_intent(account_mode="live", quantity=5),
        market_snapshot={"last_price": 100.0},
    )

    assert decision.allowed is False
    assert "live_max_shares" in decision.reason


def test_validate_new_intent_rejects_live_trade_over_live_notional_cap():
    settings = BrokerageSettings(live_enabled=True, live_max_shares=100, live_max_notional=500.0)
    policy = BrokeragePolicy(settings)

    decision = policy.validate_new_intent(
        _make_intent(account_mode="live", quantity=10),
        market_snapshot={"last_price": 100.0},
    )

    assert decision.allowed is False
    assert "live_max_notional" in decision.reason


def test_live_caps_are_independent_of_paper_caps():
    """Paper allows 25 shares; live allows 5 -- each enforced independently."""
    settings = BrokerageSettings(live_enabled=True, paper_max_shares=25, live_max_shares=5)
    policy = BrokeragePolicy(settings)

    # 10 shares OK for paper
    assert policy.validate_new_intent(
        _make_intent(quantity=10), market_snapshot={"last_price": 50.0}
    ).allowed is True

    # 10 shares TOO MANY for live
    assert policy.validate_new_intent(
        _make_intent(account_mode="live", quantity=10),
        market_snapshot={"last_price": 50.0},
    ).allowed is False


def test_validate_new_intent_blocks_disallowed_asset_class():
    """When asset_class is expanded beyond 'stock', the policy gate must reject
    classes not in the allowed_asset_classes config. We test this by creating a
    valid stock intent and then modifying its asset_class post-construction,
    simulating a future expansion of the model."""
    settings = BrokerageSettings(allowed_asset_classes=("stock",))
    policy = BrokeragePolicy(settings)

    intent = _make_intent()
    intent.asset_class = "option"  # bypass Pydantic validation for policy test

    decision = policy.validate_new_intent(intent)

    assert decision.allowed is False
    assert "asset class" in decision.reason.lower()


def test_validate_new_intent_blocks_blocked_symbol():
    settings = BrokerageSettings(blocked_symbols=("SPCE",))
    policy = BrokeragePolicy(settings)

    decision = policy.validate_new_intent(_make_intent(symbol="SPCE"))

    assert decision.allowed is False
    assert "blocked" in decision.reason.lower()


def test_validate_new_intent_blocks_symbol_not_in_allowed_list():
    settings = BrokerageSettings(allowed_symbols=("AAPL", "MSFT"))
    policy = BrokeragePolicy(settings)

    decision = policy.validate_new_intent(_make_intent(symbol="TSLA"))

    assert decision.allowed is False
    assert "not in the allowed_symbols" in decision.reason


# --- Paper confirmation validation ---

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


def test_validate_confirmation_rejects_wrong_code_for_paper():
    policy = BrokeragePolicy(BrokerageSettings())
    decision = policy.validate_confirmation(
        _make_intent(),
        confirmation_text="CONFIRM T-WRONG",
        confirmation_code="T-82K4",
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=1),
    )

    assert decision.allowed is False
    assert "does not match" in decision.reason.lower()


def test_validate_confirmation_rejects_bare_code_without_confirm_prefix():
    policy = BrokeragePolicy(BrokerageSettings())
    decision = policy.validate_confirmation(
        _make_intent(),
        confirmation_text="T-82K4",
        confirmation_code="T-82K4",
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=1),
    )

    assert decision.allowed is False
    assert "does not match" in decision.reason.lower()


# --- Live confirmation validation ---

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


def test_validate_confirmation_rejects_paper_phrase_for_live():
    policy = BrokeragePolicy(BrokerageSettings(live_enabled=True))
    decision = policy.validate_confirmation(
        _make_intent(account_mode="live", quantity=10),
        confirmation_text="CONFIRM T-82K4",
        confirmation_code="T-82K4",
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=1),
    )

    assert decision.allowed is False
    assert "does not match" in decision.reason.lower()


def test_validate_confirmation_rejects_live_phrase_with_wrong_side():
    policy = BrokeragePolicy(BrokerageSettings(live_enabled=True))
    decision = policy.validate_confirmation(
        _make_intent(account_mode="live", side="BUY", quantity=10, symbol="AAPL"),
        confirmation_text="CONFIRM LIVE SELL 10 AAPL T-82K4",
        confirmation_code="T-82K4",
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=1),
    )

    assert decision.allowed is False


def test_validate_confirmation_rejects_live_phrase_with_wrong_quantity():
    policy = BrokeragePolicy(BrokerageSettings(live_enabled=True))
    decision = policy.validate_confirmation(
        _make_intent(account_mode="live", side="BUY", quantity=10, symbol="AAPL"),
        confirmation_text="CONFIRM LIVE BUY 5 AAPL T-82K4",
        confirmation_code="T-82K4",
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=1),
    )

    assert decision.allowed is False


def test_validate_confirmation_rejects_live_phrase_with_wrong_symbol():
    policy = BrokeragePolicy(BrokerageSettings(live_enabled=True))
    decision = policy.validate_confirmation(
        _make_intent(account_mode="live", side="BUY", quantity=10, symbol="AAPL"),
        confirmation_text="CONFIRM LIVE BUY 10 MSFT T-82K4",
        confirmation_code="T-82K4",
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=1),
    )

    assert decision.allowed is False


def test_validate_confirmation_with_no_expiry_always_allowed():
    """If expires_at is None, the confirmation never expires."""
    policy = BrokeragePolicy(BrokerageSettings())
    decision = policy.validate_confirmation(
        _make_intent(),
        confirmation_text="CONFIRM T-82K4",
        confirmation_code="T-82K4",
        expires_at=None,
    )

    assert decision.allowed is True
