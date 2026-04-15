"""Tests for brokerage configuration and domain models."""

import pytest
from pydantic import ValidationError

from brokerage.config import BrokerageSettings
from brokerage.models import TradeIntent


def test_brokerage_settings_defaults_to_paper_mode_and_local_service():
    settings = BrokerageSettings()
    assert settings.service_url == "http://127.0.0.1:8787"
    assert settings.default_account_mode == "paper"
    assert settings.confirmation_ttl_seconds == 120


def test_trade_intent_normalizes_symbol_and_side():
    intent = TradeIntent(
        request_id="r1",
        account_mode="paper",
        symbol="aapl",
        side="buy",
        quantity=10,
        order_type="market",
        asset_class="stock",
    )
    assert intent.symbol == "AAPL"
    assert intent.side == "BUY"
    assert intent.order_type == "MARKET"


def test_trade_intent_requires_limit_price_for_limit_orders():
    with pytest.raises(ValidationError):
        TradeIntent(
            request_id="r1",
            account_mode="paper",
            symbol="AAPL",
            side="BUY",
            quantity=10,
            order_type="LIMIT",
            asset_class="stock",
        )


def test_trade_intent_rejects_limit_price_for_market_orders():
    with pytest.raises(ValidationError):
        TradeIntent(
            request_id="r1",
            account_mode="paper",
            symbol="AAPL",
            side="BUY",
            quantity=10,
            order_type="MARKET",
            asset_class="stock",
            limit_price=200.0,
        )


def test_trade_intent_supports_allowed_status_values():
    intent = TradeIntent(
        request_id="r1",
        account_mode="paper",
        symbol="AAPL",
        side="BUY",
        quantity=10,
        order_type="MARKET",
        asset_class="stock",
        status="pending_confirmation",
    )
    assert intent.status == "pending_confirmation"
