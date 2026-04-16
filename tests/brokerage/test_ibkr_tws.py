"""Tests for the IBKR TWS/IB Gateway broker adapter."""
import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from brokerage.brokers.ibkr_tws import IBKRTwsBrokerAdapter
from brokerage.config import BrokerageSettings
from brokerage.models import TradeIntent


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


def test_paper_mode_uses_ib_gateway_paper_port_by_default():
    adapter = IBKRTwsBrokerAdapter(BrokerageSettings())

    assert adapter._select_port("paper") == 4002


def test_live_mode_uses_ib_gateway_live_port_by_default():
    adapter = IBKRTwsBrokerAdapter(BrokerageSettings())

    assert adapter._select_port("live") == 4001


def test_build_order_maps_market_order_correctly():
    adapter = IBKRTwsBrokerAdapter(BrokerageSettings())
    order = adapter._build_order(_make_intent())

    assert order.action == "BUY"
    assert order.totalQuantity == 10
    assert order.orderType == "MKT"


def test_build_order_maps_limit_order_correctly():
    adapter = IBKRTwsBrokerAdapter(BrokerageSettings())
    order = adapter._build_order(_make_intent(order_type="LIMIT", limit_price=123.45))

    assert order.action == "BUY"
    assert order.totalQuantity == 10
    assert order.orderType == "LMT"
    assert order.lmtPrice == 123.45


def test_submit_order_returns_submission_result_from_mocked_ib():
    adapter = IBKRTwsBrokerAdapter(BrokerageSettings())
    trade = SimpleNamespace(orderStatus=SimpleNamespace(status="Submitted"), order=SimpleNamespace(orderId=1234))

    with patch.object(adapter, "_ensure_connected") as mock_connect, patch.object(adapter, "_qualify_contract") as mock_contract:
        adapter._ib = MagicMock()
        mock_connect.return_value = None
        mock_contract.return_value = SimpleNamespace(conId=1)
        adapter._ib.placeOrder.return_value = trade

        result = adapter.submit_order(_make_intent())

    assert result.accepted is True
    assert result.broker_order_id == "1234"
    assert result.broker_status == "Submitted"


def test_submit_order_rejects_non_stock_assets_defensively():
    adapter = IBKRTwsBrokerAdapter(BrokerageSettings())

    bad_intent = TradeIntent.model_construct(
        request_id="req-2",
        account_mode="paper",
        symbol="AAPL",
        side="BUY",
        quantity=1,
        order_type="MARKET",
        asset_class="option",
        limit_price=None,
        status="pending_confirmation",
    )

    with pytest.raises(ValueError):
        adapter.submit_order(bad_intent)


def test_submit_order_creates_event_loop_when_called_from_worker_thread():
    adapter = IBKRTwsBrokerAdapter(BrokerageSettings())

    class EventLoopCheckingIB:
        def __init__(self):
            self.connected = False

        def isConnected(self):
            return self.connected

        def connect(self, host, port, clientId=0):
            asyncio.get_event_loop()
            self.connected = True
            return True

        def disconnect(self):
            self.connected = False

        def qualifyContracts(self, contract):
            asyncio.get_event_loop()
            return [contract]

        def placeOrder(self, contract, order):
            asyncio.get_event_loop()
            return SimpleNamespace(
                orderStatus=SimpleNamespace(status="Submitted"),
                order=SimpleNamespace(orderId=1234),
            )

    result: list = []
    errors: list[Exception] = []

    def worker() -> None:
        try:
            result.append(adapter.submit_order(_make_intent()))
        except Exception as exc:  # pragma: no cover - exercised in the red phase
            errors.append(exc)

    with patch("brokerage.brokers.ibkr_tws.IB", EventLoopCheckingIB):
        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()

    assert errors == []
    assert result[0].accepted is True
    assert result[0].broker_order_id == "1234"
