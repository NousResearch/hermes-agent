"""Tests for IBKR TWS adapter connection lifecycle management."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from brokerage.brokers.ibkr_tws import IBKRTwsBrokerAdapter
from brokerage.config import BrokerageSettings
from brokerage.models import TradeIntent


def _make_intent(**overrides) -> TradeIntent:
    data = {
        "request_id": "test-1",
        "account_mode": "paper",
        "symbol": "AAPL",
        "side": "BUY",
        "quantity": 1,
        "order_type": "MARKET",
        "asset_class": "stock",
    }
    data.update(overrides)
    return TradeIntent(**data)


@pytest.fixture
def settings():
    return BrokerageSettings()


@pytest.fixture
def adapter(settings):
    return IBKRTwsBrokerAdapter(settings)


def _make_connected_ib(adapter, mode="paper", connected=True):
    """Set up a mock IB instance on the adapter."""
    mock_ib = MagicMock()
    adapter._ib = mock_ib
    mock_ib.isConnected.return_value = connected
    adapter._connected_mode = mode if connected else None
    return mock_ib


def _setup_ib_for_submit(mock_ib, order_id=1, status="Submitted"):
    """Configure a mock IB to handle a full submit_order call."""
    mock_ib.qualifyContracts.return_value = [MagicMock()]
    mock_trade = MagicMock()
    mock_trade.order = MagicMock(orderId=order_id)
    mock_trade.orderStatus = MagicMock(status=status)
    mock_ib.placeOrder.return_value = mock_trade
    return mock_ib


# --- Connection state ---

def test_adapter_starts_disconnected(adapter):
    assert adapter.is_connected is False


def test_adapter_tracks_account_mode_on_connect(adapter):
    _make_connected_ib(adapter, mode="paper", connected=True)
    assert adapter.connected_mode == "paper"


def test_adapter_connected_mode_is_none_when_not_connected(adapter):
    assert adapter.connected_mode is None


# --- Reconnection logic ---

def test_submit_order_reconnects_if_disconnected():
    """If the IB connection drops between orders, submit_order should reconnect."""
    settings = BrokerageSettings()
    adapter = IBKRTwsBrokerAdapter(settings)
    mock_ib = _make_connected_ib(adapter, mode="paper", connected=False)
    _setup_ib_for_submit(mock_ib)

    # Patch _ensure_ib to return our mock
    with patch.object(adapter, "_ensure_ib", return_value=mock_ib):
        result = adapter.submit_order(_make_intent())

    mock_ib.connect.assert_called_once()
    assert result.accepted is True


def test_submit_order_skips_reconnect_if_already_connected():
    """If already connected, no reconnection needed."""
    settings = BrokerageSettings()
    adapter = IBKRTwsBrokerAdapter(settings)
    mock_ib = _make_connected_ib(adapter, mode="paper", connected=True)
    _setup_ib_for_submit(mock_ib)

    with patch.object(adapter, "_ensure_ib", return_value=mock_ib):
        adapter.submit_order(_make_intent())

    # Should NOT have called connect since already connected
    mock_ib.connect.assert_not_called()


# --- Disconnect ---

def test_disconnect_closes_connection(adapter):
    mock_ib = _make_connected_ib(adapter, mode="paper", connected=True)

    adapter.disconnect()

    mock_ib.disconnect.assert_called_once()
    assert adapter._connected_mode is None


def test_disconnect_is_safe_when_not_connected(adapter):
    # Should not raise
    adapter.disconnect()


# --- Port selection ---

def test_paper_mode_uses_paper_gateway_port():
    settings = BrokerageSettings()
    adapter = IBKRTwsBrokerAdapter(settings)
    assert adapter._select_port("paper") == 4002


def test_live_mode_uses_live_gateway_port():
    settings = BrokerageSettings()
    adapter = IBKRTwsBrokerAdapter(settings)
    assert adapter._select_port("live") == 4001


# --- Health check ---

def test_health_check_returns_disconnected_when_no_ib(adapter):
    health = adapter.health_check()
    assert health["connected"] is False
    assert health["mode"] is None


def test_health_check_returns_connected_when_ib_connected():
    settings = BrokerageSettings()
    adapter = IBKRTwsBrokerAdapter(settings)
    _make_connected_ib(adapter, mode="paper", connected=True)

    health = adapter.health_check()

    assert health["connected"] is True
    assert health["mode"] == "paper"


def test_health_check_detects_dropped_connection():
    settings = BrokerageSettings()
    adapter = IBKRTwsBrokerAdapter(settings)
    # Was connected to paper, but now disconnected
    _make_connected_ib(adapter, mode="paper", connected=False)

    health = adapter.health_check()

    assert health["connected"] is False
    assert health["mode"] is None  # mode cleared because connection dropped


# --- Account mode switching ---

def test_switching_account_mode_reconnects():
    """Switching from paper to live should reconnect on the live port."""
    settings = BrokerageSettings()
    adapter = IBKRTwsBrokerAdapter(settings)
    mock_ib = _make_connected_ib(adapter, mode="paper", connected=True)
    _setup_ib_for_submit(mock_ib)

    # When we switch modes, _ensure_connected will disconnect and reconnect
    # First call to isConnected during _ensure_connected check returns True (paper mode)
    # But since mode != "live", it should disconnect and reconnect

    def fake_ensure_ib():
        adapter._ib = mock_ib
        return mock_ib

    with patch.object(adapter, "_ensure_ib", side_effect=fake_ensure_ib):
        adapter.submit_order(_make_intent(account_mode="live"))

    # Should have disconnected from paper and connected to live
    mock_ib.disconnect.assert_called_once()
    mock_ib.connect.assert_called_once_with("127.0.0.1", 4001, clientId=0)


def test_same_mode_does_not_reconnect():
    """Submitting to the same mode should not trigger reconnection."""
    settings = BrokerageSettings()
    adapter = IBKRTwsBrokerAdapter(settings)
    mock_ib = _make_connected_ib(adapter, mode="paper", connected=True)
    _setup_ib_for_submit(mock_ib)

    with patch.object(adapter, "_ensure_ib", return_value=mock_ib):
        adapter.submit_order(_make_intent(account_mode="paper"))

    # No disconnect or connect should have been called
    mock_ib.disconnect.assert_not_called()
    mock_ib.connect.assert_not_called()
