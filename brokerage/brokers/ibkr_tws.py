"""IBKR TWS / IB Gateway broker adapter with connection lifecycle management."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from brokerage.brokers.base import BrokerAdapter
from brokerage.config import BrokerageSettings
from brokerage.models import BrokerSubmissionResult, TradeIntent

try:
    from ib_insync import IB, LimitOrder, MarketOrder, Stock
except Exception:  # pragma: no cover - exercised via mocked methods in tests
    IB = None

    class MarketOrder:  # type: ignore[no-redef]
        def __init__(self, action, totalQuantity):
            self.action = action
            self.totalQuantity = totalQuantity
            self.orderType = "MKT"

    class LimitOrder:  # type: ignore[no-redef]
        def __init__(self, action, totalQuantity, lmtPrice):
            self.action = action
            self.totalQuantity = totalQuantity
            self.orderType = "LMT"
            self.lmtPrice = lmtPrice

    class Stock:  # type: ignore[no-redef]
        def __init__(self, symbol, exchange, currency):
            self.symbol = symbol
            self.exchange = exchange
            self.currency = currency


logger = logging.getLogger(__name__)


@dataclass
class _OrderContractBundle:
    contract: object
    order: object


class IBKRTwsBrokerAdapter(BrokerAdapter):
    """IBKR adapter using ib_insync with connection lifecycle management.

    Features:
    - Lazy connection: connects on first order submission
    - Auto-reconnect: detects dropped connections and reconnects
    - Mode-aware: reconnects when switching between paper/live
    - Health check: reports connection status for monitoring
    - Graceful disconnect: clean shutdown support
    """

    def __init__(self, settings: BrokerageSettings, *, host: str = "127.0.0.1"):
        self.settings = settings
        self.host = host
        self._ib: IB | None = None
        self._connected_mode: str | None = None

    # --- Connection state properties ---

    @property
    def is_connected(self) -> bool:
        """Check if currently connected to TWS/Gateway."""
        if self._ib is None:
            return False
        try:
            return self._ib.isConnected()
        except Exception:
            return False

    @property
    def connected_mode(self) -> str | None:
        """Return the account mode of the current connection, or None."""
        if not self.is_connected:
            return None
        return self._connected_mode

    # --- Port selection ---

    def _select_port(self, account_mode: str) -> int:
        """Select the IB Gateway port based on account mode.

        Paper: 4002 (Gateway) / 7497 (TWS)
        Live:  4001 (Gateway) / 7496 (TWS)
        """
        return 4002 if account_mode == "paper" else 4001

    # --- Connection management ---

    def _ensure_thread_event_loop(self):
        """Ensure the current thread has a live asyncio event loop for ib_insync."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop

        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop

    def _ensure_ib(self):
        self._ensure_thread_event_loop()
        if IB is None:
            raise RuntimeError("ib_insync is not installed")
        if self._ib is None:
            self._ib = IB()
        return self._ib

    def _ensure_connected(self, account_mode: str) -> None:
        """Ensure we have an active connection for the given account mode.

        - If not connected at all, connect.
        - If connected but to the wrong mode, reconnect.
        - If already connected to the right mode, do nothing.
        """
        ib = self._ensure_ib()

        # Already connected to the correct mode
        if self.is_connected and self._connected_mode == account_mode:
            return

        # If connected to a different mode or connection dropped, disconnect first
        if self.is_connected:
            logger.info("Disconnecting from %s mode to switch to %s", self._connected_mode, account_mode)
            self._safe_disconnect()

        # Connect to the requested mode
        port = self._select_port(account_mode)
        logger.info("Connecting to IB Gateway at %s:%d (%s mode)", self.host, port, account_mode)
        ib.connect(self.host, port, clientId=0)
        self._connected_mode = account_mode

    def _connect(self, account_mode: str) -> None:
        """Legacy connect method - delegates to _ensure_connected."""
        self._ensure_connected(account_mode)

    def _safe_disconnect(self) -> None:
        """Disconnect without raising errors."""
        if self._ib is not None:
            try:
                self._ib.disconnect()
            except Exception as exc:
                logger.warning("Error during disconnect: %s", exc)
        self._connected_mode = None

    def disconnect(self) -> None:
        """Graceful disconnect - safe to call even if not connected."""
        self._safe_disconnect()

    # --- Health check ---

    def health_check(self) -> dict:
        """Return connection health status for monitoring.

        Returns:
            dict with 'connected' (bool) and 'mode' (str|None)
        """
        connected = self.is_connected

        # If connection dropped, clear the mode
        if not connected and self._connected_mode is not None:
            logger.info("Detected dropped connection (was %s mode)", self._connected_mode)
            self._connected_mode = None

        return {
            "connected": connected,
            "mode": self._connected_mode,
        }

    # --- Order construction ---

    def _build_contract(self, intent: TradeIntent):
        if intent.asset_class != "stock":
            raise ValueError(f"Unsupported asset class: {intent.asset_class}")
        return Stock(intent.symbol, "SMART", "USD")

    def _build_order(self, intent: TradeIntent):
        if intent.order_type == "MARKET":
            return MarketOrder(intent.side, intent.quantity)
        if intent.order_type == "LIMIT":
            return LimitOrder(intent.side, intent.quantity, intent.limit_price)
        raise ValueError(f"Unsupported order type: {intent.order_type}")

    def _qualify_contract(self, contract):
        ib = self._ensure_ib()
        qualified = ib.qualifyContracts(contract)
        return qualified[0] if qualified else contract

    # --- Order submission ---

    def submit_order(self, intent: TradeIntent) -> BrokerSubmissionResult:
        if intent.asset_class != "stock":
            raise ValueError(f"Unsupported asset class: {intent.asset_class}")

        self._ensure_connected(intent.account_mode)
        contract = self._qualify_contract(self._build_contract(intent))
        order = self._build_order(intent)
        trade = self._ib.placeOrder(contract, order)
        order_id = getattr(getattr(trade, "order", None), "orderId", None)
        status = getattr(getattr(trade, "orderStatus", None), "status", None)
        return BrokerSubmissionResult(
            accepted=True,
            broker_order_id=str(order_id) if order_id is not None else None,
            broker_status=status,
        )

    def get_order_status(self, order_id: str):
        raise NotImplementedError("get_order_status not implemented yet")

    def cancel_order(self, order_id: str):
        raise NotImplementedError("cancel_order not implemented yet")
