"""Abstract broker adapter interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from brokerage.models import BrokerSubmissionResult, TradeIntent


class BrokerAdapter(ABC):
    """Abstract broker adapter interface."""

    @abstractmethod
    def submit_order(self, intent: TradeIntent) -> BrokerSubmissionResult:
        raise NotImplementedError

    @abstractmethod
    def get_order_status(
        self,
        order_id: str,
        *,
        account_mode: str | None = None,
        expected_quantity: int | None = None,
    ):
        raise NotImplementedError

    @abstractmethod
    def cancel_order(self, order_id: str):
        raise NotImplementedError

    def health_check(self) -> dict:
        """Return broker connection health status. Override in subclasses."""
        return {"connected": False, "mode": None}

    def get_positions(self, *, account_mode: str | None = None, account: str | None = None) -> list[dict]:
        """Return current account positions as a list of dicts.

        Each dict has keys: account, symbol, position (float), avg_cost (float), account_mode.
        Override in subclasses. Returns empty list by default.
        """
        return []

    def disconnect(self) -> None:
        """Graceful disconnect. Override in subclasses."""
        pass
