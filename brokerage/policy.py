"""Deterministic policy checks for brokerage intents and confirmations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from brokerage.config import BrokerageSettings
from brokerage.models import TradeIntent


@dataclass
class PolicyDecision:
    allowed: bool
    reason: str | None = None


class BrokeragePolicy:
    """Policy/risk checks that must pass before any broker submission."""

    def __init__(self, settings: BrokerageSettings):
        self.settings = settings

    def validate_new_intent(
        self,
        intent: TradeIntent,
        market_snapshot: dict[str, Any] | None = None,
    ) -> PolicyDecision:
        if intent.asset_class not in self.settings.allowed_asset_classes:
            return PolicyDecision(False, f"asset class '{intent.asset_class}' is not allowed")

        if self.settings.allowed_symbols and intent.symbol not in set(self.settings.allowed_symbols):
            return PolicyDecision(False, f"symbol '{intent.symbol}' is not in the allowed_symbols list")

        if intent.symbol in set(self.settings.blocked_symbols):
            return PolicyDecision(False, f"symbol '{intent.symbol}' is blocked")

        if intent.account_mode == "live" and not self.settings.live_enabled:
            return PolicyDecision(False, "live trading is disabled")

        if intent.account_mode == "paper":
            max_shares = self.settings.paper_max_shares
            max_notional = self.settings.paper_max_notional
        else:
            max_shares = self.settings.live_max_shares
            max_notional = self.settings.live_max_notional

        if intent.quantity > max_shares:
            label = "paper_max_shares" if intent.account_mode == "paper" else "live_max_shares"
            return PolicyDecision(False, f"quantity exceeds {label}")

        estimated_notional = self._estimate_notional(intent, market_snapshot)
        if estimated_notional is not None and estimated_notional > max_notional:
            label = "paper_max_notional" if intent.account_mode == "paper" else "live_max_notional"
            return PolicyDecision(False, f"estimated notional exceeds {label}")

        return PolicyDecision(True)

    def validate_confirmation(
        self,
        intent: TradeIntent,
        *,
        confirmation_text: str,
        confirmation_code: str,
        expires_at: datetime | None = None,
        now: datetime | None = None,
    ) -> PolicyDecision:
        current_time = now or datetime.now(timezone.utc)
        if expires_at is not None and expires_at <= current_time:
            return PolicyDecision(False, "confirmation token is expired")

        normalized = " ".join(confirmation_text.strip().split())
        if intent.account_mode == "live":
            expected = f"CONFIRM LIVE {intent.side} {intent.quantity} {intent.symbol} {confirmation_code}"
        else:
            expected = f"CONFIRM {confirmation_code}"

        if normalized != expected:
            return PolicyDecision(False, "confirmation text does not match required format")

        return PolicyDecision(True)

    @staticmethod
    def _estimate_notional(intent: TradeIntent, market_snapshot: dict[str, Any] | None = None) -> float | None:
        if intent.order_type == "LIMIT" and intent.limit_price is not None:
            return float(intent.quantity) * float(intent.limit_price)

        if market_snapshot and market_snapshot.get("last_price") is not None:
            return float(intent.quantity) * float(market_snapshot["last_price"])

        return None
