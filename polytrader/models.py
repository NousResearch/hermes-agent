from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SelectedMarket:
    slug: str
    title: str
    condition_id: str | None
    up_token_id: str
    down_token_id: str
    end_date_iso: str | None = None

    def token_for_side(self, side: str) -> str:
        normalized = side.strip().lower()
        if normalized == "up":
            return self.up_token_id
        if normalized == "down":
            return self.down_token_id
        raise ValueError(f"unsupported side: {side}")


@dataclass(frozen=True)
class MarketMetadata:
    condition_id: str | None
    token_id: str
    tick_size: float
    neg_risk: bool
    fee_rate_bps: int
    fee_exponent: int


@dataclass(frozen=True)
class OrderBookQuote:
    bid: float | None
    ask: float | None
    bid_size: float | None = None
    ask_size: float | None = None


@dataclass(frozen=True)
class TradeDecision:
    strategy: str
    action: str
    token_id: str | None
    side: str | None
    price: float | None
    collateral_size: float
    edge_after_fees: float | None
    reason: str
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class ExecutionReceipt:
    status: str
    dry_run: bool
    response: dict[str, Any]
