"""Domain models for the brokerage subsystem."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

AccountMode = Literal["paper", "live"]
TradeSide = Literal["BUY", "SELL"]
OrderType = Literal["MARKET", "LIMIT", "STOP"]
AssetClass = Literal["stock"]
TradeStatus = Literal[
    "pending_confirmation",
    "confirmed",
    "submitted",
    "submission_error",
    "filled",
    "rejected",
    "cancelled",
    "expired",
]

# Legal state transitions: from_status -> set of allowed to_statuses
LEGAL_TRANSITIONS: dict[str, set[str]] = {
    "pending_confirmation": {"confirmed", "cancelled", "expired"},
    "confirmed": {"submitted", "submission_error", "rejected"},
    "submitted": {"filled", "rejected", "cancelled"},
    "submission_error": set(),  # dead-end for now; retry path can be added later
    "filled": set(),            # terminal
    "rejected": set(),          # terminal
    "cancelled": set(),         # terminal
    "expired": set(),           # terminal
}


def is_legal_transition(from_status: str, to_status: str) -> bool:
    """Check whether a state transition is allowed by the transition graph."""
    allowed = LEGAL_TRANSITIONS.get(from_status)
    if allowed is None:
        return False
    return to_status in allowed


class TradeIntent(BaseModel):
    """A normalized request to place a trade after user confirmation."""

    request_id: str
    account_mode: AccountMode
    broker_account: str | None = None
    symbol: str
    side: TradeSide | str
    quantity: int = Field(ge=1)
    order_type: OrderType | str
    asset_class: AssetClass = "stock"
    limit_price: float | None = Field(default=None, gt=0)
    stop_price: float | None = Field(default=None, gt=0)
    status: TradeStatus = "pending_confirmation"

    @field_validator("symbol")
    @classmethod
    def _normalize_symbol(cls, value: str) -> str:
        value = value.strip().upper()
        if not value:
            raise ValueError("symbol cannot be empty")
        return value

    @field_validator("broker_account")
    @classmethod
    def _normalize_broker_account(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip().upper()
        return value or None

    @field_validator("side", mode="before")
    @classmethod
    def _normalize_side(cls, value: str) -> str:
        value = str(value).strip().upper()
        if value not in {"BUY", "SELL"}:
            raise ValueError("side must be BUY or SELL")
        return value

    @field_validator("order_type", mode="before")
    @classmethod
    def _normalize_order_type(cls, value: str) -> str:
        value = str(value).strip().upper()
        if value not in {"MARKET", "LIMIT", "STOP"}:
            raise ValueError("order_type must be MARKET, LIMIT, or STOP")
        return value

    @model_validator(mode="after")
    def _validate_limit_price_rules(self) -> "TradeIntent":
        if self.order_type == "LIMIT" and self.limit_price is None:
            raise ValueError("limit_price is required for LIMIT orders")
        if self.order_type != "LIMIT" and self.limit_price is not None:
            raise ValueError("limit_price is only allowed for LIMIT orders")
        if self.order_type == "STOP" and self.stop_price is None:
            raise ValueError("stop_price is required for STOP orders")
        if self.order_type != "STOP" and self.stop_price is not None:
            raise ValueError("stop_price is only allowed for STOP orders")
        return self


class TradeConfirmation(BaseModel):
    intent_id: str
    confirmation_code: str


class BrokerSubmissionResult(BaseModel):
    accepted: bool
    broker_order_id: str | None = None
    broker_status: str | None = None
    detail: str | None = None


class TradeEvent(BaseModel):
    intent_id: str
    event_type: str
    detail: str | None = None
