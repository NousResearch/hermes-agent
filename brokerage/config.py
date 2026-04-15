"""Configuration models for the brokerage subsystem."""

from typing import Literal

from pydantic import BaseModel, Field


class BrokerageSettings(BaseModel):
    """Runtime settings for the local brokerage service and Hermes integration."""

    enabled: bool = False
    service_url: str = "http://127.0.0.1:8787"
    service_token: str | None = None
    default_account_mode: Literal["paper", "live"] = "paper"
    confirmation_ttl_seconds: int = Field(default=120, ge=1)
    allowed_asset_classes: tuple[str, ...] = ("stock",)
    paper_max_shares: int = Field(default=25, ge=1)
    paper_max_notional: float = Field(default=2000.0, gt=0)
    live_enabled: bool = False
    live_max_shares: int = Field(default=5, ge=1)
    live_max_notional: float = Field(default=1000.0, gt=0)
    allowed_symbols: tuple[str, ...] = ()
    blocked_symbols: tuple[str, ...] = ()
