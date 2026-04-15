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
