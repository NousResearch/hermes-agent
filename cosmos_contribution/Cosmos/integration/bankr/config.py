"""
Bankr Configuration Module.

Handles API keys, spending limits, and chain preferences.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List
from decimal import Decimal
import logging

logger = logging.getLogger(__name__)


@dataclass
class BankrConfig:
    """Configuration for Bankr integration."""

    # API Authentication
    api_key: str = ""

    # Base URL (inferred from documentation)
    base_url: str = "https://api.bankr.bot"

    # Default chain for operations (supports: base, ethereum, solana, polygon)
    default_chain: str = "base"

    # Supported chains
    supported_chains: List[str] = field(default_factory=lambda: [
        "base", "ethereum", "solana", "polygon"
    ])

    # Trading limits
    trading_enabled: bool = True
    max_trade_usd: Decimal = Decimal("1000.00")
    daily_limit_usd: Decimal = Decimal("5000.00")

    # Polymarket
    polymarket_enabled: bool = True
    max_bet_usd: Decimal = Decimal("100.00")

    # x402 settings
    x402_enabled: bool = True
    x402_cost_per_request: Decimal = Decimal("0.01")  # USDC

    # Timeouts
    request_timeout: float = 30.0
    job_poll_interval: float = 2.0
    job_timeout: float = 60.0

    @classmethod
    def from_env(cls) -> "BankrConfig":
        """Create config from environment variables."""
        return cls(
            api_key=os.environ.get("BANKR_API_KEY", ""),
            base_url=os.environ.get("BANKR_BASE_URL", "https://api.bankr.bot"),
            default_chain=os.environ.get("BANKR_DEFAULT_CHAIN", "base"),
            trading_enabled=os.environ.get("BANKR_TRADING_ENABLED", "true").lower() == "true",
            max_trade_usd=Decimal(os.environ.get("BANKR_MAX_TRADE_USD", "1000.00")),
            polymarket_enabled=os.environ.get("BANKR_POLYMARKET_ENABLED", "true").lower() == "true",
            max_bet_usd=Decimal(os.environ.get("BANKR_MAX_BET_USD", "100.00")),
        )

    def validate(self) -> List[str]:
        """Validate configuration. Returns list of errors."""
        errors = []

        if not self.api_key:
            errors.append("BANKR_API_KEY not set")

        if self.default_chain not in self.supported_chains:
            errors.append(f"Invalid default chain: {self.default_chain}")

        if self.max_trade_usd <= 0:
            errors.append("max_trade_usd must be positive")

        return errors

    def is_valid(self) -> bool:
        """Check if configuration is valid."""
        return len(self.validate()) == 0


# Global config instance
_config: Optional[BankrConfig] = None


def get_bankr_config() -> BankrConfig:
    """Get or create the global Bankr config."""
    global _config
    if _config is None:
        _config = BankrConfig.from_env()
        errors = _config.validate()
        if errors:
            logger.warning(f"Bankr config validation errors: {errors}")
    return _config


def set_bankr_config(config: BankrConfig):
    """Set the global Bankr config."""
    global _config
    _config = config
