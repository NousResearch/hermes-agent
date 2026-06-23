from __future__ import annotations

import os
from dataclasses import dataclass


def _bool_env(value: str | None, default: bool) -> bool:
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return float(raw)


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return int(raw)


@dataclass(frozen=True)
class Settings:
    clob_host: str = "https://clob.polymarket.com"
    gamma_host: str = "https://gamma-api.polymarket.com"
    chain_id: int = 137
    dry_run: bool = True
    private_key: str | None = None
    safe_address: str | None = None
    funder_address: str | None = None
    signature_type: int | None = None
    clob_api_key: str | None = None
    clob_api_secret: str | None = None
    clob_api_passphrase: str | None = None
    crypto_symbol: str = "BTC"
    market_slug: str | None = None
    max_collateral_per_trade: float = 10.0
    min_collateral_balance: float = 25.0
    max_open_positions: int = 1
    min_edge: float = 0.01
    post_only: bool = False
    order_type: str = "GTC"
    poll_interval_seconds: int = 5

    @classmethod
    def from_env(cls) -> "Settings":
        safe_address = os.getenv("SAFE_ADDRESS") or None
        explicit_funder = os.getenv("FUNDER_ADDRESS") or None
        funder_address = explicit_funder or safe_address
        explicit_signature = os.getenv("SIGNATURE_TYPE")
        signature_type = int(explicit_signature) if explicit_signature else (2 if safe_address else None)
        return cls(
            clob_host=os.getenv("CLOB_HOST", cls.clob_host),
            gamma_host=os.getenv("GAMMA_HOST", cls.gamma_host),
            chain_id=_int_env("CHAIN_ID", cls.chain_id),
            dry_run=_bool_env(os.getenv("DRY_RUN"), True),
            private_key=os.getenv("PRIVATE_KEY") or None,
            safe_address=safe_address,
            funder_address=funder_address,
            signature_type=signature_type,
            clob_api_key=os.getenv("CLOB_API_KEY") or None,
            clob_api_secret=os.getenv("CLOB_API_SECRET") or None,
            clob_api_passphrase=os.getenv("CLOB_API_PASSPHRASE") or None,
            crypto_symbol=os.getenv("CRYPTO_SYMBOL", cls.crypto_symbol).upper(),
            market_slug=os.getenv("MARKET_SLUG") or None,
            max_collateral_per_trade=_float_env("MAX_COLLATERAL_PER_TRADE", cls.max_collateral_per_trade),
            min_collateral_balance=_float_env("MIN_COLLATERAL_BALANCE", cls.min_collateral_balance),
            max_open_positions=_int_env("MAX_OPEN_POSITIONS", cls.max_open_positions),
            min_edge=_float_env("MIN_EDGE", cls.min_edge),
            post_only=_bool_env(os.getenv("POST_ONLY"), cls.post_only),
            order_type=os.getenv("ORDER_TYPE", cls.order_type),
            poll_interval_seconds=_int_env("POLL_INTERVAL_SECONDS", cls.poll_interval_seconds),
        )

    def validate_for_live(self) -> None:
        if self.dry_run:
            return
        if not self.private_key:
            raise ValueError("PRIVATE_KEY is required when DRY_RUN=false")
        if self.max_collateral_per_trade <= 0:
            raise ValueError("MAX_COLLATERAL_PER_TRADE must be positive")
