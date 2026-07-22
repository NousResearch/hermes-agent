"""Normalize defi-trading-mcp outputs into a unified MarketState."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class PortfolioToken:
    chain: str
    symbol: str
    address: str
    balance: float
    balance_usd: Optional[float] = None
    price_usd: Optional[float] = None


@dataclass
class PoolSnapshot:
    chain: str
    pool_address: str
    base_token_symbol: str
    quote_token_symbol: str
    liquidity_usd: Optional[float] = None
    volume_24h_usd: Optional[float] = None
    price_change_24h_pct: Optional[float] = None
    source: str = "mcp"


@dataclass
class MarketState:
    """Unified perception snapshot for the reasoning layer."""

    chain: str
    captured_at: str
    portfolio_tokens: List[PortfolioToken] = field(default_factory=list)
    trending_pools: List[PoolSnapshot] = field(default_factory=list)
    new_pools: List[PoolSnapshot] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chain": self.chain,
            "captured_at": self.captured_at,
            "portfolio_tokens": [
                {
                    "chain": t.chain,
                    "symbol": t.symbol,
                    "address": t.address,
                    "balance": t.balance,
                    "balance_usd": t.balance_usd,
                    "price_usd": t.price_usd,
                }
                for t in self.portfolio_tokens
            ],
            "trending_pools": [_pool_to_dict(p) for p in self.trending_pools],
            "new_pools": [_pool_to_dict(p) for p in self.new_pools],
        }


def _pool_to_dict(pool: PoolSnapshot) -> dict[str, Any]:
    return {
        "chain": pool.chain,
        "pool_address": pool.pool_address,
        "base_token_symbol": pool.base_token_symbol,
        "quote_token_symbol": pool.quote_token_symbol,
        "liquidity_usd": pool.liquidity_usd,
        "volume_24h_usd": pool.volume_24h_usd,
        "price_change_24h_pct": pool.price_change_24h_pct,
        "source": pool.source,
    }


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_portfolio_entry(entry: dict[str, Any], default_chain: str) -> Optional[PortfolioToken]:
    symbol = (
        entry.get("symbol")
        or entry.get("token_symbol")
        or entry.get("name")
        or ""
    )
    address = (
        entry.get("address")
        or entry.get("token_address")
        or entry.get("contract")
        or ""
    )
    if not symbol and not address:
        return None
    balance = _coerce_float(
        entry.get("balance")
        or entry.get("amount")
        or entry.get("quantity")
    ) or 0.0
    return PortfolioToken(
        chain=str(entry.get("chain") or entry.get("network") or default_chain).lower(),
        symbol=str(symbol),
        address=str(address),
        balance=balance,
        balance_usd=_coerce_float(entry.get("balance_usd") or entry.get("value_usd")),
        price_usd=_coerce_float(entry.get("price_usd") or entry.get("price")),
    )


def _normalize_pool_entry(entry: dict[str, Any], default_chain: str, source: str) -> Optional[PoolSnapshot]:
    pool_address = (
        entry.get("pool_address")
        or entry.get("address")
        or entry.get("id")
        or ""
    )
    if not pool_address:
        return None
    base = entry.get("base_token") or {}
    quote = entry.get("quote_token") or {}
    if isinstance(base, dict):
        base_symbol = base.get("symbol") or entry.get("base_token_symbol") or "?"
    else:
        base_symbol = str(entry.get("base_token_symbol") or base or "?")
    if isinstance(quote, dict):
        quote_symbol = quote.get("symbol") or entry.get("quote_token_symbol") or "?"
    else:
        quote_symbol = str(entry.get("quote_token_symbol") or quote or "?")
    return PoolSnapshot(
        chain=str(entry.get("chain") or entry.get("network") or default_chain).lower(),
        pool_address=str(pool_address),
        base_token_symbol=str(base_symbol),
        quote_token_symbol=str(quote_symbol),
        liquidity_usd=_coerce_float(
            entry.get("liquidity_usd")
            or entry.get("reserve_in_usd")
            or entry.get("liquidity")
        ),
        volume_24h_usd=_coerce_float(
            entry.get("volume_24h_usd")
            or entry.get("volume_usd")
            or entry.get("volume_24h")
        ),
        price_change_24h_pct=_coerce_float(
            entry.get("price_change_24h")
            or entry.get("price_change_percentage_24h")
        ),
        source=source,
    )


def _extract_list(payload: Any, *keys: str) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return []
    for key in keys:
        value = payload.get(key)
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            for nested in ("data", "pools", "items", "results"):
                inner = value.get(nested)
                if isinstance(inner, list):
                    return inner
    return []


def build_market_state(
    *,
    chain: str,
    portfolio_payload: Any = None,
    trending_payload: Any = None,
    new_pools_payload: Any = None,
    captured_at: Optional[str] = None,
) -> MarketState:
    """Build MarketState from raw MCP tool JSON payloads."""
    chain_norm = chain.strip().lower()
    portfolio_tokens: list[PortfolioToken] = []
    for entry in _extract_list(portfolio_payload, "tokens", "portfolio", "balances"):
        if isinstance(entry, dict):
            token = _normalize_portfolio_entry(entry, chain_norm)
            if token:
                portfolio_tokens.append(token)

    trending: list[PoolSnapshot] = []
    for entry in _extract_list(trending_payload, "pools", "trending", "data"):
        if isinstance(entry, dict):
            pool = _normalize_pool_entry(entry, chain_norm, "trending")
            if pool:
                trending.append(pool)

    new_pools: list[PoolSnapshot] = []
    for entry in _extract_list(new_pools_payload, "pools", "new_pools", "data"):
        if isinstance(entry, dict):
            pool = _normalize_pool_entry(entry, chain_norm, "new")
            if pool:
                new_pools.append(pool)

    return MarketState(
        chain=chain_norm,
        captured_at=captured_at or _utc_now_iso(),
        portfolio_tokens=portfolio_tokens,
        trending_pools=trending,
        new_pools=new_pools,
        raw={
            "portfolio": portfolio_payload,
            "trending": trending_payload,
            "new_pools": new_pools_payload,
        },
    )