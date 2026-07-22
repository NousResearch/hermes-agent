"""MarketState summaries and feature hashes for episode retrieval."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from hermes_trader.market_state import MarketState


def build_market_summary(state: MarketState) -> dict[str, Any]:
    """Compact summary used for retrieval and embedding_id."""
    top_trending = state.trending_pools[:3]
    top_new = state.new_pools[:3]
    return {
        "chain": state.chain,
        "captured_at": state.captured_at,
        "portfolio_token_count": len(state.portfolio_tokens),
        "trending_pool_count": len(state.trending_pools),
        "new_pool_count": len(state.new_pools),
        "top_trending": [
            {
                "pool_address": p.pool_address,
                "base": p.base_token_symbol,
                "quote": p.quote_token_symbol,
                "liquidity_usd": p.liquidity_usd,
                "volume_24h_usd": p.volume_24h_usd,
            }
            for p in top_trending
        ],
        "top_new": [
            {
                "pool_address": p.pool_address,
                "base": p.base_token_symbol,
                "liquidity_usd": p.liquidity_usd,
            }
            for p in top_new
        ],
    }


def compute_embedding_id(summary: dict[str, Any]) -> str:
    """Deterministic feature fingerprint (vector embedding hook for P4+)."""
    canonical = json.dumps(summary, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def liquidity_band(liquidity_usd: float | None) -> str:
    if liquidity_usd is None or liquidity_usd <= 0:
        return "unknown"
    if liquidity_usd < 50_000:
        return "micro"
    if liquidity_usd < 150_000:
        return "small"
    if liquidity_usd < 500_000:
        return "mid"
    return "large"