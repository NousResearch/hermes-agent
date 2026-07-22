"""Layer 1 — MCP perception into MarketState."""

from __future__ import annotations

import json
from typing import Any, Callable, Optional

from hermes_trader.config import TraderConfig
from hermes_trader.market_state import MarketState, build_market_state

McpCallFn = Callable[[str, str, dict[str, Any]], Any]

READ_TOOLS_PERCEIVE = (
    "get_portfolio_tokens",
    "get_trending_pools",
    "get_new_pools",
)


def _normalize_mcp_payload(result: Any) -> Any:
    if isinstance(result, dict):
        if result.get("error"):
            return result
        inner = result.get("result")
        if inner is not None:
            if isinstance(inner, str):
                text = inner.strip()
                if text:
                    try:
                        return json.loads(text)
                    except json.JSONDecodeError:
                        return {"raw": text}
            elif isinstance(inner, (dict, list)):
                return inner
        return result
    if isinstance(result, str):
        text = result.strip()
        if not text:
            return {}
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"raw": text}
    return result


def _pool_entries(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return []
    for key in ("pools", "trending", "new_pools", "data"):
        value = payload.get(key)
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            for nested in ("pools", "items", "results", "data"):
                inner = value.get(nested)
                if isinstance(inner, list):
                    return inner
    return []


def _mcp_pool_payload_usable(payload: Any) -> bool:
    if not payload or not isinstance(payload, dict):
        return False
    if payload.get("error"):
        return False
    return len(_pool_entries(payload)) > 0


def _use_defillama_fallback(config: TraderConfig) -> bool:
    mode = (config.pool_data_fallback or "auto").strip().lower()
    if mode in ("none", "mcp", "off", "false", "0"):
        return False
    return mode in ("auto", "defillama", "on", "true", "1")


def _defillama_trending_payload(config: TraderConfig, chain: str) -> dict[str, Any]:
    from hermes_trader.sources.defillama import fetch_trending_pools_payload

    return fetch_trending_pools_payload(
        chain,
        min_liquidity_usd=config.min_pool_liquidity_usd,
    )


def _defillama_new_pools_payload(config: TraderConfig, chain: str) -> dict[str, Any]:
    from hermes_trader.sources.defillama import fetch_new_pools_payload

    return fetch_new_pools_payload(
        chain,
        min_liquidity_usd=max(10_000.0, config.min_pool_liquidity_usd * 0.1),
    )


def _resolve_pool_payload(
    config: TraderConfig,
    chain: str,
    *,
    mcp_payload: Any,
    kind: str,
) -> Any:
    fallback_mode = (config.pool_data_fallback or "auto").strip().lower()
    if fallback_mode == "defillama":
        if kind == "trending":
            return _defillama_trending_payload(config, chain)
        return _defillama_new_pools_payload(config, chain)
    if _mcp_pool_payload_usable(mcp_payload):
        return mcp_payload
    if _use_defillama_fallback(config):
        if kind == "trending":
            return _defillama_trending_payload(config, chain)
        return _defillama_new_pools_payload(config, chain)
    return mcp_payload


def perceive_market(
    config: TraderConfig,
    mcp_call: McpCallFn,
    *,
    chain: Optional[str] = None,
) -> MarketState:
    """Fetch portfolio and pool intel via read-only MCP tools."""
    chain_name = (chain or config.primary_chain).strip().lower()
    server = config.mcp_server_name
    chain_args = {"chain": chain_name}

    portfolio = _normalize_mcp_payload(
        mcp_call(server, "get_portfolio_tokens", chain_args)
    )
    trending_raw = _normalize_mcp_payload(
        mcp_call(server, "get_trending_pools", chain_args)
    )
    new_pools_raw = _normalize_mcp_payload(
        mcp_call(server, "get_new_pools", {**chain_args, "hours": 24})
    )

    trending = _resolve_pool_payload(
        config, chain_name, mcp_payload=trending_raw, kind="trending"
    )
    new_pools = _resolve_pool_payload(
        config, chain_name, mcp_payload=new_pools_raw, kind="new"
    )

    return build_market_state(
        chain=chain_name,
        portfolio_payload=portfolio,
        trending_payload=trending,
        new_pools_payload=new_pools,
    )