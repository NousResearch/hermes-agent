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
    trending = _normalize_mcp_payload(
        mcp_call(server, "get_trending_pools", chain_args)
    )
    new_pools = _normalize_mcp_payload(
        mcp_call(server, "get_new_pools", {**chain_args, "hours": 24})
    )

    return build_market_state(
        chain=chain_name,
        portfolio_payload=portfolio,
        trending_payload=trending,
        new_pools_payload=new_pools,
    )