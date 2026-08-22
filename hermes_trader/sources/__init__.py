"""Optional market-data sources when MCP pool discovery is unavailable."""

from hermes_trader.sources.defillama import (
    fetch_trending_pools_payload,
    fetch_new_pools_payload,
)

__all__ = (
    "fetch_trending_pools_payload",
    "fetch_new_pools_payload",
)