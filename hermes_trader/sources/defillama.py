"""DeFiLlama yields API — free pool discovery fallback (no API key)."""

from __future__ import annotations

from typing import Any, Callable, Optional

import httpx

YIELDS_POOLS_URL = "https://yields.llama.fi/pools"
DEX_OVERVIEW_URL = "https://api.llama.fi/overview/dexs/{chain}"

CHAIN_TO_LLAMA: dict[str, str] = {
    "base": "Base",
    "ethereum": "Ethereum",
    "arbitrum": "Arbitrum",
    "optimism": "Optimism",
    "polygon": "Polygon",
    "bsc": "BSC",
    "avalanche": "Avalanche",
}

QUOTE_STABLE_SYMBOLS = frozenset(
    {
        "USDC",
        "USDBC",
        "USDB",
        "DAI",
        "USDT",
        "USD₮",
        "USDC.E",
        "CBBTC",
        "USDE",
        "MSUSD",
        "EURC",
    }
)

HttpGet = Callable[[str], Any]


def _llama_chain_name(chain: str) -> str:
    key = chain.strip().lower()
    return CHAIN_TO_LLAMA.get(key, key.capitalize())


def _default_http_get(url: str, *, timeout: float = 60.0) -> Any:
    response = httpx.get(
        url,
        timeout=timeout,
        headers={"Accept-Encoding": "gzip, deflate"},
    )
    response.raise_for_status()
    return response.json()


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _split_symbol(symbol: str) -> tuple[str, str]:
    text = (symbol or "").strip()
    if "-" in text:
        base, quote = text.split("-", 1)
        return base.strip() or "?", quote.strip() or "?"
    return text or "?", "?"


def _pick_trade_token(symbol: str, underlying: list[Any]) -> str:
    addresses = [str(addr).strip() for addr in underlying if addr]
    if not addresses:
        return ""
    base_sym, quote_sym = _split_symbol(symbol)
    base_is_quote = base_sym.upper() in QUOTE_STABLE_SYMBOLS
    quote_is_quote = quote_sym.upper() in QUOTE_STABLE_SYMBOLS
    if not base_is_quote:
        return addresses[0]
    if not quote_is_quote and len(addresses) > 1:
        return addresses[1]
    return addresses[0]


def normalize_yields_pool(entry: dict[str, Any], *, chain: str) -> Optional[dict[str, Any]]:
    """Map a DeFiLlama yields row into MCP-compatible pool dict."""
    symbol = str(entry.get("symbol") or "")
    base_sym, quote_sym = _split_symbol(symbol)
    underlying = entry.get("underlyingTokens") or []
    trade_token = _pick_trade_token(symbol, underlying)
    pool_id = str(entry.get("pool") or trade_token or "")
    if not pool_id and not trade_token:
        return None
    liquidity = _coerce_float(entry.get("tvlUsd"))
    volume = _coerce_float(entry.get("volumeUsd1d"))
    return {
        "pool_address": trade_token or pool_id,
        "base_token": {"symbol": base_sym},
        "quote_token": {"symbol": quote_sym},
        "liquidity_usd": liquidity,
        "volume_24h_usd": volume,
        "price_change_24h": _coerce_float(entry.get("apyPct1D")),
        "chain": chain,
        "source": "defillama",
        "project": entry.get("project"),
        "defillama_pool_id": pool_id,
    }


def _filter_chain_pools(data: list[dict[str, Any]], chain: str) -> list[dict[str, Any]]:
    llama_chain = _llama_chain_name(chain)
    return [
        row
        for row in data
        if isinstance(row, dict) and str(row.get("chain") or "") == llama_chain
    ]


def fetch_yields_pools(
    chain: str,
    *,
    http_get: Optional[HttpGet] = None,
) -> list[dict[str, Any]]:
    """Return raw yields rows for *chain* from DeFiLlama."""
    getter = http_get or _default_http_get
    payload = getter(YIELDS_POOLS_URL)
    rows = payload.get("data", payload) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        return []
    return _filter_chain_pools(rows, chain)


def fetch_trending_pools_payload(
    chain: str,
    *,
    limit: int = 25,
    min_liquidity_usd: float = 0.0,
    http_get: Optional[HttpGet] = None,
) -> dict[str, Any]:
    """Trending pools sorted by 24h volume (DeFiLlama yields)."""
    rows = fetch_yields_pools(chain, http_get=http_get)
    ranked = sorted(
        rows,
        key=lambda row: _coerce_float(row.get("volumeUsd1d")) or 0.0,
        reverse=True,
    )
    pools: list[dict[str, Any]] = []
    for row in ranked:
        liq = _coerce_float(row.get("tvlUsd")) or 0.0
        if liq < min_liquidity_usd:
            continue
        normalized = normalize_yields_pool(row, chain=chain)
        if normalized:
            pools.append(normalized)
        if len(pools) >= limit:
            break
    return {"pools": pools, "source": "defillama", "kind": "trending"}


def fetch_new_pools_payload(
    chain: str,
    *,
    limit: int = 25,
    min_liquidity_usd: float = 0.0,
    http_get: Optional[HttpGet] = None,
) -> dict[str, Any]:
    """Recently surfaced pools — low history count with meaningful volume."""
    rows = fetch_yields_pools(chain, http_get=http_get)
    candidates: list[tuple[float, dict[str, Any]]] = []
    for row in rows:
        count = row.get("count")
        try:
            history = int(count) if count is not None else 99
        except (TypeError, ValueError):
            history = 99
        volume = _coerce_float(row.get("volumeUsd1d")) or 0.0
        liq = _coerce_float(row.get("tvlUsd")) or 0.0
        if history > 5 or volume < 10_000 or liq < min_liquidity_usd:
            continue
        score = volume + (_coerce_float(row.get("apyPct1D")) or 0.0) * 1000.0
        candidates.append((score, row))
    candidates.sort(key=lambda item: item[0], reverse=True)
    pools: list[dict[str, Any]] = []
    for _score, row in candidates:
        normalized = normalize_yields_pool(row, chain=chain)
        if normalized:
            pools.append(normalized)
        if len(pools) >= limit:
            break
    return {"pools": pools, "source": "defillama", "kind": "new"}