#!/usr/bin/env python3
"""
Crypto Thesis Engine — Market Data Fetcher
==========================================
Fetches live market data from CoinGecko's free API (no API key required).
Uses only Python standard library — zero external dependencies.

Usage:
    python3 fetch_market_data.py --token bitcoin --output json
    python3 fetch_market_data.py --tokens bitcoin,ethereum,solana --output json
    python3 fetch_market_data.py --category layer-1 --top 10 --output json
    python3 fetch_market_data.py --search "arb" --output json
    python3 fetch_market_data.py --resolve ETH --output json

Author: Crypto Thesis Engine Skill
License: MIT
"""

import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"
CACHE_DIR = Path.home() / ".hermes" / "thesis-cache"
CACHE_TTL_SECONDS = 300  # 5 minutes
MAX_RETRIES = 3
RETRY_DELAY_BASE = 2  # exponential backoff base in seconds
USER_AGENT = "HermesAgent-CryptoThesisEngine/1.0"

# Popular symbol → CoinGecko ID mapping for fast resolution
SYMBOL_MAP = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "BNB": "binancecoin",
    "XRP": "ripple",
    "ADA": "cardano",
    "DOGE": "dogecoin",
    "AVAX": "avalanche-2",
    "DOT": "polkadot",
    "MATIC": "matic-network",
    "POL": "matic-network",
    "LINK": "chainlink",
    "UNI": "uniswap",
    "ATOM": "cosmos",
    "LTC": "litecoin",
    "FIL": "filecoin",
    "APT": "aptos",
    "ARB": "arbitrum",
    "OP": "optimism",
    "SUI": "sui",
    "SEI": "sei-network",
    "TIA": "celestia",
    "INJ": "injective-protocol",
    "FET": "fetch-ai",
    "RNDR": "render-token",
    "NEAR": "near",
    "ICP": "internet-computer",
    "STX": "blockstack",
    "IMX": "immutable-x",
    "MKR": "maker",
    "AAVE": "aave",
    "SNX": "havven",
    "CRV": "curve-dao-token",
    "LDO": "lido-dao",
    "RPL": "rocket-pool",
    "GMX": "gmx",
    "PENDLE": "pendle",
    "ENA": "ethena",
    "EIGEN": "eigenlayer",
    "JUP": "jupiter-exchange-solana",
    "JTO": "jito-governance-token",
    "WIF": "dogwifcoin",
    "PEPE": "pepe",
    "BONK": "bonk",
    "FLOKI": "floki",
    "SHIB": "shiba-inu",
    "TON": "the-open-network",
    "TRX": "tron",
    "ALGO": "algorand",
    "VET": "vechain",
    "HBAR": "hedera-hashgraph",
    "FTM": "fantom",
    "MANA": "decentraland",
    "SAND": "the-sandbox",
    "AXS": "axie-infinity",
    "GALA": "gala",
    "ENS": "ethereum-name-service",
    "GRT": "the-graph",
    "AR": "arweave",
    "ROSE": "oasis-network",
    "OSMO": "osmosis",
    "RUNE": "thorchain",
    "CAKE": "pancakeswap-token",
    "SUSHI": "sushi",
    "COMP": "compound-governance-token",
    "YFI": "yearn-finance",
    "BAL": "balancer",
    "1INCH": "1inch",
    "DYDX": "dydx",
    "ZRO": "layerzero",
    "W": "wormhole",
    "STRK": "starknet",
    "ZK": "zksync",
    "BLAST": "blast",
    "MODE": "mode",
    "MANTA": "manta-network",
    "DYM": "dymension",
    "PYTH": "pyth-network",
    "TAO": "bittensor",
    "ONDO": "ondo-finance",
    "ETHFI": "ether-fi",
}

# Category mapping for scan command
CATEGORY_MAP = {
    "layer1": "layer-1",
    "l1": "layer-1",
    "layer2": "layer-2",
    "l2": "layer-2",
    "defi": "decentralized-finance-defi",
    "ai": "artificial-intelligence",
    "gaming": "gaming",
    "rwa": "real-world-assets-rwa",
    "meme": "meme-token",
    "nft": "non-fungible-tokens-nft",
    "infra": "infrastructure",
    "oracle": "oracle",
    "dex": "decentralized-exchange",
    "lending": "lending-borrowing",
    "yield": "yield-farming",
    "bridge": "bridge-governance-tokens",
    "privacy": "privacy-coins",
    "storage": "storage",
}


# ─────────────────────────────────────────────────────────────────────────────
# HTTP Client (stdlib only)
# ─────────────────────────────────────────────────────────────────────────────

def make_request(url: str, params: dict = None) -> dict:
    """
    Make an HTTP GET request with retry logic and caching.
    Returns parsed JSON response.
    """
    if params:
        query = urllib.parse.urlencode(params)
        url = f"{url}?{query}"

    # Check cache first
    cache_key = url.replace("/", "_").replace(":", "").replace("?", "_").replace("&", "_")
    cache_file = CACHE_DIR / f"{cache_key[:200]}.json"

    if cache_file.exists():
        cache_age = time.time() - cache_file.stat().st_mtime
        if cache_age < CACHE_TTL_SECONDS:
            try:
                with open(cache_file, "r") as f:
                    cached = json.load(f)
                cached["_meta"] = cached.get("_meta", {})
                cached["_meta"]["from_cache"] = True
                cached["_meta"]["cache_age_seconds"] = int(cache_age)
                return cached
            except (json.JSONDecodeError, IOError):
                pass  # Cache corrupted, fetch fresh

    # Fetch with retry
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": USER_AGENT,
                    "Accept": "application/json",
                }
            )
            with urllib.request.urlopen(req, timeout=15) as response:
                data = json.loads(response.read().decode("utf-8"))

                # Cache the response
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                with open(cache_file, "w") as f:
                    json.dump(data, f)

                if isinstance(data, dict):
                    data["_meta"] = {"from_cache": False, "fetched_at": datetime.now(timezone.utc).isoformat()}
                return data

        except urllib.error.HTTPError as e:
            last_error = e
            if e.code == 429:  # Rate limited
                wait = RETRY_DELAY_BASE ** (attempt + 1) + 5
                print(f"[WARN] Rate limited (429). Waiting {wait}s before retry {attempt + 1}/{MAX_RETRIES}...",
                      file=sys.stderr)
                time.sleep(wait)
            elif e.code >= 500:
                wait = RETRY_DELAY_BASE ** (attempt + 1)
                print(f"[WARN] Server error ({e.code}). Retrying in {wait}s...", file=sys.stderr)
                time.sleep(wait)
            else:
                raise
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last_error = e
            wait = RETRY_DELAY_BASE ** (attempt + 1)
            print(f"[WARN] Network error: {e}. Retrying in {wait}s...", file=sys.stderr)
            time.sleep(wait)

    raise ConnectionError(f"Failed after {MAX_RETRIES} retries. Last error: {last_error}")


# ─────────────────────────────────────────────────────────────────────────────
# Symbol Resolution
# ─────────────────────────────────────────────────────────────────────────────

def resolve_token_id(input_str: str) -> str:
    """
    Resolve a user-provided token identifier to a CoinGecko ID.
    Handles: symbols (ETH), names (ethereum), and direct IDs.
    """
    normalized = input_str.strip().upper()

    # Check symbol map first (fast path)
    if normalized in SYMBOL_MAP:
        return SYMBOL_MAP[normalized]

    # If it looks like it's already a CoinGecko ID (lowercase, has hyphens), use directly
    lower = input_str.strip().lower()
    if "-" in lower or lower == input_str.strip():
        return lower

    # Search CoinGecko for the symbol
    try:
        results = make_request(f"{COINGECKO_BASE_URL}/search", {"query": input_str})
        coins = results.get("coins", [])
        if coins:
            # Return the first match by market cap rank (usually most relevant)
            return coins[0]["id"]
    except Exception as e:
        print(f"[WARN] Search failed: {e}", file=sys.stderr)

    # Fallback: use as-is
    return lower


def search_tokens(query: str) -> list:
    """Search for tokens by name or symbol."""
    results = make_request(f"{COINGECKO_BASE_URL}/search", {"query": query})
    coins = results.get("coins", [])
    return [
        {
            "id": c["id"],
            "symbol": c["symbol"],
            "name": c["name"],
            "market_cap_rank": c.get("market_cap_rank"),
            "thumb": c.get("thumb"),
        }
        for c in coins[:10]
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Data Fetching
# ─────────────────────────────────────────────────────────────────────────────

def fetch_single_token(token_id: str) -> dict:
    """Fetch comprehensive data for a single token."""
    token_id = resolve_token_id(token_id)

    # Fetch detailed coin data
    data = make_request(
        f"{COINGECKO_BASE_URL}/coins/{token_id}",
        {
            "localization": "false",
            "tickers": "false",
            "market_data": "true",
            "community_data": "true",
            "developer_data": "false",
            "sparkline": "false",
        }
    )

    # Extract and flatten the most important fields
    md = data.get("market_data", {})
    usd = lambda field: (md.get(field) or {}).get("usd")
    btc = lambda field: (md.get(field) or {}).get("btc")

    result = {
        "token_id": token_id,
        "symbol": data.get("symbol", "").upper(),
        "name": data.get("name", ""),
        "description": (data.get("description", {}) or {}).get("en", "")[:500],
        "categories": data.get("categories", []),
        "market_cap_rank": data.get("market_cap_rank"),
        "coingecko_rank": data.get("coingecko_rank"),
        "genesis_date": data.get("genesis_date"),
        "hashing_algorithm": data.get("hashing_algorithm"),
        "links": {
            "homepage": (data.get("links", {}).get("homepage", [None]))[0] if data.get("links", {}).get("homepage") else None,
            "twitter": data.get("links", {}).get("twitter_screen_name"),
            "telegram": data.get("links", {}).get("telegram_channel_identifier"),
            "subreddit": data.get("links", {}).get("subreddit_url"),
            "github": (data.get("links", {}).get("repos_url", {}).get("github", [None]))[0] if data.get("links", {}).get("repos_url", {}).get("github") else None,
        },
        "market_data": {
            "current_price_usd": usd("current_price"),
            "current_price_btc": btc("current_price"),
            "market_cap_usd": usd("market_cap"),
            "fully_diluted_valuation_usd": usd("fully_diluted_valuation"),
            "total_volume_usd": usd("total_volume"),
            "high_24h_usd": usd("high_24h"),
            "low_24h_usd": usd("low_24h"),
            "price_change_24h": md.get("price_change_24h"),
            "price_change_percentage_24h": md.get("price_change_percentage_24h"),
            "price_change_percentage_7d": md.get("price_change_percentage_7d"),
            "price_change_percentage_14d": md.get("price_change_percentage_14d"),
            "price_change_percentage_30d": md.get("price_change_percentage_30d"),
            "price_change_percentage_60d": md.get("price_change_percentage_60d"),
            "price_change_percentage_200d": md.get("price_change_percentage_200d"),
            "price_change_percentage_1y": md.get("price_change_percentage_1y"),
            "market_cap_change_24h": md.get("market_cap_change_24h"),
            "market_cap_change_percentage_24h": md.get("market_cap_change_percentage_24h"),
            "circulating_supply": md.get("circulating_supply"),
            "total_supply": md.get("total_supply"),
            "max_supply": md.get("max_supply"),
            "ath_usd": usd("ath"),
            "ath_change_percentage": (md.get("ath_change_percentage") or {}).get("usd"),
            "ath_date": (md.get("ath_date") or {}).get("usd"),
            "atl_usd": usd("atl"),
            "atl_change_percentage": (md.get("atl_change_percentage") or {}).get("usd"),
            "atl_date": (md.get("atl_date") or {}).get("usd"),
            "total_value_locked": md.get("total_value_locked", {}).get("usd") if isinstance(md.get("total_value_locked"), dict) else md.get("total_value_locked"),
            "mcap_to_tvl_ratio": md.get("mcap_to_tvl_ratio"),
            "fdv_to_tvl_ratio": md.get("fdv_to_tvl_ratio"),
        },
        "community_data": {
            "twitter_followers": data.get("community_data", {}).get("twitter_followers"),
            "reddit_subscribers": data.get("community_data", {}).get("reddit_subscribers"),
            "reddit_average_posts_48h": data.get("community_data", {}).get("reddit_average_posts_48h"),
            "reddit_average_comments_48h": data.get("community_data", {}).get("reddit_average_comments_48h"),
            "telegram_channel_user_count": data.get("community_data", {}).get("telegram_channel_user_count"),
        },
        "derived_metrics": {},
        "_meta": {
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "data_source": "coingecko",
            "skill_version": "1.0.0",
        }
    }

    # ── Compute Derived Metrics ──────────────────────────────────────────
    m = result["market_data"]
    derived = result["derived_metrics"]

    # MCap/FDV ratio (circulating ratio)
    if m["market_cap_usd"] and m["fully_diluted_valuation_usd"] and m["fully_diluted_valuation_usd"] > 0:
        derived["mcap_fdv_ratio"] = round(m["market_cap_usd"] / m["fully_diluted_valuation_usd"] * 100, 2)
    else:
        derived["mcap_fdv_ratio"] = None

    # Volume/MCap ratio (liquidity indicator)
    if m["total_volume_usd"] and m["market_cap_usd"] and m["market_cap_usd"] > 0:
        derived["volume_mcap_ratio"] = round(m["total_volume_usd"] / m["market_cap_usd"] * 100, 4)
    else:
        derived["volume_mcap_ratio"] = None

    # ATH distance
    if m["ath_usd"] and m["current_price_usd"] and m["ath_usd"] > 0:
        derived["distance_from_ath_pct"] = round((m["current_price_usd"] - m["ath_usd"]) / m["ath_usd"] * 100, 2)
    else:
        derived["distance_from_ath_pct"] = None

    # ATL distance
    if m["atl_usd"] and m["current_price_usd"] and m["atl_usd"] > 0:
        derived["distance_from_atl_pct"] = round((m["current_price_usd"] - m["atl_usd"]) / m["atl_usd"] * 100, 2)
    else:
        derived["distance_from_atl_pct"] = None

    # Circulating/Total supply ratio
    if m["circulating_supply"] and m["total_supply"] and m["total_supply"] > 0:
        derived["circulating_supply_ratio"] = round(m["circulating_supply"] / m["total_supply"] * 100, 2)
    else:
        derived["circulating_supply_ratio"] = None

    # Max supply inflation headroom
    if m["circulating_supply"] and m["max_supply"] and m["max_supply"] > 0:
        derived["supply_inflation_headroom_pct"] = round(
            (m["max_supply"] - m["circulating_supply"]) / m["max_supply"] * 100, 2
        )
    else:
        derived["supply_inflation_headroom_pct"] = None

    # Momentum classification
    pct_24h = m["price_change_percentage_24h"] or 0
    pct_7d = m["price_change_percentage_7d"] or 0
    pct_30d = m["price_change_percentage_30d"] or 0

    if pct_24h > 0 and pct_7d > 0 and pct_30d > 0:
        if pct_24h > pct_7d / 7:
            derived["momentum"] = "ACCELERATING_UP"
        else:
            derived["momentum"] = "STEADY_UP"
    elif pct_24h < 0 and pct_7d < 0 and pct_30d < 0:
        if abs(pct_24h) > abs(pct_7d / 7):
            derived["momentum"] = "ACCELERATING_DOWN"
        else:
            derived["momentum"] = "STEADY_DOWN"
    elif pct_30d > 0:
        derived["momentum"] = "MIXED_BULLISH"
    elif pct_30d < 0:
        derived["momentum"] = "MIXED_BEARISH"
    else:
        derived["momentum"] = "NEUTRAL"

    # Dilution risk flag
    if derived["mcap_fdv_ratio"] is not None:
        if derived["mcap_fdv_ratio"] < 30:
            derived["dilution_risk"] = "HIGH"
        elif derived["mcap_fdv_ratio"] < 60:
            derived["dilution_risk"] = "MEDIUM"
        else:
            derived["dilution_risk"] = "LOW"
    else:
        derived["dilution_risk"] = "UNKNOWN"

    # Liquidity classification
    vol_ratio = derived["volume_mcap_ratio"]
    if vol_ratio is not None:
        if vol_ratio > 20:
            derived["liquidity_grade"] = "VERY_HIGH"
        elif vol_ratio > 10:
            derived["liquidity_grade"] = "HIGH"
        elif vol_ratio > 3:
            derived["liquidity_grade"] = "MODERATE"
        elif vol_ratio > 1:
            derived["liquidity_grade"] = "LOW"
        else:
            derived["liquidity_grade"] = "VERY_LOW"
    else:
        derived["liquidity_grade"] = "UNKNOWN"

    return result


def fetch_multiple_tokens(token_ids: list) -> list:
    """Fetch data for multiple tokens (for comparison)."""
    results = []
    for idx, token_id in enumerate(token_ids):
        if idx > 0:
            time.sleep(1.5)  # Rate limit protection
        try:
            result = fetch_single_token(token_id)
            results.append(result)
        except Exception as e:
            results.append({
                "token_id": token_id,
                "error": str(e),
                "_meta": {"fetched_at": datetime.now(timezone.utc).isoformat()}
            })
    return results


def fetch_category(category: str, top_n: int = 10) -> dict:
    """Fetch top tokens in a category."""
    category_id = CATEGORY_MAP.get(category.lower(), category)

    data = make_request(
        f"{COINGECKO_BASE_URL}/coins/markets",
        {
            "vs_currency": "usd",
            "category": category_id,
            "order": "market_cap_desc",
            "per_page": str(top_n),
            "page": "1",
            "sparkline": "false",
            "price_change_percentage": "24h,7d,30d",
        }
    )

    if isinstance(data, list):
        tokens = []
        for coin in data:
            tokens.append({
                "id": coin.get("id"),
                "symbol": coin.get("symbol", "").upper(),
                "name": coin.get("name"),
                "market_cap_rank": coin.get("market_cap_rank"),
                "current_price": coin.get("current_price"),
                "market_cap": coin.get("market_cap"),
                "total_volume": coin.get("total_volume"),
                "price_change_percentage_24h": coin.get("price_change_percentage_24h"),
                "price_change_percentage_7d_in_currency": coin.get("price_change_percentage_7d_in_currency"),
                "price_change_percentage_30d_in_currency": coin.get("price_change_percentage_30d_in_currency"),
                "ath": coin.get("ath"),
                "ath_change_percentage": coin.get("ath_change_percentage"),
                "circulating_supply": coin.get("circulating_supply"),
                "total_supply": coin.get("total_supply"),
                "fully_diluted_valuation": coin.get("fully_diluted_valuation"),
            })

        return {
            "category": category,
            "category_id": category_id,
            "token_count": len(tokens),
            "tokens": tokens,
            "_meta": {
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "data_source": "coingecko",
            }
        }
    else:
        return {
            "category": category,
            "category_id": category_id,
            "error": "Unexpected response format",
            "raw": data,
        }


def fetch_global_data() -> dict:
    """Fetch global crypto market data for context."""
    data = make_request(f"{COINGECKO_BASE_URL}/global")
    gd = data.get("data", {})
    return {
        "total_market_cap_usd": gd.get("total_market_cap", {}).get("usd"),
        "total_volume_24h_usd": gd.get("total_volume", {}).get("usd"),
        "btc_dominance": gd.get("market_cap_percentage", {}).get("btc"),
        "eth_dominance": gd.get("market_cap_percentage", {}).get("eth"),
        "active_cryptocurrencies": gd.get("active_cryptocurrencies"),
        "markets": gd.get("markets"),
        "market_cap_change_percentage_24h": gd.get("market_cap_change_percentage_24h_usd"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Formatting
# ─────────────────────────────────────────────────────────────────────────────

def format_number(n, decimals=2, prefix="$"):
    """Format a number with appropriate suffix (K, M, B, T)."""
    if n is None:
        return "N/A"
    if isinstance(n, str):
        return n

    abs_n = abs(n)
    sign = "-" if n < 0 else ""

    if abs_n >= 1e12:
        formatted = f"{abs_n / 1e12:.{decimals}f}T"
    elif abs_n >= 1e9:
        formatted = f"{abs_n / 1e9:.{decimals}f}B"
    elif abs_n >= 1e6:
        formatted = f"{abs_n / 1e6:.{decimals}f}M"
    elif abs_n >= 1e3:
        formatted = f"{abs_n / 1e3:.{decimals}f}K"
    elif abs_n >= 1:
        formatted = f"{abs_n:.{decimals}f}"
    elif abs_n >= 0.01:
        formatted = f"{abs_n:.4f}"
    elif abs_n >= 0.0001:
        formatted = f"{abs_n:.6f}"
    else:
        formatted = f"{abs_n:.8f}"

    return f"{sign}{prefix}{formatted}" if prefix else f"{sign}{formatted}"


def format_percentage(n):
    """Format a percentage with + or - prefix."""
    if n is None:
        return "N/A"
    sign = "+" if n >= 0 else ""
    return f"{sign}{n:.2f}%"


def print_human_readable(data: dict):
    """Pretty-print data in a human-readable table format."""
    if "error" in data:
        print(f"❌ Error: {data['error']}")
        return

    m = data.get("market_data", {})
    d = data.get("derived_metrics", {})

    print(f"\n{'═' * 60}")
    print(f"  {data.get('name', '?')} ({data.get('symbol', '?')})")
    print(f"  Rank #{data.get('market_cap_rank', '?')} | CoinGecko Rank: {data.get('coingecko_rank', '?')}")
    print(f"{'═' * 60}")
    print()

    # Price section
    print("  📊 MARKET SNAPSHOT")
    print(f"  {'─' * 40}")
    print(f"  Price:         {format_number(m.get('current_price_usd'))}")
    print(f"  Market Cap:    {format_number(m.get('market_cap_usd'))}")
    print(f"  FDV:           {format_number(m.get('fully_diluted_valuation_usd'))}")
    print(f"  24h Volume:    {format_number(m.get('total_volume_usd'))}")
    print(f"  MCap/FDV:      {d.get('mcap_fdv_ratio', 'N/A')}%")
    print(f"  Vol/MCap:      {d.get('volume_mcap_ratio', 'N/A')}%")
    print()

    # Price changes
    print("  📈 PRICE CHANGES")
    print(f"  {'─' * 40}")
    print(f"  24h:           {format_percentage(m.get('price_change_percentage_24h'))}")
    print(f"  7d:            {format_percentage(m.get('price_change_percentage_7d'))}")
    print(f"  30d:           {format_percentage(m.get('price_change_percentage_30d'))}")
    print(f"  200d:          {format_percentage(m.get('price_change_percentage_200d'))}")
    print(f"  1y:            {format_percentage(m.get('price_change_percentage_1y'))}")
    print()

    # ATH/ATL
    print("  🏔️  ATH / ATL")
    print(f"  {'─' * 40}")
    print(f"  ATH:           {format_number(m.get('ath_usd'))} ({m.get('ath_date', 'N/A')[:10]})")
    print(f"  From ATH:      {format_percentage(d.get('distance_from_ath_pct'))}")
    print(f"  ATL:           {format_number(m.get('atl_usd'))} ({m.get('atl_date', 'N/A')[:10]})")
    print(f"  From ATL:      {format_percentage(d.get('distance_from_atl_pct'))}")
    print()

    # Supply
    print("  🪙 SUPPLY DYNAMICS")
    print(f"  {'─' * 40}")
    print(f"  Circulating:   {format_number(m.get('circulating_supply'), prefix='')}")
    print(f"  Total:         {format_number(m.get('total_supply'), prefix='')}")
    print(f"  Max:           {format_number(m.get('max_supply'), prefix='')}")
    print(f"  Circ/Total:    {d.get('circulating_supply_ratio', 'N/A')}%")
    print(f"  Inflation Room:{d.get('supply_inflation_headroom_pct', 'N/A')}%")
    print()

    # Derived signals
    print("  🧠 DERIVED SIGNALS")
    print(f"  {'─' * 40}")
    print(f"  Momentum:      {d.get('momentum', 'N/A')}")
    print(f"  Dilution Risk: {d.get('dilution_risk', 'N/A')}")
    print(f"  Liquidity:     {d.get('liquidity_grade', 'N/A')}")
    print()

    print(f"  ⏰ Data fetched: {data.get('_meta', {}).get('fetched_at', 'N/A')}")
    if data.get("_meta", {}).get("from_cache"):
        print(f"  📦 (from cache, age: {data['_meta'].get('cache_age_seconds', '?')}s)")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI Interface
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Crypto Thesis Engine — Market Data Fetcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --token bitcoin --output json
  %(prog)s --tokens bitcoin,ethereum,solana --output json
  %(prog)s --category layer2 --top 10 --output json
  %(prog)s --search "arbitrum" --output json
  %(prog)s --resolve ETH --output json
  %(prog)s --global --output json
        """
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--token", type=str, help="Single token ID or symbol to fetch")
    group.add_argument("--tokens", type=str, help="Comma-separated token IDs for comparison")
    group.add_argument("--category", type=str, help="Category to scan (e.g., layer1, defi, ai)")
    group.add_argument("--search", type=str, help="Search for tokens by name or symbol")
    group.add_argument("--resolve", type=str, help="Resolve a symbol to CoinGecko ID")
    group.add_argument("--global", dest="global_data", action="store_true", help="Fetch global market data")

    parser.add_argument("--top", type=int, default=10, help="Number of tokens for category scan (default: 10)")
    parser.add_argument("--output", choices=["json", "human", "both"], default="json",
                        help="Output format (default: json)")
    parser.add_argument("--no-cache", action="store_true", help="Skip cache, fetch fresh data")
    parser.add_argument("--cache-ttl", type=int, default=300, help="Cache TTL in seconds (default: 300)")

    args = parser.parse_args()

    # Apply cache settings
    global CACHE_TTL_SECONDS
    CACHE_TTL_SECONDS = args.cache_ttl
    if args.no_cache:
        CACHE_TTL_SECONDS = 0

    try:
        if args.token:
            data = fetch_single_token(args.token)
            if args.output in ("json", "both"):
                print(json.dumps(data, indent=2, ensure_ascii=False))
            if args.output in ("human", "both"):
                print_human_readable(data)

        elif args.tokens:
            token_list = [t.strip() for t in args.tokens.split(",") if t.strip()]
            if len(token_list) < 2:
                print("Error: --tokens requires at least 2 comma-separated tokens", file=sys.stderr)
                sys.exit(1)
            if len(token_list) > 5:
                print("Warning: Limiting to first 5 tokens to avoid rate limits", file=sys.stderr)
                token_list = token_list[:5]

            data = fetch_multiple_tokens(token_list)
            if args.output in ("json", "both"):
                print(json.dumps(data, indent=2, ensure_ascii=False))
            if args.output in ("human", "both"):
                for item in data:
                    print_human_readable(item)

        elif args.category:
            data = fetch_category(args.category, args.top)
            if args.output in ("json", "both"):
                print(json.dumps(data, indent=2, ensure_ascii=False))
            if args.output == "human":
                print(f"\n📂 Category: {data.get('category', '?')} ({data.get('token_count', 0)} tokens)")
                print(f"{'─' * 60}")
                for t in data.get("tokens", []):
                    print(f"  #{t.get('market_cap_rank', '?'):>4} {t.get('symbol', '?'):>6} "
                          f"{t.get('name', '?'):<25} "
                          f"{format_number(t.get('current_price')):>12} "
                          f"{format_number(t.get('market_cap')):>12} "
                          f"{format_percentage(t.get('price_change_percentage_24h')):>8}")

        elif args.search:
            results = search_tokens(args.search)
            if args.output in ("json", "both"):
                print(json.dumps(results, indent=2, ensure_ascii=False))
            if args.output == "human":
                print(f"\n🔍 Search results for '{args.search}':")
                for r in results:
                    rank = f"#{r['market_cap_rank']}" if r['market_cap_rank'] else "unranked"
                    print(f"  {r['symbol']:>6} {r['name']:<30} {rank:>10}  id: {r['id']}")

        elif args.resolve:
            resolved = resolve_token_id(args.resolve)
            if args.output in ("json", "both"):
                print(json.dumps({"input": args.resolve, "resolved_id": resolved}, indent=2))
            if args.output == "human":
                print(f"✅ '{args.resolve}' → '{resolved}'")

        elif args.global_data:
            data = fetch_global_data()
            if args.output in ("json", "both"):
                print(json.dumps(data, indent=2, ensure_ascii=False))
            if args.output == "human":
                print("\n🌍 Global Crypto Market")
                print(f"  Total MCap:    {format_number(data.get('total_market_cap_usd'))}")
                print(f"  24h Volume:    {format_number(data.get('total_volume_24h_usd'))}")
                print(f"  BTC Dominance: {data.get('btc_dominance', 'N/A'):.1f}%")
                print(f"  ETH Dominance: {data.get('eth_dominance', 'N/A'):.1f}%")
                print(f"  24h Change:    {format_percentage(data.get('market_cap_change_percentage_24h'))}")

    except KeyboardInterrupt:
        print("\nAborted.", file=sys.stderr)
        sys.exit(130)
    except ConnectionError as e:
        print(f"❌ Connection error: {e}", file=sys.stderr)
        sys.exit(1)
    except urllib.error.HTTPError as e:
        error_body = ""
        try:
            error_body = e.read().decode("utf-8")
        except Exception:
            pass
        print(json.dumps({
            "error": f"HTTP {e.code}: {e.reason}",
            "detail": error_body,
            "token": args.token or args.tokens or args.category or "",
        }, indent=2))
        sys.exit(1)
    except Exception as e:
        print(json.dumps({
            "error": str(e),
            "type": type(e).__name__,
        }, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()
