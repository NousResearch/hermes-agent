#!/usr/bin/env python3
"""
crypto-prices -- Fetch real-time prices for top 10 coins from CoinGecko.
Shows price (USD), 24h % change, market cap, and 24h volume in a formatted table.
Usage: python3 crypto_prices.py [--top N] [--coins btc,eth,sol]
"""

import sys
import json
import urllib.request
import urllib.error
import argparse
from datetime import datetime, timezone

# ── ANSI colours ────────────────────────────────────────────────────────────
RED    = "\033[91m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

COINGECKO_URL = (
    "https://api.coingecko.com/api/v3/coins/markets"
    "?vs_currency=usd"
    "&order=market_cap_desc"
    "&per_page={per_page}"
    "&page=1"
    "&sparkline=false"
    "&price_change_percentage=24h"
)

# CoinGecko IDs for common tickers
TICKER_TO_ID = {
    "btc": "bitcoin", "eth": "ethereum", "sol": "solana",
    "bnb": "binancecoin", "xrp": "ripple", "usdt": "tether",
    "usdc": "usd-coin", "ada": "cardano", "avax": "avalanche-2",
    "doge": "dogecoin", "dot": "polkadot", "matic": "matic-network",
    "link": "chainlink", "ltc": "litecoin", "uni": "uniswap",
    "atom": "cosmos", "xlm": "stellar", "near": "near",
}


def fetch_coins(per_page=10, ids=None):
    if ids:
        url = (
            "https://api.coingecko.com/api/v3/coins/markets"
            "?vs_currency=usd"
            f"&ids={','.join(ids)}"
            "&order=market_cap_desc"
            "&sparkline=false"
            "&price_change_percentage=24h"
        )
    else:
        url = COINGECKO_URL.format(per_page=per_page)

    req = urllib.request.Request(url, headers={"User-Agent": "crypto-prices/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        if e.code == 429:
            print(f"{RED}Rate limited by CoinGecko. Wait 60s and retry.{RESET}")
        else:
            print(f"{RED}HTTP error {e.code}: {e.reason}{RESET}")
        sys.exit(1)
    except urllib.error.URLError as e:
        print(f"{RED}Network error: {e.reason}{RESET}")
        sys.exit(1)


def fmt_price(val):
    if val >= 1_000:
        return f"${val:>12,.2f}"
    elif val >= 1:
        return f"${val:>12,.4f}"
    else:
        return f"${val:>12,.6f}"


def fmt_mcap(val):
    if val >= 1_000_000_000_000:
        return f"${val / 1_000_000_000_000:>7.2f}T"
    elif val >= 1_000_000_000:
        return f"${val / 1_000_000_000:>7.2f}B"
    elif val >= 1_000_000:
        return f"${val / 1_000_000:>7.2f}M"
    else:
        return f"${val:>10,.0f}"


def fmt_change(val):
    sign  = "+" if val >= 0 else ""
    color = GREEN if val >= 0 else RED
    arrow = "▲" if val >= 0 else "▼"
    return f"{color}{arrow} {sign}{val:>6.2f}%{RESET}"


def fmt_volume(val):
    if val >= 1_000_000_000:
        return f"${val / 1_000_000_000:>6.2f}B"
    elif val >= 1_000_000:
        return f"${val / 1_000_000:>6.2f}M"
    else:
        return f"${val:>8,.0f}"


def render_table(coins):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    W_RANK   = 4
    W_NAME   = 18
    W_SYMBOL = 6
    W_PRICE  = 15
    W_CHANGE = 12
    W_MCAP   = 13
    W_VOL    = 12

    sep = "─"
    header_line = (
        f"{'#':<{W_RANK}}  "
        f"{'Name':<{W_NAME}}"
        f"{'Sym':<{W_SYMBOL}}"
        f"{'Price (USD)':>{W_PRICE}}"
        f"  {'24h Chg':>{W_CHANGE - 2}}"
        f"  {'Market Cap':>{W_MCAP}}"
        f"  {'24h Volume':>{W_VOL}}"
    )
    total_width = len(header_line)

    print()
    print(f"{BOLD}{CYAN}  Crypto Prices  {DIM}(CoinGecko • {now}){RESET}")
    print(f"{DIM}  {sep * total_width}{RESET}")
    print(f"{BOLD}  {header_line}{RESET}")
    print(f"{DIM}  {sep * total_width}{RESET}")

    for coin in coins:
        rank    = coin.get("market_cap_rank") or "?"
        name    = (coin.get("name") or "?")[:W_NAME]
        symbol  = (coin.get("symbol") or "?").upper()[:W_SYMBOL]
        price   = coin.get("current_price") or 0
        change  = coin.get("price_change_percentage_24h_in_currency") or 0
        mcap    = coin.get("market_cap") or 0
        volume  = coin.get("total_volume") or 0

        price_s  = fmt_price(price)
        change_s = fmt_change(change)
        mcap_s   = fmt_mcap(mcap)
        volume_s = fmt_volume(volume)

        print(
            f"  {str(rank):<{W_RANK}}  "
            f"{name:<{W_NAME}}"
            f"{symbol:<{W_SYMBOL}}"
            f"{price_s}"
            f"  {change_s}"
            f"  {mcap_s:>{W_MCAP}}"
            f"  {volume_s:>{W_VOL}}"
        )

    print(f"{DIM}  {sep * total_width}{RESET}")
    print(f"{DIM}  {len(coins)} coins  •  Data: CoinGecko Free API{RESET}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Show real-time crypto prices from CoinGecko"
    )
    parser.add_argument(
        "--top", type=int, default=10, metavar="N",
        help="Show top N coins by market cap (default: 10)"
    )
    parser.add_argument(
        "--coins", type=str, metavar="TICKERS",
        help="Comma-separated tickers (e.g. btc,eth,sol). Overrides --top."
    )
    args = parser.parse_args()

    ids = None
    if args.coins:
        tickers = [t.strip().lower() for t in args.coins.split(",")]
        ids = [TICKER_TO_ID.get(t, t) for t in tickers]

    coins = fetch_coins(per_page=args.top, ids=ids)
    if not coins:
        print(f"{RED}No data returned. Check your ticker names or try again.{RESET}")
        sys.exit(1)

    render_table(coins)


if __name__ == "__main__":
    main()
