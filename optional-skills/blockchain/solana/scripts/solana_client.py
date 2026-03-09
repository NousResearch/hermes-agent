#!/usr/bin/env python3
"""
Solana Blockchain CLI Tool for Hermes Agent
--------------------------------------------
Queries the Solana JSON-RPC API and CoinGecko for enriched on-chain data.
Uses only Python standard library — no external packages required.

Usage:
  python3 solana_client.py stats
  python3 solana_client.py wallet   <address> [--limit N] [--all] [--no-prices]
  python3 solana_client.py tx       <signature>
  python3 solana_client.py token    <mint_address>
  python3 solana_client.py activity <address> [--limit N]
  python3 solana_client.py nft      <address>
  python3 solana_client.py whales   [--min-sol N]
  python3 solana_client.py price    <mint_address_or_symbol>

Environment:
  SOLANA_RPC_URL  Override the default RPC endpoint (default: mainnet-beta public)
"""

import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional

RPC_URL = os.environ.get(
    "SOLANA_RPC_URL",
    "https://api.mainnet-beta.solana.com",
)

LAMPORTS_PER_SOL = 1_000_000_000

# Well-known Solana token names — avoids API calls for common tokens.
# Maps mint address → (symbol, name).
KNOWN_TOKENS: Dict[str, tuple] = {
    "So11111111111111111111111111111111111111112":  ("SOL",   "Solana"),
    "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v": ("USDC",  "USD Coin"),
    "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB":  ("USDT",  "Tether"),
    "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263": ("BONK",  "Bonk"),
    "JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN":  ("JUP",   "Jupiter"),
    "7vfCXTUXx5WJV5JADk17DUJ4ksgau7utNKj4b963voxs": ("WETH",  "Wrapped Ether"),
    "jtojtomepa8beP8AuQc6eXt5FriJwfFMwQx2v2f9mCL":  ("JTO",   "Jito"),
    "mSoLzYCxHdYgdzU16g5QSh3i5K3z3KZK7ytfqcJm7So":  ("mSOL",  "Marinade Staked SOL"),
    "7dHbWXmci3dT8UFYWYZweBLXgycu7Y3iL6trKn1Y7ARj": ("stSOL", "Lido Staked SOL"),
    "HZ1JovNiVvGrGNiiYvEozEVgZ58xaU3RKwX8eACQBCt3": ("PYTH",  "Pyth Network"),
    "RLBxxFkseAZ4RgJH3Sqn8jXxhmGoz9jWxDNJMh8pL7a":  ("RLBB",  "Rollbit"),
    "hntyVP6YFm1Hg25TN9WGLqM12b8TQmcknKrdu1oxWux":  ("HNT",   "Helium"),
    "rndrizKT3MK1iimdxRdWabcF7Zg7AR5T4nud4EkHBof":  ("RNDR",  "Render"),
    "WENWENvqqNya429ubCdR81ZmD69brwQaaBYY6p91oHQQ":  ("WEN",   "Wen"),
    "85VBFQZC9TZkfaptBWjvUw7YbZjy52A6mjtPGjstQAmQ": ("W",     "Wormhole"),
    "TNSRxcUxoT9xBG3de7PiJyTDYu7kskLqcpddxnEJAS6":  ("TNSR",  "Tensor"),
    "DriFtupJYLTosbwoN8koMbEYSx54aFAVLddWsbksjwg7":  ("DRIFT", "Drift"),
    "bSo13r4TkiE4KumL71LsHTPpL2euBYLFx6h9HP3piy1":  ("bSOL",  "BlazeStake Staked SOL"),
    "27G8MtK7VtTcCHkpASjSDdkWWYfoqT6ggEuKidVJidD4": ("JLP",   "Jupiter LP"),
    "EKpQGSJtjMFqKZ9KQanSqYXRcF8fBopzLHYxdM65zcjm": ("WIF",   "dogwifhat"),
    "MEW1gQWJ3nEXg2qgERiKu7FAFj79PHvQVREQUzScPP5":  ("MEW",   "cat in a dogs world"),
    "ukHH6c7mMyiWCf1b9pnWe25TSpkDDt3H5pQZgZ74J82":  ("BOME",  "Book of Meme"),
    "A8C3xuqscfmyLrte3VwJvtPHXvcSN3FjDbUaSMAkQrCS": ("PENGU", "Pudgy Penguins"),
}

# Reverse lookup: symbol → mint (for the `price` command).
_SYMBOL_TO_MINT = {v[0].upper(): k for k, v in KNOWN_TOKENS.items()}


# ---------------------------------------------------------------------------
# HTTP / RPC helpers
# ---------------------------------------------------------------------------

def _http_get_json(url: str, timeout: int = 10, retries: int = 2) -> Any:
    """GET JSON from a URL with retry on 429 rate-limit. Returns parsed JSON or None."""
    for attempt in range(retries + 1):
        req = urllib.request.Request(
            url, headers={"Accept": "application/json", "User-Agent": "HermesAgent/1.0"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt < retries:
                time.sleep(2.0 * (attempt + 1))
                continue
            return None
        except Exception:
            return None
    return None


def _rpc_call(method: str, params: list = None, retries: int = 2) -> Any:
    """Send a JSON-RPC request with retry on 429 rate-limit."""
    payload = json.dumps({
        "jsonrpc": "2.0", "id": 1,
        "method": method, "params": params or [],
    }).encode()

    for attempt in range(retries + 1):
        req = urllib.request.Request(
            RPC_URL, data=payload,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                body = json.load(resp)
            if "error" in body:
                err = body["error"]
                # Rate-limit: retry after delay
                if isinstance(err, dict) and err.get("code") == 429:
                    if attempt < retries:
                        time.sleep(1.5 * (attempt + 1))
                        continue
                sys.exit(f"RPC error: {err}")
            return body.get("result")
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt < retries:
                time.sleep(1.5 * (attempt + 1))
                continue
            sys.exit(f"RPC HTTP error: {exc}")
        except urllib.error.URLError as exc:
            sys.exit(f"RPC connection error: {exc}")
    return None


# Keep backward compat — the rest of the code uses `rpc()`.
rpc = _rpc_call


def rpc_batch(calls: list) -> list:
    """Send a batch of JSON-RPC requests (with retry on 429)."""
    payload = json.dumps([
        {"jsonrpc": "2.0", "id": i, "method": c["method"], "params": c.get("params", [])}
        for i, c in enumerate(calls)
    ]).encode()

    for attempt in range(3):
        req = urllib.request.Request(
            RPC_URL, data=payload,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt < 2:
                time.sleep(1.5 * (attempt + 1))
                continue
            sys.exit(f"RPC batch HTTP error: {exc}")
        except urllib.error.URLError as exc:
            sys.exit(f"RPC batch error: {exc}")
    return []


def lamports_to_sol(lamports: int) -> float:
    return lamports / LAMPORTS_PER_SOL


def print_json(obj: Any) -> None:
    print(json.dumps(obj, indent=2))


def _short_mint(mint: str) -> str:
    """Abbreviate a mint address for display: first 4 + last 4."""
    if len(mint) <= 12:
        return mint
    return f"{mint[:4]}...{mint[-4:]}"


# ---------------------------------------------------------------------------
# Price & token name helpers (CoinGecko — free, no API key)
# ---------------------------------------------------------------------------

def fetch_prices(mints: List[str], max_lookups: int = 20) -> Dict[str, float]:
    """Fetch USD prices for mint addresses via CoinGecko (one per request).

    CoinGecko free tier doesn't support batch Solana token lookups,
    so we do individual calls — capped at *max_lookups* to stay within
    rate limits. Returns {mint: usd_price}.
    """
    prices: Dict[str, float] = {}
    for i, mint in enumerate(mints[:max_lookups]):
        url = (
            f"https://api.coingecko.com/api/v3/simple/token_price/solana"
            f"?contract_addresses={mint}&vs_currencies=usd"
        )
        data = _http_get_json(url, timeout=10)
        if data and isinstance(data, dict):
            for addr, info in data.items():
                if isinstance(info, dict) and "usd" in info:
                    prices[mint] = info["usd"]
                    break
        # Pause between calls to respect CoinGecko free-tier rate-limits
        if i < len(mints[:max_lookups]) - 1:
            time.sleep(1.0)
    return prices


def fetch_sol_price() -> Optional[float]:
    """Fetch current SOL price in USD via CoinGecko."""
    data = _http_get_json(
        "https://api.coingecko.com/api/v3/simple/price?ids=solana&vs_currencies=usd"
    )
    if data and "solana" in data:
        return data["solana"].get("usd")
    return None


def resolve_token_name(mint: str) -> Optional[Dict[str, str]]:
    """Look up token name and symbol from CoinGecko by mint address.

    Returns {"name": ..., "symbol": ...} or None.
    """
    if mint in KNOWN_TOKENS:
        sym, name = KNOWN_TOKENS[mint]
        return {"symbol": sym, "name": name}
    url = f"https://api.coingecko.com/api/v3/coins/solana/contract/{mint}"
    data = _http_get_json(url, timeout=10)
    if data and "symbol" in data:
        return {"symbol": data["symbol"].upper(), "name": data.get("name", "")}
    return None


def _token_label(mint: str) -> str:
    """Return a human-readable label for a mint: symbol if known, else abbreviated address."""
    if mint in KNOWN_TOKENS:
        return KNOWN_TOKENS[mint][0]
    return _short_mint(mint)


# ---------------------------------------------------------------------------
# 1. Network Stats
# ---------------------------------------------------------------------------

def cmd_stats(_args):
    """Live Solana network: slot, epoch, TPS, supply, version, SOL price."""
    results = rpc_batch([
        {"method": "getSlot"},
        {"method": "getEpochInfo"},
        {"method": "getRecentPerformanceSamples", "params": [1]},
        {"method": "getSupply"},
        {"method": "getVersion"},
    ])

    by_id = {r["id"]: r.get("result") for r in results}

    slot         = by_id.get(0)
    epoch_info   = by_id.get(1)
    perf_samples = by_id.get(2)
    supply       = by_id.get(3)
    version      = by_id.get(4)

    tps = None
    if perf_samples:
        s = perf_samples[0]
        tps = round(s["numTransactions"] / s["samplePeriodSecs"], 1)

    total_supply = lamports_to_sol(supply["value"]["total"])      if supply else None
    circ_supply  = lamports_to_sol(supply["value"]["circulating"]) if supply else None

    sol_price = fetch_sol_price()

    out = {
        "slot":                   slot,
        "epoch":                  epoch_info.get("epoch")     if epoch_info else None,
        "slot_in_epoch":          epoch_info.get("slotIndex") if epoch_info else None,
        "tps":                    tps,
        "total_supply_SOL":       round(total_supply, 2) if total_supply else None,
        "circulating_supply_SOL": round(circ_supply, 2)  if circ_supply  else None,
        "validator_version":      version.get("solana-core")  if version   else None,
    }
    if sol_price is not None:
        out["sol_price_usd"] = sol_price
        if circ_supply:
            out["market_cap_usd"] = round(sol_price * circ_supply, 0)
    print_json(out)


# ---------------------------------------------------------------------------
# 2. Wallet Info (enhanced with prices, sorting, filtering)
# ---------------------------------------------------------------------------

def cmd_wallet(args):
    """SOL balance + SPL token holdings with USD values."""
    address = args.address
    show_all = getattr(args, "all", False)
    limit = getattr(args, "limit", 20) or 20
    skip_prices = getattr(args, "no_prices", False)

    # Fetch SOL balance
    balance_result = rpc("getBalance", [address])
    sol_balance = lamports_to_sol(balance_result["value"])

    # Fetch all SPL token accounts
    token_result = rpc("getTokenAccountsByOwner", [
        address,
        {"programId": "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"},
        {"encoding": "jsonParsed"},
    ])

    raw_tokens = []
    for acct in (token_result.get("value") or []):
        info = acct["account"]["data"]["parsed"]["info"]
        ta = info["tokenAmount"]
        amount = float(ta.get("uiAmountString") or 0)
        if amount > 0:
            raw_tokens.append({
                "mint":     info["mint"],
                "amount":   amount,
                "decimals": ta["decimals"],
            })

    # Separate NFTs (amount=1, decimals=0) from fungible tokens
    nfts = [t for t in raw_tokens if t["decimals"] == 0 and t["amount"] == 1]
    fungible = [t for t in raw_tokens if not (t["decimals"] == 0 and t["amount"] == 1)]

    # Fetch prices for fungible tokens (cap lookups to avoid API abuse)
    sol_price = None
    prices: Dict[str, float] = {}
    if not skip_prices and fungible:
        sol_price = fetch_sol_price()
        # Prioritize known tokens, then a small sample of unknowns.
        # CoinGecko free tier = 1 request per mint, so we cap lookups.
        known_mints = [t["mint"] for t in fungible if t["mint"] in KNOWN_TOKENS]
        other_mints = [t["mint"] for t in fungible if t["mint"] not in KNOWN_TOKENS][:15]
        mints_to_price = known_mints + other_mints
        if mints_to_price:
            prices = fetch_prices(mints_to_price, max_lookups=30)

    # Enrich tokens with labels and USD values
    enriched = []
    dust_count = 0
    dust_value = 0.0
    for t in fungible:
        mint = t["mint"]
        label = _token_label(mint)
        usd_price = prices.get(mint)
        usd_value = round(usd_price * t["amount"], 2) if usd_price else None

        # Filter dust (< $0.01) unless --all
        if not show_all and usd_value is not None and usd_value < 0.01:
            dust_count += 1
            dust_value += usd_value
            continue

        entry = {"token": label, "mint": mint, "amount": t["amount"]}
        if usd_price is not None:
            entry["price_usd"] = usd_price
            entry["value_usd"] = usd_value
        enriched.append(entry)

    # Sort: tokens with known USD value first (highest→lowest), then unknowns
    enriched.sort(key=lambda x: (x.get("value_usd") is not None, x.get("value_usd") or 0), reverse=True)

    # Apply limit unless --all
    total_tokens = len(enriched)
    if not show_all and len(enriched) > limit:
        enriched = enriched[:limit]

    # Compute portfolio total
    total_usd = sum(t.get("value_usd", 0) for t in enriched)
    sol_value_usd = round(sol_price * sol_balance, 2) if sol_price else None
    if sol_value_usd:
        total_usd += sol_value_usd
    total_usd += dust_value

    output = {
        "address":     address,
        "sol_balance":  round(sol_balance, 9),
    }
    if sol_price:
        output["sol_price_usd"] = sol_price
        output["sol_value_usd"] = sol_value_usd
    output["tokens_shown"] = len(enriched)
    if total_tokens > len(enriched):
        output["tokens_hidden"] = total_tokens - len(enriched)
    output["spl_tokens"] = enriched
    if dust_count > 0:
        output["dust_filtered"] = {"count": dust_count, "total_value_usd": round(dust_value, 4)}
    output["nft_count"] = len(nfts)
    if nfts:
        output["nfts"] = [_token_label(n["mint"]) + f" ({_short_mint(n['mint'])})" for n in nfts[:10]]
        if len(nfts) > 10:
            output["nfts"].append(f"... and {len(nfts) - 10} more")
    if total_usd > 0:
        output["portfolio_total_usd"] = round(total_usd, 2)

    print_json(output)


# ---------------------------------------------------------------------------
# 3. Transaction Details
# ---------------------------------------------------------------------------

def cmd_tx(args):
    """Full transaction details by signature."""
    result = rpc("getTransaction", [
        args.signature,
        {"encoding": "jsonParsed", "maxSupportedTransactionVersion": 0},
    ])

    if result is None:
        sys.exit("Transaction not found (may be too old for public RPC history).")

    meta         = result.get("meta", {}) or {}
    msg          = result.get("transaction", {}).get("message", {})
    account_keys = msg.get("accountKeys", [])

    pre  = meta.get("preBalances",  [])
    post = meta.get("postBalances", [])

    balance_changes = []
    for i, key in enumerate(account_keys):
        acct_key = key["pubkey"] if isinstance(key, dict) else key
        if i < len(pre) and i < len(post):
            change = lamports_to_sol(post[i] - pre[i])
            if change != 0:
                balance_changes.append({"account": acct_key, "change_SOL": round(change, 9)})

    programs = []
    for ix in msg.get("instructions", []):
        prog = ix.get("programId")
        if prog is None and "programIdIndex" in ix:
            k = account_keys[ix["programIdIndex"]]
            prog = k["pubkey"] if isinstance(k, dict) else k
        if prog:
            programs.append(prog)

    # Add USD value for SOL changes
    sol_price = fetch_sol_price()
    if sol_price and balance_changes:
        for bc in balance_changes:
            bc["change_USD"] = round(bc["change_SOL"] * sol_price, 2)

    print_json({
        "signature":        args.signature,
        "slot":             result.get("slot"),
        "block_time":       result.get("blockTime"),
        "fee_SOL":          lamports_to_sol(meta.get("fee", 0)),
        "status":           "success" if meta.get("err") is None else "failed",
        "balance_changes":  balance_changes,
        "programs_invoked": list(dict.fromkeys(programs)),
    })


# ---------------------------------------------------------------------------
# 4. Token Info (enhanced with name + price)
# ---------------------------------------------------------------------------

def cmd_token(args):
    """SPL token metadata, supply, decimals, price, top holders."""
    mint = args.mint

    mint_info = rpc("getAccountInfo", [mint, {"encoding": "jsonParsed"}])
    if mint_info is None or mint_info.get("value") is None:
        sys.exit("Mint account not found.")

    parsed       = mint_info["value"]["data"]["parsed"]["info"]
    decimals     = parsed.get("decimals", 0)
    supply_raw   = int(parsed.get("supply", 0))
    supply_human = supply_raw / (10 ** decimals) if decimals else supply_raw

    largest = rpc("getTokenLargestAccounts", [mint])
    holders = []
    for acct in (largest.get("value") or [])[:5]:
        amount = float(acct.get("uiAmountString") or 0)
        pct = round((amount / supply_human * 100), 4) if supply_human > 0 else 0
        holders.append({
            "account": acct["address"],
            "amount":  amount,
            "percent": pct,
        })

    # Resolve name + price
    token_meta = resolve_token_name(mint)
    price_data = fetch_prices([mint])

    out = {"mint": mint}
    if token_meta:
        out["name"] = token_meta["name"]
        out["symbol"] = token_meta["symbol"]
    out["decimals"] = decimals
    out["supply"] = round(supply_human, min(decimals, 6))
    out["mint_authority"] = parsed.get("mintAuthority")
    out["freeze_authority"] = parsed.get("freezeAuthority")
    if mint in price_data:
        out["price_usd"] = price_data[mint]
        out["market_cap_usd"] = round(price_data[mint] * supply_human, 0)
    out["top_5_holders"] = holders

    print_json(out)


# ---------------------------------------------------------------------------
# 5. Recent Activity
# ---------------------------------------------------------------------------

def cmd_activity(args):
    """Recent transaction signatures for an address."""
    limit  = min(args.limit, 25)
    result = rpc("getSignaturesForAddress", [args.address, {"limit": limit}])

    txs = [
        {
            "signature": item["signature"],
            "slot":       item.get("slot"),
            "block_time": item.get("blockTime"),
            "err":        item.get("err"),
        }
        for item in (result or [])
    ]

    print_json({"address": args.address, "transactions": txs})


# ---------------------------------------------------------------------------
# 6. NFT Portfolio
# ---------------------------------------------------------------------------

def cmd_nft(args):
    """NFTs owned by a wallet (amount=1 && decimals=0 heuristic)."""
    result = rpc("getTokenAccountsByOwner", [
        args.address,
        {"programId": "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"},
        {"encoding": "jsonParsed"},
    ])

    nfts = [
        acct["account"]["data"]["parsed"]["info"]["mint"]
        for acct in (result.get("value") or [])
        if acct["account"]["data"]["parsed"]["info"]["tokenAmount"]["decimals"] == 0
        and int(acct["account"]["data"]["parsed"]["info"]["tokenAmount"]["amount"]) == 1
    ]

    print_json({
        "address":   args.address,
        "nft_count": len(nfts),
        "nfts":      nfts,
        "note":      "Heuristic only. Compressed NFTs (cNFTs) are not detected.",
    })


# ---------------------------------------------------------------------------
# 7. Whale Detector (enhanced with USD values)
# ---------------------------------------------------------------------------

def cmd_whales(args):
    """Scan the latest block for large SOL transfers."""
    min_lamports = int(args.min_sol * LAMPORTS_PER_SOL)

    slot  = rpc("getSlot")
    block = rpc("getBlock", [
        slot,
        {
            "encoding": "jsonParsed",
            "transactionDetails": "full",
            "maxSupportedTransactionVersion": 0,
            "rewards": False,
        },
    ])

    if block is None:
        sys.exit("Could not retrieve latest block.")

    sol_price = fetch_sol_price()

    whales = []
    for tx in (block.get("transactions") or []):
        meta = tx.get("meta", {}) or {}
        if meta.get("err") is not None:
            continue

        msg          = tx["transaction"].get("message", {})
        account_keys = msg.get("accountKeys", [])
        pre          = meta.get("preBalances",  [])
        post         = meta.get("postBalances", [])

        for i in range(len(pre)):
            change = post[i] - pre[i]
            if change >= min_lamports:
                k        = account_keys[i]
                receiver = k["pubkey"] if isinstance(k, dict) else k
                sender   = None
                for j in range(len(pre)):
                    if pre[j] - post[j] >= min_lamports:
                        sk     = account_keys[j]
                        sender = sk["pubkey"] if isinstance(sk, dict) else sk
                        break
                entry = {
                    "sender":     sender,
                    "receiver":   receiver,
                    "amount_SOL": round(lamports_to_sol(change), 4),
                }
                if sol_price:
                    entry["amount_USD"] = round(lamports_to_sol(change) * sol_price, 2)
                whales.append(entry)

    out = {
        "slot":              slot,
        "min_threshold_SOL": args.min_sol,
        "large_transfers":   whales,
        "note":              "Scans latest block only — point-in-time snapshot.",
    }
    if sol_price:
        out["sol_price_usd"] = sol_price
    print_json(out)


# ---------------------------------------------------------------------------
# 8. Price Lookup
# ---------------------------------------------------------------------------

def cmd_price(args):
    """Quick price lookup for a token by mint address or known symbol."""
    query = args.token

    # Check if it's a known symbol
    mint = _SYMBOL_TO_MINT.get(query.upper(), query)

    # Try to resolve name
    token_meta = resolve_token_name(mint)

    # Fetch price
    prices = fetch_prices([mint])

    out = {"query": query, "mint": mint}
    if token_meta:
        out["name"] = token_meta["name"]
        out["symbol"] = token_meta["symbol"]
    if mint in prices:
        out["price_usd"] = prices[mint]
    else:
        out["price_usd"] = None
        out["note"] = "Price not available — token may not be listed on CoinGecko."
    print_json(out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        prog="solana_client.py",
        description="Solana blockchain query tool for Hermes Agent",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("stats", help="Network stats: slot, epoch, TPS, supply, SOL price")

    p_wallet = sub.add_parser("wallet", help="SOL balance + SPL tokens with USD values")
    p_wallet.add_argument("address")
    p_wallet.add_argument("--limit", type=int, default=20,
                          help="Max tokens to display (default: 20)")
    p_wallet.add_argument("--all", action="store_true",
                          help="Show all tokens (no limit, no dust filter)")
    p_wallet.add_argument("--no-prices", action="store_true",
                          help="Skip price lookups (faster, RPC-only)")

    p_tx = sub.add_parser("tx", help="Transaction details by signature")
    p_tx.add_argument("signature")

    p_token = sub.add_parser("token", help="SPL token metadata, price, and top holders")
    p_token.add_argument("mint")

    p_activity = sub.add_parser("activity", help="Recent transactions for an address")
    p_activity.add_argument("address")
    p_activity.add_argument("--limit", type=int, default=10,
                            help="Number of transactions (max 25, default 10)")

    p_nft = sub.add_parser("nft", help="NFT portfolio for a wallet")
    p_nft.add_argument("address")

    p_whales = sub.add_parser("whales", help="Large SOL transfers in the latest block")
    p_whales.add_argument("--min-sol", type=float, default=1000.0,
                          help="Minimum SOL transfer size (default: 1000)")

    p_price = sub.add_parser("price", help="Quick price lookup by mint or symbol")
    p_price.add_argument("token", help="Mint address or known symbol (SOL, BONK, JUP, ...)")

    args = parser.parse_args()

    dispatch = {
        "stats":    cmd_stats,
        "wallet":   cmd_wallet,
        "tx":       cmd_tx,
        "token":    cmd_token,
        "activity": cmd_activity,
        "nft":      cmd_nft,
        "whales":   cmd_whales,
        "price":    cmd_price,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
