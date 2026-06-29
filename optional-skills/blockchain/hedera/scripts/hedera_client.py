#!/usr/bin/env python3
"""
hedera_client.py — Hedera blockchain CLI tool for the Hermes Agent project.

Read-only queries via the Hedera Mirror Node REST API and CoinGecko.
Zero external dependencies — Python standard library only:
  urllib, json, argparse, base64, re, threading, time, os, sys, datetime.

Commands:
  stats      Network health: latest block, supply, node count, HBAR price
  account    Account info: HBAR balance + HTS token holdings (capped at 10 tokens)
  token      HTS token metadata: name, type, supply, keys, custom fees
  tx         Transaction details by ID
  activity   Recent transactions for an account
  nft        NFTs held by an account, grouped by collection
  price      HBAR or HTS token price via CoinGecko
  fees       Common Hedera operation costs at live exchange rate
  topic      HCS topic metadata + recent messages
  contract   Smart contract info: EVM address ↔ account ID, bytecode size, keys

Global flags:
  --network {mainnet,testnet}  (default: mainnet)

Environment:
  HEDERA_MIRROR_URL  Override mirror node base URL (takes precedence over --network)
"""

from __future__ import annotations

import argparse
import base64
import datetime
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Network registry
# ---------------------------------------------------------------------------

NETWORKS: Dict[str, Dict[str, str]] = {
    "mainnet": {
        "mirror": "https://mainnet-public.mirrornode.hedera.com",
        "explorer": "https://hashscan.io/mainnet",
    },
    "testnet": {
        "mirror": "https://testnet.mirrornode.hedera.com",
        "explorer": "https://hashscan.io/testnet",
    },
}

# Set once by main() before any command runs; read by all helpers.
_NETWORK: str = "mainnet"

TINYBAR_PER_HBAR: int = 100_000_000

# Maximum HTS tokens enriched with metadata in the `account` command.
# Caps the number of concurrent /api/v1/tokens/{id} calls per account lookup.
ACCOUNT_TOKEN_CAP: int = 10

# ---------------------------------------------------------------------------
# Known HTS tokens  {token_id: (symbol, name, coingecko_id_or_None)}
# Used by `price` and `account` for USD lookups without extra API round-trips.
# ---------------------------------------------------------------------------

KNOWN_TOKENS: Dict[str, Tuple[str, str, Optional[str]]] = {
    "0.0.731861": ("SAUCE", "SaucerSwap",    "saucerswap"),
    "0.0.834870":  ("HBARX", "Stader HBARX", "stader-hbarx"),
}

# Reverse lookup: uppercase symbol → token_id
_SYMBOL_TO_TOKEN_ID: Dict[str, str] = {v[0].upper(): k for k, v in KNOWN_TOKENS.items()}

# ---------------------------------------------------------------------------
# Hedera fee schedule (USD costs, 2025-Q4 published schedule).
# Source: https://docs.hedera.com/hedera/networks/mainnet/fees
# Fees are denominated in USD; HBAR equivalent computed at the live exchange
# rate returned by /api/v1/network/exchangerate.
# ---------------------------------------------------------------------------

FEE_SCHEDULE: List[Dict[str, Any]] = [
    {"operation": "CryptoCreate",           "description": "Create account",                 "usd": 0.05},
    {"operation": "CryptoTransfer (HBAR)",  "description": "Transfer HBAR",                 "usd": 0.0001},
    {"operation": "CryptoTransfer (HTS)",   "description": "Transfer fungible HTS token",   "usd": 0.001},
    {"operation": "CryptoTransfer (NFT)",   "description": "Transfer NFT",                  "usd": 0.001},
    {"operation": "TokenCreate",            "description": "Create HTS token",               "usd": 1.00},
    {"operation": "TokenAssociate",         "description": "Associate token to account",     "usd": 0.05},
    {"operation": "TokenMint (fungible)",   "description": "Mint fungible tokens",           "usd": 0.00022},
    {"operation": "TokenMint (NFT)",        "description": "Mint one NFT",                  "usd": 0.02},
    {"operation": "ConsensusCreateTopic",   "description": "Create HCS topic",              "usd": 0.01},
    {"operation": "ConsensusSubmitMessage", "description": "Submit HCS message (≤6 KB)",    "usd": 0.0001},
    {"operation": "ContractCreate",         "description": "Deploy contract (base, +gas)",  "usd": 1.00},
    {"operation": "ContractCall",           "description": "Call contract (base, +gas)",    "usd": 0.05},
    {"operation": "FileCreate",             "description": "Create file (first 1 KB)",      "usd": 0.05},
]

# ---------------------------------------------------------------------------
# Network / config helpers
# ---------------------------------------------------------------------------


def _mirror_base() -> str:
    env = os.environ.get("HEDERA_MIRROR_URL", "").strip()
    if env:
        return env.rstrip("/")
    return NETWORKS[_NETWORK]["mirror"]


def _explorer_base() -> str:
    return NETWORKS.get(_NETWORK, NETWORKS["mainnet"])["explorer"]


# ---------------------------------------------------------------------------
# HTTP layer
# ---------------------------------------------------------------------------


def _http_get(
    path: str,
    params: Optional[Dict[str, str]] = None,
    retries: int = 3,
    timeout: int = 15,
) -> Optional[Any]:
    """GET from the mirror node REST API.

    Returns parsed JSON on success, None on HTTP 404, and raises RuntimeError
    on other non-retriable failures.  Retries up to `retries` times on 429
    with exponential back-off (1 s → 2 s → 4 s).
    """
    url = f"{_mirror_base()}{path}"
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    headers = {"Accept": "application/json", "User-Agent": "HermesAgent/1.0"}
    req = urllib.request.Request(url, headers=headers)
    delay = 1.0
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8", errors="replace"))
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            if exc.code == 429 and attempt < retries - 1:
                time.sleep(delay)
                delay = min(delay * 2, 8.0)
                continue
            raise RuntimeError(f"HTTP {exc.code} for {url}") from exc
        except urllib.error.URLError as exc:
            if attempt < retries - 1:
                time.sleep(delay)
                delay = min(delay * 2, 8.0)
                continue
            raise RuntimeError(f"Connection error for {url}: {exc}") from exc
    return None


def _cg_get(path: str, timeout: int = 10) -> Optional[Any]:
    """GET from CoinGecko free API. Returns parsed JSON or None on any failure."""
    url = f"https://api.coingecko.com/api/v3{path}"
    headers = {"Accept": "application/json", "User-Agent": "HermesAgent/1.0"}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace"))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Unit conversions and display helpers
# ---------------------------------------------------------------------------


def tinybar_to_hbar(tinybar: int) -> float:
    return tinybar / TINYBAR_PER_HBAR


def hbar_to_usd(hbar: float, price_usd: float) -> float:
    return round(hbar * price_usd, 6)


def decode_b64(s: str) -> str:
    """Base64-decode a string.

    Returns the decoded bytes as a UTF-8 string when valid; falls back to a
    lowercase hex representation for non-UTF-8 binary content.
    """
    if not s:
        return ""
    try:
        raw = base64.b64decode(s)
        return raw.decode("utf-8")
    except Exception:
        pass
    try:
        raw = base64.b64decode(s)
        return raw.hex()
    except Exception:
        return s


def _fmt_ts(ts: str) -> str:
    """Format a Hedera consensus timestamp ('SSSSSSSSSS.NNNNNNNNN') as ISO-8601."""
    if not ts:
        return ts
    parts = ts.split(".")
    if not parts[0]:
        return ts
    try:
        dt = datetime.datetime.fromtimestamp(int(parts[0]), tz=datetime.timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return ts


def print_json(obj: Any) -> None:
    print(json.dumps(obj, indent=2, default=str))


# ---------------------------------------------------------------------------
# Input validators
# ---------------------------------------------------------------------------

_DOTTED_ID_RE = re.compile(r"^\d+\.\d+\.\d+$")
_EVM_ADDR_RE = re.compile(r"^0x[0-9a-fA-F]{40}$")
# Canonical form:  0.0.1234-1234567890-000000000
_TX_CANONICAL_RE = re.compile(r"^\d+\.\d+\.\d+-\d+-\d+$")
# SDK (@) form:    0.0.1234@1234567890.000000000
_TX_AT_RE = re.compile(r"^(\d+\.\d+\.\d+)@(\d+)\.(\d+)$")


def _normalize_tx_id(s: str) -> str:
    """Convert SDK-form tx ID to canonical dash form, or return as-is."""
    m = _TX_AT_RE.match(s.strip())
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return s.strip()


def require_account_id(s: str, field: str = "account ID") -> str:
    s = s.strip()
    if _DOTTED_ID_RE.match(s) or _EVM_ADDR_RE.match(s):
        return s
    sys.stderr.write(
        f"error: invalid {field} {s!r}: expected '0.0.XXXX' or 40-char EVM address\n"
    )
    sys.exit(2)


def require_topic_id(s: str) -> str:
    s = s.strip()
    if _DOTTED_ID_RE.match(s):
        return s
    sys.stderr.write(f"error: invalid topic ID {s!r}: expected '0.0.XXXX'\n")
    sys.exit(2)


def require_tx_id(s: str) -> str:
    s = _normalize_tx_id(s)
    if _TX_CANONICAL_RE.match(s):
        return s
    sys.stderr.write(
        f"error: invalid transaction ID {s!r}: "
        f"expected '0.0.X-SSSSSSSSSS-NNNNNNNNN' or '0.0.X@SSSSSSSSSS.NNNNNNNNN'\n"
    )
    sys.exit(2)


def require_contract_id(s: str) -> str:
    s = s.strip()
    if _DOTTED_ID_RE.match(s) or _EVM_ADDR_RE.match(s):
        return s
    sys.stderr.write(
        f"error: invalid contract ID {s!r}: expected '0.0.XXXX' or 40-char EVM address\n"
    )
    sys.exit(2)


# ---------------------------------------------------------------------------
# Price helpers
# ---------------------------------------------------------------------------


def fetch_hbar_price() -> Optional[float]:
    data = _cg_get("/simple/price?ids=hedera-hashgraph&vs_currencies=usd")
    if data and "hedera-hashgraph" in data:
        return data["hedera-hashgraph"].get("usd")
    return None


def fetch_token_price(coingecko_id: str) -> Optional[float]:
    data = _cg_get(
        f"/simple/price?ids={urllib.parse.quote(coingecko_id)}&vs_currencies=usd"
    )
    if data and coingecko_id in data:
        return data[coingecko_id].get("usd")
    return None


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_stats(args: argparse.Namespace) -> int:
    results: Dict[str, Any] = {}
    errors: List[str] = []

    def fetch_blocks() -> None:
        try:
            data = _http_get("/api/v1/blocks", {"limit": "1", "order": "desc"})
            if data and data.get("blocks"):
                b = data["blocks"][0]
                results["latest_block"] = b.get("number")
                results["latest_block_timestamp"] = _fmt_ts(
                    str(b.get("consensus_end_timestamp", ""))
                )
        except Exception as exc:
            errors.append(f"blocks: {exc}")

    def fetch_supply() -> None:
        try:
            data = _http_get("/api/v1/network/supply")
            if data:
                released = data.get("released_supply")
                total = data.get("total_supply")
                results["total_supply_hbar"] = (
                    round(tinybar_to_hbar(int(total)), 0) if total else None
                )
                results["released_supply_hbar"] = (
                    round(tinybar_to_hbar(int(released)), 0) if released else None
                )
        except Exception as exc:
            errors.append(f"supply: {exc}")

    def fetch_nodes() -> None:
        try:
            data = _http_get("/api/v1/network/nodes", {"limit": "100"})
            if data:
                results["node_count"] = len(data.get("nodes", []))
        except Exception as exc:
            errors.append(f"nodes: {exc}")

    threads = [
        threading.Thread(target=fetch_blocks),
        threading.Thread(target=fetch_supply),
        threading.Thread(target=fetch_nodes),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    hbar_price = fetch_hbar_price()
    results["hbar_price_usd"] = hbar_price

    released = results.get("released_supply_hbar")
    if hbar_price and released:
        results["market_cap_usd"] = round(hbar_price * released, 0)

    results["network"] = _NETWORK
    results["explorer"] = _explorer_base()

    if errors:
        results["warnings"] = errors

    print_json(results)
    return 0


def cmd_account(args: argparse.Namespace) -> int:
    account_id = require_account_id(args.account_id)

    data = _http_get(f"/api/v1/accounts/{account_id}")
    if data is None:
        sys.stderr.write(f"error: account {account_id!r} not found\n")
        return 1

    bal_obj = data.get("balance", {})
    hbar_tinybar = int(bal_obj.get("balance", 0))
    hbar = tinybar_to_hbar(hbar_tinybar)

    hbar_price: Optional[float] = None
    if not args.no_prices:
        hbar_price = fetch_hbar_price()

    hbar_usd = hbar_to_usd(hbar, hbar_price) if hbar_price else None

    raw_tokens: List[Dict[str, Any]] = bal_obj.get("tokens", [])
    tokens_to_enrich = raw_tokens[:ACCOUNT_TOKEN_CAP]

    token_meta: Dict[str, Dict[str, Any]] = {}
    meta_lock = threading.Lock()

    def fetch_meta(token_id: str) -> None:
        meta = _http_get(f"/api/v1/tokens/{token_id}")
        if meta:
            with meta_lock:
                token_meta[token_id] = meta

    meta_threads = [
        threading.Thread(target=fetch_meta, args=(t["token_id"],))
        for t in tokens_to_enrich
    ]
    for th in meta_threads:
        th.start()
    for th in meta_threads:
        th.join()

    token_rows: List[Dict[str, Any]] = []
    for t in tokens_to_enrich:
        tid = t["token_id"]
        raw_bal = int(t.get("balance", 0))
        meta = token_meta.get(tid, {})
        decimals = int(meta.get("decimals") or 0)
        symbol = meta.get("symbol") or tid
        name = meta.get("name") or ""
        amount = raw_bal / (10 ** decimals) if decimals else raw_bal

        price_usd: Optional[float] = None
        if not args.no_prices and tid in KNOWN_TOKENS:
            cg_id = KNOWN_TOKENS[tid][2]
            if cg_id:
                price_usd = fetch_token_price(cg_id)

        row: Dict[str, Any] = {
            "token_id": tid,
            "symbol": symbol,
            "name": name,
            "balance": round(amount, min(decimals, 8)) if decimals else amount,
        }
        if price_usd is not None:
            row["price_usd"] = price_usd
            row["value_usd"] = round(amount * price_usd, 4)

        token_rows.append(row)

    token_rows.sort(key=lambda r: r.get("value_usd", -1), reverse=True)

    omitted = len(raw_tokens) - len(tokens_to_enrich)

    out: Dict[str, Any] = {
        "account_id": data.get("account"),
        "evm_address": data.get("evm_address"),
        "memo": data.get("memo") or None,
        "receiver_sig_required": data.get("receiver_sig_required"),
        "hbar_balance": round(hbar, 8),
    }
    if hbar_usd is not None:
        out["hbar_value_usd"] = hbar_usd

    out["tokens"] = token_rows

    if omitted > 0:
        out["tokens_omitted"] = omitted
        out["tokens_note"] = (
            f"showing first {ACCOUNT_TOKEN_CAP} of {len(raw_tokens)} associated tokens"
        )

    total_usd = (hbar_usd or 0.0) + sum(r.get("value_usd", 0.0) for r in token_rows)
    if hbar_price is not None:
        out["total_portfolio_usd"] = round(total_usd, 4)

    out["hashscan_url"] = f"{_explorer_base()}/account/{account_id}"
    print_json(out)
    return 0


def cmd_token(args: argparse.Namespace) -> int:
    token_id = args.token_id.strip()
    if not _DOTTED_ID_RE.match(token_id) and not _EVM_ADDR_RE.match(token_id):
        sys.stderr.write(f"error: invalid token ID {token_id!r}: expected '0.0.XXXX'\n")
        sys.exit(2)

    data = _http_get(f"/api/v1/tokens/{token_id}")
    if data is None:
        sys.stderr.write(f"error: token {token_id!r} not found\n")
        return 1

    decimals = int(data.get("decimals") or 0)
    total_supply_raw = int(data.get("total_supply") or 0)
    initial_supply_raw = int(data.get("initial_supply") or 0)
    divisor = 10 ** decimals if decimals else 1
    total_supply = total_supply_raw / divisor
    initial_supply = initial_supply_raw / divisor

    out: Dict[str, Any] = {
        "token_id": data.get("token_id"),
        "name": data.get("name"),
        "symbol": data.get("symbol"),
        "type": data.get("type"),
        "decimals": decimals,
        "total_supply": total_supply,
        "initial_supply": initial_supply,
        "max_supply": data.get("max_supply"),
        "supply_type": data.get("supply_type"),
        "treasury_account_id": data.get("treasury_account_id"),
        "memo": data.get("memo") or None,
        "freeze_default": data.get("freeze_default"),
        "pause_status": data.get("pause_status"),
        # Key presence as booleans — raw key material is not exposed
        "admin_key":  bool(data.get("admin_key")),
        "supply_key": bool(data.get("supply_key")),
        "freeze_key": bool(data.get("freeze_key")),
        "kyc_key":    bool(data.get("kyc_key")),
        "wipe_key":   bool(data.get("wipe_key")),
        "pause_key":  bool(data.get("pause_key")),
        "custom_fees": data.get("custom_fees"),
        "created_timestamp": _fmt_ts(str(data.get("created_timestamp", ""))),
        "hashscan_url": f"{_explorer_base()}/token/{data.get('token_id', token_id)}",
    }

    if token_id in KNOWN_TOKENS:
        cg_id = KNOWN_TOKENS[token_id][2]
        if cg_id:
            price = fetch_token_price(cg_id)
            if price is not None:
                out["price_usd"] = price
                out["market_cap_usd"] = round(price * total_supply, 2)

    print_json(out)
    return 0


def cmd_tx(args: argparse.Namespace) -> int:
    tx_id = require_tx_id(args.tx_id)

    data = _http_get(f"/api/v1/transactions/{tx_id}")
    if data is None:
        sys.stderr.write(f"error: transaction {tx_id!r} not found\n")
        return 1

    txs: List[Dict[str, Any]] = (
        data.get("transactions", [data]) if "transactions" in data else [data]
    )
    if not txs:
        sys.stderr.write(f"error: transaction {tx_id!r} not found\n")
        return 1
    tx = txs[0]

    fee_tinybar = int(tx.get("charged_tx_fee") or 0)
    fee_hbar = tinybar_to_hbar(fee_tinybar)
    hbar_price = fetch_hbar_price()

    transfers = sorted(
        [
            {
                "account": t.get("account"),
                "amount_hbar": round(tinybar_to_hbar(int(t.get("amount") or 0)), 8),
            }
            for t in tx.get("transfers", [])
        ],
        key=lambda r: abs(r["amount_hbar"]),
        reverse=True,
    )

    out: Dict[str, Any] = {
        "transaction_id": tx.get("transaction_id"),
        "type": tx.get("name"),
        "result": tx.get("result"),
        "consensus_timestamp": _fmt_ts(str(tx.get("consensus_timestamp", ""))),
        "valid_start_timestamp": _fmt_ts(str(tx.get("valid_start_timestamp", ""))),
        "charged_tx_fee_hbar": round(fee_hbar, 8),
    }

    if hbar_price is not None:
        out["charged_tx_fee_usd"] = round(fee_hbar * hbar_price, 6)

    memo_b64 = tx.get("memo_base64", "")
    if memo_b64:
        out["memo"] = decode_b64(memo_b64)

    out["transfers"] = transfers

    token_transfers = tx.get("token_transfers", [])
    if token_transfers:
        out["token_transfers"] = token_transfers

    nft_transfers = tx.get("nft_transfers", [])
    if nft_transfers:
        out["nft_transfers"] = nft_transfers

    out["hashscan_url"] = (
        f"{_explorer_base()}/transaction/{tx.get('transaction_id', tx_id)}"
    )
    print_json(out)
    return 0


def cmd_activity(args: argparse.Namespace) -> int:
    account_id = require_account_id(args.account_id)
    limit = max(1, min(args.limit, 100))

    data = _http_get(
        "/api/v1/transactions",
        {"account.id": account_id, "limit": str(limit), "order": "desc"},
    )
    if data is None:
        sys.stderr.write(f"error: could not fetch activity for {account_id!r}\n")
        return 1

    rows = [
        {
            "transaction_id": tx.get("transaction_id"),
            "type": tx.get("name"),
            "result": tx.get("result"),
            "consensus_timestamp": _fmt_ts(str(tx.get("consensus_timestamp", ""))),
            "fee_hbar": round(tinybar_to_hbar(int(tx.get("charged_tx_fee") or 0)), 8),
        }
        for tx in data.get("transactions", [])
    ]

    print_json({"account_id": account_id, "transactions": rows, "count": len(rows)})
    return 0


def cmd_nft(args: argparse.Namespace) -> int:
    account_id = require_account_id(args.account_id)
    limit = max(1, min(args.limit, 100))

    data = _http_get(
        f"/api/v1/accounts/{account_id}/nfts",
        {"limit": str(limit), "order": "desc"},
    )
    if data is None:
        sys.stderr.write(f"error: could not fetch NFTs for {account_id!r}\n")
        return 1

    collections: Dict[str, List[Dict[str, Any]]] = {}
    for nft in data.get("nfts", []):
        tid = nft.get("token_id", "unknown")
        meta_raw = nft.get("metadata", "")
        entry: Dict[str, Any] = {
            "serial_number": nft.get("serial_number"),
            "metadata": decode_b64(meta_raw) if meta_raw else None,
            "created_timestamp": _fmt_ts(str(nft.get("created_timestamp", ""))),
        }
        collections.setdefault(tid, []).append(entry)

    print_json(
        {
            "account_id": account_id,
            "total_nfts": sum(len(v) for v in collections.values()),
            "collections": [
                {"token_id": tid, "count": len(items), "nfts": items}
                for tid, items in collections.items()
            ],
        }
    )
    return 0


def cmd_price(args: argparse.Namespace) -> int:
    query = args.token.strip()

    if query.upper() == "HBAR":
        price = fetch_hbar_price()
        if price is None:
            sys.stderr.write("error: could not fetch HBAR price from CoinGecko\n")
            return 1
        print_json({"symbol": "HBAR", "name": "Hedera", "price_usd": price, "source": "coingecko"})
        return 0

    if _DOTTED_ID_RE.match(query):
        token_id = query
        if token_id in KNOWN_TOKENS:
            sym, name, cg_id = KNOWN_TOKENS[token_id]
            if cg_id:
                price = fetch_token_price(cg_id)
                if price is not None:
                    print_json({"token_id": token_id, "symbol": sym, "name": name,
                                "price_usd": price, "source": "coingecko"})
                    return 0
        print_json({"token_id": token_id, "price_usd": None,
                    "note": "not in known token registry"})
        return 0

    tid = _SYMBOL_TO_TOKEN_ID.get(query.upper())
    if tid:
        sym, name, cg_id = KNOWN_TOKENS[tid]
        if cg_id:
            price = fetch_token_price(cg_id)
            if price is not None:
                print_json({"token_id": tid, "symbol": sym, "name": name,
                            "price_usd": price, "source": "coingecko"})
                return 0

    print_json({"query": query, "price_usd": None,
                "note": "not in known token registry — use token ID or add to KNOWN_TOKENS"})
    return 0


def cmd_fees(args: argparse.Namespace) -> int:
    rate_data = _http_get("/api/v1/network/exchangerate")

    hbar_per_usd: Optional[float] = None
    rate_source = "unavailable"

    if rate_data and "current_rate" in rate_data:
        cr = rate_data["current_rate"]
        cent_eq = int(cr.get("cent_equivalent") or 0)
        hbar_eq = int(cr.get("hbar_equivalent") or 0)
        if cent_eq > 0 and hbar_eq > 0:
            # cent_equivalent cents = hbar_equivalent HBAR
            # → USD per HBAR = (cent_eq / hbar_eq) / 100
            usd_per_hbar = (cent_eq / hbar_eq) / 100
            hbar_per_usd = 1.0 / usd_per_hbar
            rate_source = "mirror_node"

    rows: List[Dict[str, Any]] = []
    for entry in FEE_SCHEDULE:
        usd = entry["usd"]
        row: Dict[str, Any] = {
            "operation": entry["operation"],
            "description": entry["description"],
            "cost_usd": usd,
        }
        if hbar_per_usd is not None:
            row["cost_hbar"] = round(usd * hbar_per_usd, 6)
        rows.append(row)

    out: Dict[str, Any] = {
        "network": _NETWORK,
        "exchange_rate_source": rate_source,
        "fee_schedule_version": "2025-Q4",
        "fee_schedule_url": "https://docs.hedera.com/hedera/networks/mainnet/fees",
        "operations": rows,
    }
    if hbar_per_usd is not None:
        out["hbar_per_usd"] = round(hbar_per_usd, 4)

    print_json(out)
    return 0


def cmd_topic(args: argparse.Namespace) -> int:
    topic_id = require_topic_id(args.topic_id)
    msg_limit = max(1, min(args.messages, 100))

    topic_data = _http_get(f"/api/v1/topics/{topic_id}")
    if topic_data is None:
        sys.stderr.write(f"error: topic {topic_id!r} not found\n")
        return 1

    msgs_data = _http_get(
        f"/api/v1/topics/{topic_id}/messages",
        {"limit": str(msg_limit), "order": "desc"},
    )

    messages: List[Dict[str, Any]] = []
    if msgs_data:
        for m in msgs_data.get("messages", []):
            raw_msg = m.get("message", "")
            rh = m.get("running_hash", "")
            messages.append(
                {
                    "sequence_number": m.get("sequence_number"),
                    "consensus_timestamp": _fmt_ts(str(m.get("consensus_timestamp", ""))),
                    "message": decode_b64(raw_msg) if raw_msg else None,
                    "running_hash_prefix": rh[:16] if rh else None,
                }
            )

    out: Dict[str, Any] = {
        "topic_id": topic_data.get("topic_id"),
        "memo": topic_data.get("memo") or None,
        "admin_key": bool(topic_data.get("admin_key")),
        "submit_key": bool(topic_data.get("submit_key")),
        "auto_renew_period": topic_data.get("auto_renew_period"),
        "auto_renew_account": topic_data.get("auto_renew_account"),
        "created_timestamp": _fmt_ts(str(topic_data.get("created_timestamp", ""))),
        "deleted": topic_data.get("deleted", False),
        "recent_messages": messages,
        "hashscan_url": f"{_explorer_base()}/topic/{topic_data.get('topic_id', topic_id)}",
    }
    print_json(out)
    return 0


def cmd_contract(args: argparse.Namespace) -> int:
    contract_id = require_contract_id(args.contract_id)

    data = _http_get(f"/api/v1/contracts/{contract_id}")
    if data is None:
        sys.stderr.write(f"error: contract {contract_id!r} not found\n")
        return 1

    bytecode = data.get("bytecode") or data.get("runtime_bytecode", "") or ""
    if bytecode.startswith("0x") or bytecode.startswith("0X"):
        bytecode = bytecode[2:]
    bytecode_size = len(bytecode) // 2

    balance_tinybar = int(data.get("balance") or 0)

    out: Dict[str, Any] = {
        "contract_id": data.get("contract_id"),
        "evm_address": data.get("evm_address"),
        "memo": data.get("memo") or None,
        "admin_key": bool(data.get("admin_key")),
        "auto_renew_account_id": data.get("auto_renew_account"),
        "auto_renew_period": data.get("auto_renew_period"),
        "bytecode_size_bytes": bytecode_size,
        "balance_hbar": round(tinybar_to_hbar(balance_tinybar), 8),
        "created_timestamp": _fmt_ts(str(data.get("created_timestamp", ""))),
        "hashscan_url": f"{_explorer_base()}/contract/{data.get('contract_id', contract_id)}",
    }
    print_json(out)
    return 0


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    global _NETWORK

    parser = argparse.ArgumentParser(
        prog="hedera_client.py",
        description="Read-only Hedera blockchain queries via the Mirror Node REST API.",
    )
    parser.add_argument(
        "--network",
        choices=["mainnet", "testnet"],
        default="mainnet",
        help="Hedera network to query (default: mainnet). Override with HEDERA_MIRROR_URL.",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("stats", help="Network health, latest block, HBAR price")

    p_acct = sub.add_parser("account", help="HBAR + HTS token balances")
    p_acct.add_argument("account_id", help="Account ID (0.0.XXXX) or EVM address")
    p_acct.add_argument(
        "--no-prices", action="store_true", help="Skip CoinGecko price lookups"
    )

    p_tok = sub.add_parser("token", help="HTS token metadata")
    p_tok.add_argument("token_id", help="Token ID (0.0.XXXX)")

    p_tx = sub.add_parser("tx", help="Transaction details")
    p_tx.add_argument(
        "tx_id",
        help="Transaction ID (0.0.X-SSSSSSSSSS-NNNNNNNNN or 0.0.X@SSSSSSSSSS.NNNNNNNNN)",
    )

    p_act = sub.add_parser("activity", help="Recent transactions for an account")
    p_act.add_argument("account_id", help="Account ID (0.0.XXXX) or EVM address")
    p_act.add_argument(
        "--limit", type=int, default=25, help="Number of transactions (max 100, default 25)"
    )

    p_nft = sub.add_parser("nft", help="NFTs held by an account")
    p_nft.add_argument("account_id", help="Account ID (0.0.XXXX) or EVM address")
    p_nft.add_argument(
        "--limit", type=int, default=50, help="Max NFTs to fetch (max 100, default 50)"
    )

    p_price = sub.add_parser("price", help="HBAR or HTS token price")
    p_price.add_argument(
        "token", help="'HBAR', a token symbol (e.g. SAUCE), or token ID (0.0.XXXX)"
    )

    sub.add_parser("fees", help="Common Hedera operation costs at live exchange rate")

    p_topic = sub.add_parser("topic", help="HCS topic metadata + recent messages")
    p_topic.add_argument("topic_id", help="Topic ID (0.0.XXXX)")
    p_topic.add_argument(
        "--messages", type=int, default=10,
        help="Number of recent messages to fetch (max 100, default 10)",
    )

    p_contract = sub.add_parser("contract", help="Smart contract info")
    p_contract.add_argument(
        "contract_id", help="Contract ID (0.0.XXXX) or EVM address"
    )

    args = parser.parse_args(argv)
    _NETWORK = args.network

    dispatch = {
        "stats":    cmd_stats,
        "account":  cmd_account,
        "token":    cmd_token,
        "tx":       cmd_tx,
        "activity": cmd_activity,
        "nft":      cmd_nft,
        "price":    cmd_price,
        "fees":     cmd_fees,
        "topic":    cmd_topic,
        "contract": cmd_contract,
    }
    return dispatch[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
