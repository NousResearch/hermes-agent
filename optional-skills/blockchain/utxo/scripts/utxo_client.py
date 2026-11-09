#!/usr/bin/env python3
"""
UTXO chain CLI for Hermes Agent
-------------------------------
Read-only Bitcoin, Litecoin, Dogecoin and Zcash lookups through the SoChain
REST API v3 (https://sochain.com/docs). Python standard library only.

Usage:
  python3 utxo_client.py balance  <address> [--network NET] [--time T | --height H]
  python3 utxo_client.py address  <address> [--network NET]
  python3 utxo_client.py txs      <address> [--network NET] [--page N]
                                  [--from-time T] [--to-time T] [--direction in|out|both]
                                  [--page-size N] [--cursor C]
  python3 utxo_client.py utxos    <address> [--network NET] [--page N]
  python3 utxo_client.py validate <address> [--network NET]
  python3 utxo_client.py tx       <txid> [--network NET]
  python3 utxo_client.py status   <txid> [--network NET]
  python3 utxo_client.py block    <hash_or_height> [--network NET]
  python3 utxo_client.py network  [--network NET]
  python3 utxo_client.py price    [--network NET] [--at UNIX_SECONDS]

Networks: BTC, LTC, DOGE, ZEC and the testnets BTCTEST (testnet4), LTCTEST,
DOGETEST, ZECTEST. Address commands infer mainnet networks from the address
prefix when --network is omitted; other commands default to BTC.

Environment:
  SOCHAIN_API_KEY  Optional. Sent as the API-KEY header for higher limits
                   (free account at https://sochain.com/dashboard). Without
                   it, lookups work up to 2,500 requests a day per IP.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Optional

API_BASE = "https://sochain.com/api/v3"
USER_AGENT = "HermesAgent/1.0"
KEY_ENV = "SOCHAIN_API_KEY"
DASHBOARD_URL = "https://sochain.com/dashboard"
NETWORKS = ("BTC", "LTC", "DOGE", "ZEC", "BTCTEST", "LTCTEST", "DOGETEST", "ZECTEST")
MAX_RETRY_WAIT = 30

# Longest prefix first: "tltc1" and "tb1" must win over the Zcash "t" prefixes.
_PREFIXES = (
    ("tltc1", "LTCTEST"),
    ("ltc1", "LTC"),
    ("tb1", "BTCTEST"),
    ("bc1", "BTC"),
    ("t1", "ZEC"),
    ("t3", "ZEC"),
    ("tm", "ZECTEST"),
    ("t2", "ZECTEST"),
    ("1", "BTC"),
    ("3", "BTC"),
    ("L", "LTC"),
    ("M", "LTC"),
    ("D", "DOGE"),
    ("A", "DOGE"),
    ("9", "DOGE"),
)


class SoChainError(Exception):
    """An error with a message that names the cause and what to do."""


def infer_network(address: str) -> str:
    """The network an address belongs to, from its prefix; testnet m/n/2 addresses are ambiguous."""
    lowered = address.lower()
    for prefix, network in _PREFIXES:
        candidate = lowered if prefix.islower() or prefix.isdigit() else address
        if candidate.startswith(prefix):
            return network
    raise SoChainError(
        f"can't tell the network of {address!r} from its prefix — pass --network "
        f"({', '.join(NETWORKS)})"
    )


def _resolve_network(network: Optional[str], address: Optional[str] = None) -> str:
    if network:
        upper = network.upper()
        if upper not in NETWORKS:
            raise SoChainError(f"unknown network {network!r} — use one of {', '.join(NETWORKS)}")
        return upper
    return infer_network(address) if address else "BTC"


def _headers() -> dict[str, str]:
    headers = {"Accept": "application/json", "User-Agent": USER_AGENT}
    key = os.environ.get(KEY_ENV, "").strip()
    if key:
        headers["API-KEY"] = key
    return headers


def _http_error(exc: urllib.error.HTTPError) -> SoChainError:
    try:
        body = json.loads(exc.read().decode("utf-8"))
        message = (body.get("data") or {}).get("error_message") or ""
    except (ValueError, AttributeError, OSError):
        message = ""
    if exc.code == 401:
        return SoChainError(
            f"SoChain rejected the API key in {KEY_ENV} — check it at {DASHBOARD_URL}, "
            "or unset it to use keyless access"
        )
    if exc.code == 429:
        return SoChainError(
            "SoChain rate limit reached (keyless: 2,500 requests a day per IP) — set "
            f"{KEY_ENV} to a key from a free account at {DASHBOARD_URL}"
        )
    if exc.code == 404:
        return SoChainError(f"not found: {(message or exc.reason).rstrip('.')} — check the network and the value")
    return SoChainError(f"SoChain HTTP {exc.code}: {message or exc.reason}")


def _retry_after(exc: urllib.error.HTTPError, attempt: int) -> float:
    header = exc.headers.get("Retry-After") if exc.headers else None
    try:
        wait = float(header) if header else 1.5 * (attempt + 1)
    except ValueError:
        wait = 1.5 * (attempt + 1)
    return min(max(wait, 0.0), MAX_RETRY_WAIT)


def _get(path: str, params: Optional[dict[str, Any]] = None, timeout: int = 20, retries: int = 2) -> Any:
    """GET a v3 path and return its ``data``; raise SoChainError on any failure."""
    query = {k: v for k, v in (params or {}).items() if v is not None}
    url = f"{API_BASE}{path}" + (f"?{urllib.parse.urlencode(query)}" if query else "")
    for attempt in range(retries + 1):
        request = urllib.request.Request(url, headers=_headers(), method="GET")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.load(response)
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt < retries:
                time.sleep(_retry_after(exc, attempt))
                continue
            raise _http_error(exc) from exc
        except urllib.error.URLError as exc:
            raise SoChainError(f"can't reach SoChain ({exc.reason}) — check the network connection") from exc
        except json.JSONDecodeError as exc:
            raise SoChainError(f"SoChain answered with invalid JSON: {exc}") from exc
        if body.get("status") != "success":
            message = (body.get("data") or {}).get("error_message") or "request failed"
            raise SoChainError(f"SoChain: {message}")
        return body.get("data")
    raise SoChainError("SoChain rate limit reached; retries exhausted")


def _seg(value: str) -> str:
    return urllib.parse.quote(str(value), safe="")


def cmd_balance(args: argparse.Namespace) -> Any:
    if args.time and args.height is not None:
        raise SoChainError("pass --time or --height, not both")
    net = _resolve_network(args.network, args.address)
    return _get(f"/balance/{net}/{_seg(args.address)}", {"time": args.time, "height": args.height})


def cmd_address(args: argparse.Namespace) -> Any:
    net = _resolve_network(args.network, args.address)
    return _get(f"/address_summary/{net}/{_seg(args.address)}")


def cmd_txs(args: argparse.Namespace) -> Any:
    net = _resolve_network(args.network, args.address)
    filters = {
        "from_time": args.from_time, "to_time": args.to_time, "direction": args.direction,
        "page_size": args.page_size, "cursor": args.cursor,
    }
    if any(v is not None for v in filters.values()):
        if args.page is not None:
            raise SoChainError("filtered history pages by --cursor (next_cursor), not --page")
        return _get(f"/transactions/{net}/{_seg(args.address)}", filters)
    page = f"/{args.page}" if args.page is not None else ""
    return _get(f"/transactions/{net}/{_seg(args.address)}{page}")


def cmd_utxos(args: argparse.Namespace) -> Any:
    net = _resolve_network(args.network, args.address)
    page = f"/{args.page}" if args.page is not None else ""
    return _get(f"/unspent_outputs/{net}/{_seg(args.address)}{page}")


def cmd_validate(args: argparse.Namespace) -> Any:
    net = _resolve_network(args.network, args.address)
    return {"network": net, **(_get(f"/is_valid_address/{net}/{_seg(args.address)}") or {})}


def cmd_tx(args: argparse.Namespace) -> Any:
    return _get(f"/transaction/{_resolve_network(args.network)}/{_seg(args.txid)}")


def cmd_status(args: argparse.Namespace) -> Any:
    return _get(f"/transaction_status/{_resolve_network(args.network)}/{_seg(args.txid)}")


def cmd_block(args: argparse.Namespace) -> Any:
    return _get(f"/block/{_resolve_network(args.network)}/{_seg(args.block)}")


def cmd_network(args: argparse.Namespace) -> Any:
    return _get(f"/network_info/{_resolve_network(args.network)}")


def cmd_price(args: argparse.Namespace) -> Any:
    at = f"/{args.at}" if args.at is not None else ""
    return _get(f"/price/{_resolve_network(args.network)}{at}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only BTC/LTC/DOGE/ZEC lookups via SoChain.")
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name: str, func, help_text: str, target: Optional[str] = None) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help_text)
        if target:
            p.add_argument(target)
        p.add_argument("--network", help=f"one of {', '.join(NETWORKS)}")
        p.set_defaults(func=func)
        return p

    p = add("balance", cmd_balance, "confirmed and pending balance", "address")
    p.add_argument("--time", help="balance at this time: unix seconds, YYYY-MM-DD or RFC 3339")
    p.add_argument("--height", type=int, help="balance at the end of this block")
    add("address", cmd_address, "sent, received, balance and counts", "address")
    p = add("txs", cmd_txs, "transactions, newest first", "address")
    p.add_argument("--page", type=int, help="page of 10 (unfiltered form)")
    p.add_argument("--from-time", help="from this time: unix seconds, YYYY-MM-DD or RFC 3339")
    p.add_argument("--to-time", help="up to this time")
    p.add_argument("--direction", choices=("in", "out", "both"))
    p.add_argument("--page-size", type=int, help="rows per filtered page, 1-100")
    p.add_argument("--cursor", help="next_cursor from the previous filtered page")
    p = add("utxos", cmd_utxos, "unspent outputs", "address")
    p.add_argument("--page", type=int)
    add("validate", cmd_validate, "is this a valid address on the network", "address")
    add("tx", cmd_tx, "one transaction with inputs and outputs", "txid")
    add("status", cmd_status, "pending, mined, replaced or dropped", "txid")
    add("block", cmd_block, "one block by height or hash", "block")
    add("network", cmd_network, "newest block, mempool and fee rates")
    p = add("price", cmd_price, "latest price, or at a past time")
    p.add_argument("--at", type=int, help="unix seconds")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        data = args.func(args)
    except SoChainError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(data, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
