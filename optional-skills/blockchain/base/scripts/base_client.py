#!/usr/bin/env python3
"""
Base Blockchain CLI Tool for Hermes Agent
------------------------------------------
Queries the Base (Ethereum L2) JSON-RPC API and CoinGecko for enriched on-chain data.
Uses only Python standard library — no external packages required.

Usage:
  python3 base_client.py stats
  python3 base_client.py wallet   <address> [--limit N] [--all] [--no-prices]
  python3 base_client.py tx       <hash>
  python3 base_client.py token    <contract_address>
  python3 base_client.py gas
  python3 base_client.py contract <address>
  python3 base_client.py whales   [--min-eth N]
  python3 base_client.py price    <contract_address_or_symbol>

Environment:
  BASE_RPC_URL  Override the default RPC endpoint (default: https://mainnet.base.org)
"""

import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional, Tuple

RPC_URL = os.environ.get(
    "BASE_RPC_URL",
    "https://mainnet.base.org",
)

WEI_PER_ETH = 10**18
GWEI = 10**9

# ERC-20 function selectors (first 4 bytes of keccak256 hash)
SEL_BALANCE_OF   = "70a08231"
SEL_NAME         = "06fdde03"
SEL_SYMBOL       = "95d89b41"
SEL_DECIMALS     = "313ce567"
SEL_TOTAL_SUPPLY = "18160ddd"

# ERC-165 supportsInterface(bytes4) selector
SEL_SUPPORTS_INTERFACE = "01ffc9a7"

# Interface IDs for ERC-165 detection
IFACE_ERC721  = "80ac58cd"
IFACE_ERC1155 = "d9b67a26"

# Transfer(address,address,uint256) event topic
TRANSFER_TOPIC = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"

# Well-known Base tokens — maps lowercase address -> (symbol, name, decimals).
KNOWN_TOKENS: Dict[str, Tuple[str, str, int]] = {
    "0x4200000000000000000000000000000000000006": ("WETH",   "Wrapped Ether",               18),
    "0x833589fcd6edb6e08f4c7c32d4f71b54bda02913": ("USDC",   "USD Coin",                     6),
    "0x2ae3f1ec7f1f5012cfeab0185bfc7aa3cf0dec22": ("cbETH",  "Coinbase Wrapped Staked ETH", 18),
    "0x940181a94a35a4569e4529a3cdfb74e38fd98631": ("AERO",   "Aerodrome Finance",           18),
    "0x4ed4e862860bed51a9570b96d89af5e1b0efefed": ("DEGEN",  "Degen",                       18),
    "0xac1bd2486aaf3b5c0fc3fd868558b082a531b2b4": ("TOSHI",  "Toshi",                       18),
    "0x532f27101965dd16442e59d40670faf5ebb142e4": ("BRETT",  "Brett",                       18),
    "0xa88594d404727625a9437c3f886c7643872296ae": ("WELL",   "Moonwell",                    18),
    "0xc1cba3fcea344f92d9239c08c0568f6f2f0ee452": ("wstETH", "Wrapped Lido Staked ETH",     18),
    "0xb6fe221fe9eef5aba221c348ba20a1bf5e73624c": ("rETH",   "Rocket Pool ETH",             18),
    "0xcbb7c0000ab88b473b1f5afd9ef808440eed33bf": ("cbBTC",  "Coinbase Wrapped BTC",         8),
}

# Reverse lookup: symbol -> contract address (for the `price` command).
_SYMBOL_TO_ADDRESS = {v[0].upper(): k for k, v in KNOWN_TOKENS.items()}
_SYMBOL_TO_ADDRESS["ETH"] = "ETH"


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

    _headers = {"Content-Type": "application/json", "User-Agent": "HermesAgent/1.0"}

    for attempt in range(retries + 1):
        req = urllib.request.Request(
            RPC_URL, data=payload, headers=_headers, method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                body = json.load(resp)
            if "error" in body:
                err = body["error"]
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


# Keep backward compat alias.
rpc = _rpc_call


_BATCH_LIMIT = 10  # Base public RPC limits to 10 calls per batch


def _rpc_batch_chunk(items: list) -> list:
    """Send a single batch of JSON-RPC requests (max _BATCH_LIMIT)."""
    payload = json.dumps(items).encode()
    _headers = {"Content-Type": "application/json", "User-Agent": "HermesAgent/1.0"}

    for attempt in range(3):
        req = urllib.request.Request(
            RPC_URL, data=payload, headers=_headers, method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.load(resp)
            # If the RPC returns an error dict instead of a list, treat as failure
            if isinstance(data, dict) and "error" in data:
                sys.exit(f"RPC batch error: {data['error']}")
            return data if isinstance(data, list) else []
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt < 2:
                time.sleep(1.5 * (attempt + 1))
                continue
            sys.exit(f"RPC batch HTTP error: {exc}")
        except urllib.error.URLError as exc:
            sys.exit(f"RPC batch error: {exc}")
    return []


def rpc_batch(calls: list) -> list:
    """Send a batch of JSON-RPC requests, auto-chunking to respect limits."""
    items = [
        {"jsonrpc": "2.0", "id": i, "method": c["method"], "params": c.get("params", [])}
        for i, c in enumerate(calls)
    ]

    if len(items) <= _BATCH_LIMIT:
        return _rpc_batch_chunk(items)

    # Split into chunks of _BATCH_LIMIT
    all_results = []
    for start in range(0, len(items), _BATCH_LIMIT):
        chunk = items[start:start + _BATCH_LIMIT]
        all_results.extend(_rpc_batch_chunk(chunk))
    return all_results


def wei_to_eth(wei: int) -> float:
    return wei / WEI_PER_ETH


def wei_to_gwei(wei: int) -> float:
    return wei / GWEI


def hex_to_int(hex_str: Optional[str]) -> int:
    """Convert hex string (0x...) to int. Returns 0 for None/empty."""
    if not hex_str or hex_str == "0x":
        return 0
    return int(hex_str, 16)


def print_json(obj: Any) -> None:
    print(json.dumps(obj, indent=2))


def _short_addr(addr: str) -> str:
    """Abbreviate an address for display: first 6 + last 4."""
    if len(addr) <= 14:
        return addr
    return f"{addr[:6]}...{addr[-4:]}"


# ---------------------------------------------------------------------------
# ABI encoding / decoding helpers
# ---------------------------------------------------------------------------

def _encode_address(addr: str) -> str:
    """ABI-encode an address as a 32-byte hex string (no 0x prefix)."""
    clean = addr.lower().replace("0x", "")
    return clean.zfill(64)


def _decode_uint(hex_data: Optional[str]) -> int:
    """Decode a hex-encoded uint256 return value."""
    if not hex_data or hex_data == "0x":
        return 0
    return int(hex_data.replace("0x", ""), 16)


def _decode_string(hex_data: Optional[str]) -> str:
    """Decode an ABI-encoded string return value."""
    if not hex_data or hex_data == "0x" or len(hex_data) < 130:
        return ""
    data = hex_data[2:] if hex_data.startswith("0x") else hex_data
    try:
        length = int(data[64:128], 16)
        if length == 0 or length > 256:
            return ""
        str_hex = data[128:128 + length * 2]
        return bytes.fromhex(str_hex).decode("utf-8").strip("\x00")
    except (ValueError, UnicodeDecodeError):
        return ""


def _eth_call(to: str, selector: str, args: str = "", block: str = "latest") -> Optional[str]:
    """Execute eth_call with a function selector. Returns None on revert/error."""
    data = "0x" + selector + args
    try:
        payload = json.dumps({
            "jsonrpc": "2.0", "id": 1,
            "method": "eth_call", "params": [{"to": to, "data": data}, block],
        }).encode()
        req = urllib.request.Request(
            RPC_URL, data=payload,
            headers={"Content-Type": "application/json", "User-Agent": "HermesAgent/1.0"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            body = json.load(resp)
        if "error" in body:
            return None
        return body.get("result")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Price & token name helpers (CoinGecko — free, no API key)
# ---------------------------------------------------------------------------

def fetch_prices(addresses: List[str], max_lookups: int = 20) -> Dict[str, float]:
    """Fetch USD prices for Base token addresses via CoinGecko (one per request).

    CoinGecko free tier doesn't support batch Base token lookups,
    so we do individual calls — capped at *max_lookups* to stay within
    rate limits. Returns {lowercase_address: usd_price}.
    """
    prices: Dict[str, float] = {}
    for i, addr in enumerate(addresses[:max_lookups]):
        url = (
            f"https://api.coingecko.com/api/v3/simple/token_price/base"
            f"?contract_addresses={addr}&vs_currencies=usd"
        )
        data = _http_get_json(url, timeout=10)
        if data and isinstance(data, dict):
            for key, info in data.items():
                if isinstance(info, dict) and "usd" in info:
                    prices[addr.lower()] = info["usd"]
                    break
        # Pause between calls to respect CoinGecko free-tier rate-limits
        if i < len(addresses[:max_lookups]) - 1:
            time.sleep(1.0)
    return prices


def fetch_eth_price() -> Optional[float]:
    """Fetch current ETH price in USD via CoinGecko."""
    data = _http_get_json(
        "https://api.coingecko.com/api/v3/simple/price?ids=ethereum&vs_currencies=usd"
    )
    if data and "ethereum" in data:
        return data["ethereum"].get("usd")
    return None


def resolve_token_name(addr: str) -> Optional[Dict[str, str]]:
    """Look up token name and symbol. Checks known tokens first, then on-chain.

    Returns {"name": ..., "symbol": ...} or None.
    """
    addr_lower = addr.lower()
    if addr_lower in KNOWN_TOKENS:
        sym, name, _ = KNOWN_TOKENS[addr_lower]
        return {"symbol": sym, "name": name}
    # Try reading name() and symbol() from the contract
    name_hex = _eth_call(addr, SEL_NAME)
    symbol_hex = _eth_call(addr, SEL_SYMBOL)
    name = _decode_string(name_hex) if name_hex else ""
    symbol = _decode_string(symbol_hex) if symbol_hex else ""
    if symbol:
        return {"symbol": symbol.upper(), "name": name}
    return None


def _token_label(addr: str) -> str:
    """Return a human-readable label: symbol if known, else abbreviated address."""
    addr_lower = addr.lower()
    if addr_lower in KNOWN_TOKENS:
        return KNOWN_TOKENS[addr_lower][0]
    return _short_addr(addr)


# ---------------------------------------------------------------------------
# 1. Network Stats
# ---------------------------------------------------------------------------

def cmd_stats(_args):
    """Base network health: block, gas, chain ID, ETH price."""
    results = rpc_batch([
        {"method": "eth_blockNumber"},
        {"method": "eth_gasPrice"},
        {"method": "eth_chainId"},
        {"method": "eth_getBlockByNumber", "params": ["latest", False]},
    ])

    by_id = {r["id"]: r.get("result") for r in results}

    block_num = hex_to_int(by_id.get(0))
    gas_price = hex_to_int(by_id.get(1))
    chain_id  = hex_to_int(by_id.get(2))
    block     = by_id.get(3) or {}

    base_fee  = hex_to_int(block.get("baseFeePerGas")) if block.get("baseFeePerGas") else None
    timestamp = hex_to_int(block.get("timestamp")) if block.get("timestamp") else None
    gas_used  = hex_to_int(block.get("gasUsed")) if block.get("gasUsed") else None
    gas_limit = hex_to_int(block.get("gasLimit")) if block.get("gasLimit") else None
    tx_count  = len(block.get("transactions", []))

    eth_price = fetch_eth_price()

    out = {
        "chain":            "Base" if chain_id == 8453 else f"Chain {chain_id}",
        "chain_id":         chain_id,
        "latest_block":     block_num,
        "gas_price_gwei":   round(wei_to_gwei(gas_price), 4),
    }
    if base_fee is not None:
        out["base_fee_gwei"] = round(wei_to_gwei(base_fee), 4)
    if timestamp:
        out["block_timestamp"] = timestamp
    if gas_used is not None and gas_limit:
        out["block_gas_used"]         = gas_used
        out["block_gas_limit"]        = gas_limit
        out["block_utilization_pct"]  = round(gas_used / gas_limit * 100, 2)
    out["block_tx_count"] = tx_count
    if eth_price is not None:
        out["eth_price_usd"] = eth_price
    print_json(out)


# ---------------------------------------------------------------------------
# 2. Wallet Info (ETH + ERC-20 balances with prices)
# ---------------------------------------------------------------------------

def cmd_wallet(args):
    """ETH balance + ERC-20 token holdings with USD values."""
    address  = args.address.lower()
    show_all = getattr(args, "all", False)
    limit    = getattr(args, "limit", 20) or 20
    skip_prices = getattr(args, "no_prices", False)

    # Batch: ETH balance + balanceOf for all known tokens
    calls = [{"method": "eth_getBalance", "params": [address, "latest"]}]
    token_addrs = list(KNOWN_TOKENS.keys())
    for token_addr in token_addrs:
        calls.append({
            "method": "eth_call",
            "params": [
                {"to": token_addr, "data": "0x" + SEL_BALANCE_OF + _encode_address(address)},
                "latest",
            ],
        })

    results = rpc_batch(calls)
    by_id = {r["id"]: r.get("result") for r in results}

    eth_balance = wei_to_eth(hex_to_int(by_id.get(0)))

    # Parse token balances
    tokens = []
    for i, token_addr in enumerate(token_addrs):
        raw = hex_to_int(by_id.get(i + 1))
        if raw == 0:
            continue
        sym, name, decimals = KNOWN_TOKENS[token_addr]
        amount = raw / (10 ** decimals)
        tokens.append({
            "address":  token_addr,
            "symbol":   sym,
            "name":     name,
            "amount":   amount,
            "decimals": decimals,
        })

    # Fetch prices
    eth_price = None
    prices: Dict[str, float] = {}
    if not skip_prices:
        eth_price = fetch_eth_price()
        if tokens:
            mints_to_price = [t["address"] for t in tokens]
            prices = fetch_prices(mints_to_price, max_lookups=20)

    # Enrich with USD values, filter dust, sort
    enriched = []
    dust_count = 0
    dust_value = 0.0
    for t in tokens:
        usd_price = prices.get(t["address"])
        usd_value = round(usd_price * t["amount"], 2) if usd_price else None

        if not show_all and usd_value is not None and usd_value < 0.01:
            dust_count += 1
            dust_value += usd_value
            continue

        entry = {"token": t["symbol"], "address": t["address"], "amount": t["amount"]}
        if usd_price is not None:
            entry["price_usd"] = usd_price
            entry["value_usd"] = usd_value
        enriched.append(entry)

    # Sort: tokens with known USD value first (highest->lowest), then unknowns
    enriched.sort(
        key=lambda x: (x.get("value_usd") is not None, x.get("value_usd") or 0),
        reverse=True,
    )

    # Apply limit unless --all
    total_tokens = len(enriched)
    if not show_all and len(enriched) > limit:
        enriched = enriched[:limit]
    hidden_tokens = total_tokens - len(enriched)

    # Compute portfolio total
    total_usd = sum(t.get("value_usd", 0) for t in enriched)
    eth_value_usd = round(eth_price * eth_balance, 2) if eth_price else None
    if eth_value_usd:
        total_usd += eth_value_usd
    total_usd += dust_value

    output = {
        "address":     args.address,
        "eth_balance": round(eth_balance, 18),
    }
    if eth_price:
        output["eth_price_usd"] = eth_price
        output["eth_value_usd"] = eth_value_usd
    output["tokens_shown"] = len(enriched)
    if hidden_tokens > 0:
        output["tokens_hidden"] = hidden_tokens
    output["erc20_tokens"] = enriched
    if dust_count > 0:
        output["dust_filtered"] = {"count": dust_count, "total_value_usd": round(dust_value, 4)}
    if total_usd > 0:
        output["portfolio_total_usd"] = round(total_usd, 2)
    if hidden_tokens > 0 and not show_all:
        output["warning"] = (
            "portfolio_total_usd may be partial because hidden tokens are not "
            "included when --limit is applied."
        )
    output["note"] = f"Checked {len(KNOWN_TOKENS)} known Base tokens. Unknown ERC-20s not shown."

    print_json(output)


# ---------------------------------------------------------------------------
# 3. Transaction Details
# ---------------------------------------------------------------------------

def cmd_tx(args):
    """Full transaction details by hash."""
    tx_hash = args.hash

    results = rpc_batch([
        {"method": "eth_getTransactionByHash", "params": [tx_hash]},
        {"method": "eth_getTransactionReceipt", "params": [tx_hash]},
    ])

    by_id = {r["id"]: r.get("result") for r in results}
    tx      = by_id.get(0)
    receipt = by_id.get(1)

    if tx is None:
        sys.exit("Transaction not found.")

    value_wei = hex_to_int(tx.get("value"))
    tx_gas_price = hex_to_int(tx.get("gasPrice"))
    gas_used = hex_to_int(receipt.get("gasUsed")) if receipt else None
    effective_gas_price = (
        hex_to_int(receipt.get("effectiveGasPrice")) if receipt and receipt.get("effectiveGasPrice")
        else tx_gas_price
    )
    l2_fee_wei = effective_gas_price * gas_used if gas_used is not None else None
    l1_fee_wei = hex_to_int(receipt.get("l1Fee")) if receipt and receipt.get("l1Fee") else 0
    fee_wei = (l2_fee_wei + l1_fee_wei) if l2_fee_wei is not None else None

    eth_price = fetch_eth_price()

    out = {
        "hash":           tx_hash,
        "block":          hex_to_int(tx.get("blockNumber")),
        "from":           tx.get("from"),
        "to":             tx.get("to"),
        "value_ETH":      round(wei_to_eth(value_wei), 18) if value_wei else 0,
        "gas_price_gwei": round(wei_to_gwei(effective_gas_price), 4),
    }
    if gas_used is not None:
        out["gas_used"] = gas_used
    if l2_fee_wei is not None:
        out["l2_fee_ETH"] = round(wei_to_eth(l2_fee_wei), 12)
    if l1_fee_wei:
        out["l1_fee_ETH"] = round(wei_to_eth(l1_fee_wei), 12)
    if fee_wei is not None:
        out["fee_ETH"] = round(wei_to_eth(fee_wei), 12)
    if receipt:
        out["status"] = "success" if receipt.get("status") == "0x1" else "failed"
        out["contract_created"] = receipt.get("contractAddress")
        out["log_count"] = len(receipt.get("logs", []))

    # Decode ERC-20 transfers from logs
    transfers = []
    if receipt:
        for log in receipt.get("logs", []):
            topics = log.get("topics", [])
            if len(topics) >= 3 and topics[0] == TRANSFER_TOPIC:
                from_addr = "0x" + topics[1][-40:]
                to_addr   = "0x" + topics[2][-40:]
                token_contract = log.get("address", "")
                label = _token_label(token_contract)

                entry = {
                    "token":    label,
                    "contract": token_contract,
                    "from":     from_addr,
                    "to":       to_addr,
                }
                # ERC-20: 3 topics, amount in data
                if len(topics) == 3:
                    amount_hex = log.get("data", "0x")
                    if amount_hex and amount_hex != "0x":
                        raw_amount = hex_to_int(amount_hex)
                        addr_lower = token_contract.lower()
                        if addr_lower in KNOWN_TOKENS:
                            decimals = KNOWN_TOKENS[addr_lower][2]
                            entry["amount"] = raw_amount / (10 ** decimals)
                        else:
                            entry["raw_amount"] = raw_amount
                # ERC-721: 4 topics, tokenId in topics[3]
                elif len(topics) == 4:
                    entry["token_id"] = hex_to_int(topics[3])
                    entry["type"] = "ERC-721"

                transfers.append(entry)

    if transfers:
        out["token_transfers"] = transfers

    if eth_price is not None:
        if value_wei:
            out["value_USD"] = round(wei_to_eth(value_wei) * eth_price, 2)
        if l2_fee_wei is not None:
            out["l2_fee_USD"] = round(wei_to_eth(l2_fee_wei) * eth_price, 4)
        if l1_fee_wei:
            out["l1_fee_USD"] = round(wei_to_eth(l1_fee_wei) * eth_price, 4)
        if fee_wei is not None:
            out["fee_USD"] = round(wei_to_eth(fee_wei) * eth_price, 4)

    print_json(out)


# ---------------------------------------------------------------------------
# 4. Token Info
# ---------------------------------------------------------------------------

def cmd_token(args):
    """ERC-20 token metadata, supply, price, market cap."""
    addr = args.address.lower()

    # Batch: name, symbol, decimals, totalSupply, code check
    calls = [
        {"method": "eth_call", "params": [{"to": addr, "data": "0x" + SEL_NAME}, "latest"]},
        {"method": "eth_call", "params": [{"to": addr, "data": "0x" + SEL_SYMBOL}, "latest"]},
        {"method": "eth_call", "params": [{"to": addr, "data": "0x" + SEL_DECIMALS}, "latest"]},
        {"method": "eth_call", "params": [{"to": addr, "data": "0x" + SEL_TOTAL_SUPPLY}, "latest"]},
        {"method": "eth_getCode", "params": [addr, "latest"]},
    ]
    results = rpc_batch(calls)
    by_id = {r["id"]: r.get("result") for r in results}

    code = by_id.get(4)
    if not code or code == "0x":
        sys.exit("Address is not a contract.")

    name     = _decode_string(by_id.get(0))
    symbol   = _decode_string(by_id.get(1))
    decimals_raw = by_id.get(2)
    decimals = _decode_uint(decimals_raw)
    total_supply_raw = _decode_uint(by_id.get(3))

    # Fall back to known tokens if on-chain read failed
    if not symbol and addr in KNOWN_TOKENS:
        symbol   = KNOWN_TOKENS[addr][0]
        name     = KNOWN_TOKENS[addr][1]
        decimals = KNOWN_TOKENS[addr][2]

    is_known_token = addr in KNOWN_TOKENS
    is_erc20 = bool((symbol or is_known_token) and decimals_raw and decimals_raw != "0x")
    if not is_erc20:
        sys.exit("Contract does not appear to be an ERC-20 token.")

    total_supply = total_supply_raw / (10 ** decimals) if decimals else total_supply_raw

    # Fetch price
    price_data = fetch_prices([addr])

    out = {"address": args.address}
    if name:
        out["name"] = name
    if symbol:
        out["symbol"] = symbol
    out["decimals"]    = decimals
    out["total_supply"] = round(total_supply, min(decimals, 6))
    out["code_size_bytes"] = (len(code) - 2) // 2
    if addr in price_data:
        out["price_usd"]      = price_data[addr]
        out["market_cap_usd"] = round(price_data[addr] * total_supply, 0)

    print_json(out)


# ---------------------------------------------------------------------------
# 5. Gas Analysis (Base-specific: L2 execution + L1 data costs)
# ---------------------------------------------------------------------------

def cmd_gas(_args):
    """Detailed gas analysis with L1 data fee context and cost estimates."""
    latest_hex = _rpc_call("eth_blockNumber")
    latest = hex_to_int(latest_hex)

    # Get last 10 blocks for trend analysis + current gas price
    block_calls = []
    for i in range(10):
        block_calls.append({
            "method": "eth_getBlockByNumber",
            "params": [hex(latest - i), False],
        })
    block_calls.append({"method": "eth_gasPrice"})

    results = rpc_batch(block_calls)
    by_id = {r["id"]: r.get("result") for r in results}

    current_gas_price = hex_to_int(by_id.get(10))

    base_fees = []
    gas_utilizations = []
    tx_counts = []
    latest_block_info = None

    for i in range(10):
        b = by_id.get(i)
        if not b:
            continue
        bf  = hex_to_int(b.get("baseFeePerGas", "0x0"))
        gu  = hex_to_int(b.get("gasUsed", "0x0"))
        gl  = hex_to_int(b.get("gasLimit", "0x0"))
        txc = len(b.get("transactions", []))
        base_fees.append(bf)
        if gl > 0:
            gas_utilizations.append(gu / gl * 100)
        tx_counts.append(txc)

        if i == 0:
            latest_block_info = {
                "block":            hex_to_int(b.get("number")),
                "base_fee_gwei":    round(wei_to_gwei(bf), 6),
                "gas_used":         gu,
                "gas_limit":        gl,
                "utilization_pct":  round(gu / gl * 100, 2) if gl > 0 else 0,
                "tx_count":         txc,
            }

    avg_base_fee    = sum(base_fees) / len(base_fees) if base_fees else 0
    avg_utilization = sum(gas_utilizations) / len(gas_utilizations) if gas_utilizations else 0
    avg_tx_count    = sum(tx_counts) / len(tx_counts) if tx_counts else 0

    # Estimate costs for common operations
    eth_price = fetch_eth_price()

    simple_transfer_gas = 21_000
    erc20_transfer_gas  = 65_000
    swap_gas            = 200_000

    def _estimate_cost(gas: int) -> Dict[str, Any]:
        cost_wei = gas * current_gas_price
        cost_eth = wei_to_eth(cost_wei)
        entry: Dict[str, Any] = {"gas_units": gas, "cost_ETH": round(cost_eth, 10)}
        if eth_price:
            entry["cost_USD"] = round(cost_eth * eth_price, 6)
        return entry

    out: Dict[str, Any] = {
        "current_gas_price_gwei": round(wei_to_gwei(current_gas_price), 6),
        "latest_block":           latest_block_info,
        "trend_10_blocks": {
            "avg_base_fee_gwei":    round(wei_to_gwei(avg_base_fee), 6),
            "avg_utilization_pct":  round(avg_utilization, 2),
            "avg_tx_count":         round(avg_tx_count, 1),
            "min_base_fee_gwei":    round(wei_to_gwei(min(base_fees)), 6) if base_fees else None,
            "max_base_fee_gwei":    round(wei_to_gwei(max(base_fees)), 6) if base_fees else None,
        },
        "cost_estimates": {
            "eth_transfer":   _estimate_cost(simple_transfer_gas),
            "erc20_transfer": _estimate_cost(erc20_transfer_gas),
            "swap":           _estimate_cost(swap_gas),
        },
        "note": "Base is an L2. Total tx cost = L2 execution fee + L1 data posting fee. "
                "L1 data fee depends on calldata size and L1 gas prices (not shown here). "
                "Actual costs may be slightly higher than estimates.",
    }
    if eth_price:
        out["eth_price_usd"] = eth_price
    print_json(out)


# ---------------------------------------------------------------------------
# 6. Contract Inspection
# ---------------------------------------------------------------------------

def cmd_contract(args):
    """Inspect an address: EOA vs contract, ERC type detection, proxy resolution."""
    addr = args.address.lower()

    # Batch: getCode, getBalance, name, symbol, decimals, totalSupply, ERC-721, ERC-1155
    calls = [
        {"method": "eth_getCode",    "params": [addr, "latest"]},
        {"method": "eth_getBalance", "params": [addr, "latest"]},
        {"method": "eth_call", "params": [{"to": addr, "data": "0x" + SEL_NAME}, "latest"]},
        {"method": "eth_call", "params": [{"to": addr, "data": "0x" + SEL_SYMBOL}, "latest"]},
        {"method": "eth_call", "params": [{"to": addr, "data": "0x" + SEL_DECIMALS}, "latest"]},
        {"method": "eth_call", "params": [{"to": addr, "data": "0x" + SEL_TOTAL_SUPPLY}, "latest"]},
        {"method": "eth_call", "params": [
            {"to": addr, "data": "0x" + SEL_SUPPORTS_INTERFACE + IFACE_ERC721.zfill(64)},
            "latest",
        ]},
        {"method": "eth_call", "params": [
            {"to": addr, "data": "0x" + SEL_SUPPORTS_INTERFACE + IFACE_ERC1155.zfill(64)},
            "latest",
        ]},
    ]
    results = rpc_batch(calls)

    # Handle per-item errors gracefully
    by_id: Dict[int, Any] = {}
    for r in results:
        if "error" not in r:
            by_id[r["id"]] = r.get("result")
        else:
            by_id[r["id"]] = None

    code        = by_id.get(0, "0x")
    eth_balance = hex_to_int(by_id.get(1))

    if not code or code == "0x":
        out = {
            "address":     args.address,
            "is_contract": False,
            "eth_balance": round(wei_to_eth(eth_balance), 18),
            "note":        "This is an externally owned account (EOA), not a contract.",
        }
        print_json(out)
        return

    code_size = (len(code) - 2) // 2

    # Check ERC-20
    name         = _decode_string(by_id.get(2))
    symbol       = _decode_string(by_id.get(3))
    decimals_raw = by_id.get(4)
    supply_raw   = by_id.get(5)
    is_erc20     = bool(symbol and decimals_raw and decimals_raw != "0x")

    # Check ERC-721 / ERC-1155 via ERC-165
    erc721_result  = by_id.get(6)
    erc1155_result = by_id.get(7)
    is_erc721  = erc721_result is not None and _decode_uint(erc721_result) == 1
    is_erc1155 = erc1155_result is not None and _decode_uint(erc1155_result) == 1

    # Detect proxy pattern (EIP-1967 implementation slot)
    impl_slot = "0x360894a13ba1a3210667c828492db98dca3e2076cc3735a920a3ca505d382bbc"
    impl_result = _rpc_call("eth_getStorageAt", [addr, impl_slot, "latest"])
    is_proxy = False
    impl_address = None
    if impl_result and impl_result != "0x" + "0" * 64:
        impl_address = "0x" + impl_result[-40:]
        if impl_address != "0x" + "0" * 40:
            is_proxy = True

    out: Dict[str, Any] = {
        "address":        args.address,
        "is_contract":    True,
        "code_size_bytes": code_size,
        "eth_balance":    round(wei_to_eth(eth_balance), 18),
    }

    interfaces = []
    if is_erc20:
        interfaces.append("ERC-20")
    if is_erc721:
        interfaces.append("ERC-721")
    if is_erc1155:
        interfaces.append("ERC-1155")
    if interfaces:
        out["detected_interfaces"] = interfaces

    if is_erc20:
        decimals = _decode_uint(decimals_raw)
        supply   = _decode_uint(supply_raw)
        out["erc20"] = {
            "name":         name,
            "symbol":       symbol,
            "decimals":     decimals,
            "total_supply": supply / (10 ** decimals) if decimals else supply,
        }

    if is_proxy:
        out["proxy"] = {
            "is_proxy":       True,
            "implementation": impl_address,
            "standard":       "EIP-1967",
        }

    # Check known tokens
    if addr in KNOWN_TOKENS:
        sym, tname, _ = KNOWN_TOKENS[addr]
        out["known_token"] = {"symbol": sym, "name": tname}

    print_json(out)


# ---------------------------------------------------------------------------
# 7. Whale Detector
# ---------------------------------------------------------------------------

def cmd_whales(args):
    """Scan the latest block for large ETH transfers with USD values."""
    min_wei = int(args.min_eth * WEI_PER_ETH)

    block = rpc("eth_getBlockByNumber", ["latest", True])
    if block is None:
        sys.exit("Could not retrieve latest block.")

    eth_price = fetch_eth_price()

    whales = []
    for tx in (block.get("transactions") or []):
        value = hex_to_int(tx.get("value"))
        if value >= min_wei:
            entry: Dict[str, Any] = {
                "hash": tx.get("hash"),
                "from": tx.get("from"),
                "to":   tx.get("to"),
                "value_ETH": round(wei_to_eth(value), 6),
            }
            if eth_price:
                entry["value_USD"] = round(wei_to_eth(value) * eth_price, 2)
            whales.append(entry)

    # Sort by value descending
    whales.sort(key=lambda x: x["value_ETH"], reverse=True)

    out: Dict[str, Any] = {
        "block":              hex_to_int(block.get("number")),
        "block_time":         hex_to_int(block.get("timestamp")),
        "min_threshold_ETH":  args.min_eth,
        "large_transfers":    whales,
        "note":               "Scans latest block only — point-in-time snapshot.",
    }
    if eth_price:
        out["eth_price_usd"] = eth_price
    print_json(out)


# ---------------------------------------------------------------------------
# 8. Price Lookup
# ---------------------------------------------------------------------------

def cmd_price(args):
    """Quick price lookup for a token by contract address or known symbol."""
    query = args.token

    # Check if it's a known symbol
    addr = _SYMBOL_TO_ADDRESS.get(query.upper(), query).lower()

    # Special case: ETH itself
    if addr == "eth":
        eth_price = fetch_eth_price()
        out: Dict[str, Any] = {"query": query, "token": "ETH", "name": "Ethereum"}
        if eth_price:
            out["price_usd"] = eth_price
        else:
            out["price_usd"] = None
            out["note"] = "Price not available."
        print_json(out)
        return

    # Resolve name
    token_meta = resolve_token_name(addr)

    # Fetch price
    prices = fetch_prices([addr])

    out = {"query": query, "address": addr}
    if token_meta:
        out["name"]   = token_meta["name"]
        out["symbol"] = token_meta["symbol"]
    if addr in prices:
        out["price_usd"] = prices[addr]
    else:
        out["price_usd"] = None
        out["note"] = "Price not available — token may not be listed on CoinGecko."
    print_json(out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        prog="base_client.py",
        description="Base blockchain query tool for Hermes Agent",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("stats", help="Network stats: block, gas, chain ID, ETH price")

    p_wallet = sub.add_parser("wallet", help="ETH balance + ERC-20 tokens with USD values")
    p_wallet.add_argument("address")
    p_wallet.add_argument("--limit", type=int, default=20,
                          help="Max tokens to display (default: 20)")
    p_wallet.add_argument("--all", action="store_true",
                          help="Show all tokens (no limit, no dust filter)")
    p_wallet.add_argument("--no-prices", action="store_true",
                          help="Skip price lookups (faster, RPC-only)")

    p_tx = sub.add_parser("tx", help="Transaction details by hash")
    p_tx.add_argument("hash")

    p_token = sub.add_parser("token", help="ERC-20 token metadata, price, and market cap")
    p_token.add_argument("address")

    sub.add_parser("gas", help="Gas analysis with cost estimates and L1 data fee context")

    p_contract = sub.add_parser("contract", help="Contract inspection: type detection, proxy check")
    p_contract.add_argument("address")

    p_whales = sub.add_parser("whales", help="Large ETH transfers in the latest block")
    p_whales.add_argument("--min-eth", type=float, default=1.0,
                          help="Minimum ETH transfer size (default: 1.0)")

    p_price = sub.add_parser("price", help="Quick price lookup by address or symbol")
    p_price.add_argument("token", help="Contract address or known symbol (ETH, USDC, AERO, ...)")

    args = parser.parse_args()

    dispatch = {
        "stats":    cmd_stats,
        "wallet":   cmd_wallet,
        "tx":       cmd_tx,
        "token":    cmd_token,
        "gas":      cmd_gas,
        "contract": cmd_contract,
        "whales":   cmd_whales,
        "price":    cmd_price,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
