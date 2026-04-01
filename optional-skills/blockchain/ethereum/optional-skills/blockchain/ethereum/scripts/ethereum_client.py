#!/usr/bin/env python3
"""
Ethereum Mainnet CLI Tool for Hermes Agent
-------------------------------------------
Queries the Ethereum mainnet JSON-RPC API and CoinGecko for enriched on-chain data.
Uses only Python standard library — no external packages required.

Usage:
  python3 ethereum_client.py stats
  python3 ethereum_client.py wallet   <address_or_ens> [--limit N] [--all] [--no-prices]
  python3 ethereum_client.py tx       <hash>
  python3 ethereum_client.py token    <contract_address>
  python3 ethereum_client.py gas
  python3 ethereum_client.py contract <address>
  python3 ethereum_client.py whales   [--min-eth N]
  python3 ethereum_client.py ens      <name_or_address>
  python3 ethereum_client.py price    <contract_address_or_symbol>

Environment:
  ETH_RPC_URL  Override the default RPC endpoint (default: https://ethereum.publicnode.com)
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
    "ETH_RPC_URL",
    "https://ethereum.publicnode.com",
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

# ENS Public Resolver — addr(bytes32) selector
ENS_REGISTRY   = "0x00000000000C2E074eC69A0dFb2997BA6C7d2e1e"
SEL_RESOLVER   = "0178b8bf"  # resolver(bytes32)
SEL_ADDR       = "3b3b57de"  # addr(bytes32)
SEL_NAME_ENS   = "691f3431"  # name(bytes32) — reverse resolution

# Well-known Ethereum mainnet tokens
KNOWN_TOKENS: Dict[str, Tuple[str, str, int]] = {
    "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2": ("WETH",   "Wrapped Ether",               18),
    "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48": ("USDC",   "USD Coin",                     6),
    "0xdac17f958d2ee523a2206206994597c13d831ec7": ("USDT",   "Tether USD",                   6),
    "0x6b175474e89094c44da98b954eedeac495271d0f": ("DAI",    "Dai Stablecoin",               18),
    "0x2260fac5e5542a773aa44fbcfedf7c193bc2c599": ("WBTC",   "Wrapped Bitcoin",               8),
    "0x1f9840a85d5af5bf1d1762f925bdaddc4201f984": ("UNI",    "Uniswap",                     18),
    "0x7fc66500c84a76ad7e9c93437bfc5ac33e2ddae9": ("AAVE",   "Aave",                        18),
    "0x514910771af9ca656af840dff83e8264ecf986ca": ("LINK",   "Chainlink",                   18),
    "0xae7ab96520de3a18e5e111b5eaab095312d7fe84": ("stETH",  "Lido Staked Ether",           18),
    "0x7f39c581f595b53c5cb19bd0b3f8da6c935e2ca0": ("wstETH", "Wrapped Lido Staked ETH",     18),
    "0xbe9895146f7af43049ca1c1ae358b0541ea49704": ("cbETH",  "Coinbase Wrapped Staked ETH", 18),
    "0xd533a949740bb3306d119cc777fa900ba034cd52": ("CRV",    "Curve DAO Token",             18),
    "0x9f8f72aa9304c8b593d555f12ef6589cc3a579a2": ("MKR",    "Maker",                       18),
    "0xc00e94cb662c3520282e6f5717214004a7f26888": ("COMP",   "Compound",                    18),
    "0x0bc529c00c6401aef6d220be8c6ea1667f6ad93e": ("YFI",    "yearn.finance",               18),
    "0xba100000625a3754423978a60c9317c58a424e3d": ("BAL",    "Balancer",                    18),
    "0x111111111117dc0aa78b770fa6a738034120c302": ("1INCH",  "1inch",                       18),
    "0x5a98fcbea516cf06857215779fd812ca3bef1b32": ("LDO",    "Lido DAO",                    18),
    "0xb50721bcf8d664c30412cfbc6cf7a15145234ad1": ("ARB",    "Arbitrum",                    18),
    "0x4200000000000000000000000000000000000042": ("OP",     "Optimism",                    18),
}

# Reverse lookup: symbol -> contract address
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


rpc = _rpc_call

_BATCH_LIMIT = 10


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
    if not hex_str or hex_str == "0x":
        return 0
    return int(hex_str, 16)


def print_json(obj: Any) -> None:
    print(json.dumps(obj, indent=2))


def _short_addr(addr: str) -> str:
    if len(addr) <= 14:
        return addr
    return f"{addr[:6]}...{addr[-4:]}"


# ---------------------------------------------------------------------------
# ABI encoding / decoding helpers
# ---------------------------------------------------------------------------

def _encode_address(addr: str) -> str:
    clean = addr.lower().replace("0x", "")
    return clean.zfill(64)


def _decode_uint(hex_data: Optional[str]) -> int:
    if not hex_data or hex_data == "0x":
        return 0
    return int(hex_data.replace("0x", ""), 16)


def _decode_string(hex_data: Optional[str]) -> str:
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
# Price helpers (CoinGecko)
# ---------------------------------------------------------------------------

def fetch_prices(addresses: List[str], max_lookups: int = 20) -> Dict[str, float]:
    """Fetch USD prices for Ethereum token addresses via CoinGecko."""
    prices: Dict[str, float] = {}
    for i, addr in enumerate(addresses[:max_lookups]):
        url = (
            f"https://api.coingecko.com/api/v3/simple/token_price/ethereum"
            f"?contract_addresses={addr}&vs_currencies=usd"
        )
        data = _http_get_json(url, timeout=10)
        if data and isinstance(data, dict):
            for key, info in data.items():
                if isinstance(info, dict) and "usd" in info:
                    prices[addr.lower()] = info["usd"]
                    break
        if i < len(addresses[:max_lookups]) - 1:
            time.sleep(1.0)
    return prices


def fetch_eth_price() -> Optional[float]:
    data = _http_get_json(
        "https://api.coingecko.com/api/v3/simple/price?ids=ethereum&vs_currencies=usd"
    )
    if data and "ethereum" in data:
        return data["ethereum"].get("usd")
    return None


def resolve_token_name(addr: str) -> Optional[Dict[str, str]]:
    addr_lower = addr.lower()
    if addr_lower in KNOWN_TOKENS:
        sym, name, _ = KNOWN_TOKENS[addr_lower]
        return {"symbol": sym, "name": name}
    name_hex = _eth_call(addr, SEL_NAME)
    symbol_hex = _eth_call(addr, SEL_SYMBOL)
    name = _decode_string(name_hex) if name_hex else ""
    symbol = _decode_string(symbol_hex) if symbol_hex else ""
    if symbol:
        return {"symbol": symbol.upper(), "name": name}
    return None


def _token_label(addr: str) -> str:
    addr_lower = addr.lower()
    if addr_lower in KNOWN_TOKENS:
        return KNOWN_TOKENS[addr_lower][0]
    return _short_addr(addr)


# ---------------------------------------------------------------------------
# ENS helpers
# ---------------------------------------------------------------------------

def _namehash(name: str) -> str:
    """Compute ENS namehash for a domain name."""
    node = b"\x00" * 32
    if name:
        labels = name.split(".")
        for label in reversed(labels):
            label_hash = _keccak256(label.encode())
            node = _keccak256(node + label_hash)
    return "0x" + node.hex()


def _keccak256(data: bytes) -> bytes:
    """Minimal keccak256 using hashlib (Python 3.6+ ships with SHA-3 / keccak support)."""
    import hashlib
    return hashlib.new("sha3_256", data).digest()


def resolve_ens_name(name: str) -> Optional[str]:
    """Resolve an ENS name to an Ethereum address."""
    node = _namehash(name)
    node_hex = node[2:].zfill(64)

    # Step 1: get resolver from ENS registry
    resolver_result = _eth_call(ENS_REGISTRY, SEL_RESOLVER, node_hex)
    if not resolver_result or resolver_result == "0x" + "0" * 64:
        return None
    resolver_addr = "0x" + resolver_result[-40:]
    if resolver_addr == "0x" + "0" * 40:
        return None

    # Step 2: call addr(bytes32) on the resolver
    addr_result = _eth_call(resolver_addr, SEL_ADDR, node_hex)
    if not addr_result or addr_result == "0x" + "0" * 64:
        return None
    resolved = "0x" + addr_result[-40:]
    if resolved == "0x" + "0" * 40:
        return None
    return resolved


def reverse_resolve_ens(address: str) -> Optional[str]:
    """Reverse-resolve an Ethereum address to an ENS name."""
    # Reverse resolution uses <address_without_0x>.addr.reverse
    addr_clean = address.lower().replace("0x", "")
    reverse_name = f"{addr_clean}.addr.reverse"
    node = _namehash(reverse_name)
    node_hex = node[2:].zfill(64)

    # Get resolver
    resolver_result = _eth_call(ENS_REGISTRY, SEL_RESOLVER, node_hex)
    if not resolver_result or resolver_result == "0x" + "0" * 64:
        return None
    resolver_addr = "0x" + resolver_result[-40:]
    if resolver_addr == "0x" + "0" * 40:
        return None

    # Call name(bytes32)
    name_result = _eth_call(resolver_addr, SEL_NAME_ENS, node_hex)
    if not name_result:
        return None
    return _decode_string(name_result) or None


# ---------------------------------------------------------------------------
# 1. Network Stats
# ---------------------------------------------------------------------------

def cmd_stats(_args):
    """Ethereum mainnet health: block, gas, EIP-1559 base fee, ETH price."""
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

    out: Dict[str, Any] = {
        "chain":          "Ethereum Mainnet" if chain_id == 1 else f"Chain {chain_id}",
        "chain_id":       chain_id,
        "latest_block":   block_num,
        "gas_price_gwei": round(wei_to_gwei(gas_price), 4),
    }
    if base_fee is not None:
        out["base_fee_gwei"]     = round(wei_to_gwei(base_fee), 4)
        out["priority_fee_gwei"] = round(wei_to_gwei(max(0, gas_price - base_fee)), 4)
    if timestamp:
        out["block_timestamp"] = timestamp
    if gas_used is not None and gas_limit:
        out["block_gas_used"]        = gas_used
        out["block_gas_limit"]       = gas_limit
        out["block_utilization_pct"] = round(gas_used / gas_limit * 100, 2)
    out["block_tx_count"] = tx_count
    if eth_price is not None:
        out["eth_price_usd"] = eth_price
    print_json(out)


# ---------------------------------------------------------------------------
# 2. Wallet Info
# ---------------------------------------------------------------------------

def cmd_wallet(args):
    """ETH balance + ERC-20 token holdings with USD values. Accepts ENS names."""
    address_input = args.address
    show_all   = getattr(args, "all", False)
    limit      = getattr(args, "limit", 20) or 20
    skip_prices = getattr(args, "no_prices", False)

    # Resolve ENS name if needed
    address = address_input
    ens_name = None
    if not address_input.startswith("0x"):
        resolved = resolve_ens_name(address_input)
        if not resolved:
            sys.exit(f"Could not resolve ENS name: {address_input}")
        ens_name = address_input
        address  = resolved
    else:
        ens_name = reverse_resolve_ens(address_input)

    address = address.lower()

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

    eth_price = None
    prices: Dict[str, float] = {}
    if not skip_prices:
        eth_price = fetch_eth_price()
        if tokens:
            prices = fetch_prices([t["address"] for t in tokens], max_lookups=20)

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

        entry: Dict[str, Any] = {"token": t["symbol"], "address": t["address"], "amount": t["amount"]}
        if usd_price is not None:
            entry["price_usd"] = usd_price
            entry["value_usd"] = usd_value
        enriched.append(entry)

    enriched.sort(
        key=lambda x: (x.get("value_usd") is not None, x.get("value_usd") or 0),
        reverse=True,
    )

    total_tokens = len(enriched)
    if not show_all and len(enriched) > limit:
        enriched = enriched[:limit]
    hidden_tokens = total_tokens - len(enriched)

    total_usd = sum(t.get("value_usd", 0) for t in enriched)
    eth_value_usd = round(eth_price * eth_balance, 2) if eth_price else None
    if eth_value_usd:
        total_usd += eth_value_usd
    total_usd += dust_value

    output: Dict[str, Any] = {"address": args.address}
    if ens_name:
        output["ens_name"] = ens_name
    output["eth_balance"] = round(eth_balance, 18)
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
    output["note"] = f"Checked {len(KNOWN_TOKENS)} known Ethereum tokens. Unknown ERC-20s not shown."
    print_json(output)


# ---------------------------------------------------------------------------
# 3. Transaction Details
# ---------------------------------------------------------------------------

def cmd_tx(args):
    """Full transaction details by hash, including EIP-1559 fields."""
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

    value_wei  = hex_to_int(tx.get("value"))
    gas_used   = hex_to_int(receipt.get("gasUsed")) if receipt else None
    effective_gas_price = (
        hex_to_int(receipt.get("effectiveGasPrice"))
        if receipt and receipt.get("effectiveGasPrice")
        else hex_to_int(tx.get("gasPrice"))
    )
    fee_wei = effective_gas_price * gas_used if gas_used is not None else None

    # EIP-1559 fields
    max_fee_per_gas          = hex_to_int(tx.get("maxFeePerGas")) if tx.get("maxFeePerGas") else None
    max_priority_fee_per_gas = hex_to_int(tx.get("maxPriorityFeePerGas")) if tx.get("maxPriorityFeePerGas") else None

    eth_price = fetch_eth_price()

    out: Dict[str, Any] = {
        "hash":           tx_hash,
        "block":          hex_to_int(tx.get("blockNumber")),
        "from":           tx.get("from"),
        "to":             tx.get("to"),
        "value_ETH":      round(wei_to_eth(value_wei), 18) if value_wei else 0,
        "effective_gas_price_gwei": round(wei_to_gwei(effective_gas_price), 4),
    }
    if max_fee_per_gas is not None:
        out["max_fee_per_gas_gwei"] = round(wei_to_gwei(max_fee_per_gas), 4)
    if max_priority_fee_per_gas is not None:
        out["max_priority_fee_gwei"] = round(wei_to_gwei(max_priority_fee_per_gas), 4)
    if gas_used is not None:
        out["gas_used"] = gas_used
    if fee_wei is not None:
        out["fee_ETH"] = round(wei_to_eth(fee_wei), 12)
    if receipt:
        out["status"]           = "success" if receipt.get("status") == "0x1" else "failed"
        out["contract_created"] = receipt.get("contractAddress")
        out["log_count"]        = len(receipt.get("logs", []))

    # Decode ERC-20 / ERC-721 transfers from logs
    transfers = []
    if receipt:
        for log in receipt.get("logs", []):
            topics = log.get("topics", [])
            if len(topics) >= 3 and topics[0] == TRANSFER_TOPIC:
                from_addr      = "0x" + topics[1][-40:]
                to_addr        = "0x" + topics[2][-40:]
                token_contract = log.get("address", "")
                label          = _token_label(token_contract)
                entry: Dict[str, Any] = {
                    "token":    label,
                    "contract": token_contract,
                    "from":     from_addr,
                    "to":       to_addr,
                }
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
                elif len(topics) == 4:
                    entry["token_id"] = hex_to_int(topics[3])
                    entry["type"]     = "ERC-721"
                transfers.append(entry)

    if transfers:
        out["token_transfers"] = transfers

    if eth_price is not None:
        if value_wei:
            out["value_USD"] = round(wei_to_eth(value_wei) * eth_price, 2)
        if fee_wei is not None:
            out["fee_USD"] = round(wei_to_eth(fee_wei) * eth_price, 4)

    print_json(out)


# ---------------------------------------------------------------------------
# 4. Token Info
# ---------------------------------------------------------------------------

def cmd_token(args):
    """ERC-20 token metadata, supply, price, market cap."""
    addr = args.address.lower()

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

    name         = _decode_string(by_id.get(0))
    symbol       = _decode_string(by_id.get(1))
    decimals_raw = by_id.get(2)
    decimals     = _decode_uint(decimals_raw)
    total_supply_raw = _decode_uint(by_id.get(3))

    if not symbol and addr in KNOWN_TOKENS:
        symbol   = KNOWN_TOKENS[addr][0]
        name     = KNOWN_TOKENS[addr][1]
        decimals = KNOWN_TOKENS[addr][2]

    is_erc20 = bool((symbol or addr in KNOWN_TOKENS) and decimals_raw and decimals_raw != "0x")
    if not is_erc20:
        sys.exit("Contract does not appear to be an ERC-20 token.")

    total_supply = total_supply_raw / (10 ** decimals) if decimals else total_supply_raw
    price_data   = fetch_prices([addr])

    out: Dict[str, Any] = {"address": args.address}
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
# 5. Gas Analysis (EIP-1559 aware)
# ---------------------------------------------------------------------------

def cmd_gas(_args):
    """Gas analysis with EIP-1559 base fee, priority fee, and cost estimates."""
    latest_hex = _rpc_call("eth_blockNumber")
    latest = hex_to_int(latest_hex)

    block_calls = []
    for i in range(10):
        block_calls.append({
            "method": "eth_getBlockByNumber",
            "params": [hex(latest - i), False],
        })
    block_calls.append({"method": "eth_gasPrice"})
    # eth_maxPriorityFeePerGas (EIP-1559)
    block_calls.append({"method": "eth_maxPriorityFeePerGas"})

    results = rpc_batch(block_calls)
    by_id = {r["id"]: r.get("result") for r in results}

    current_gas_price      = hex_to_int(by_id.get(10))
    max_priority_fee       = hex_to_int(by_id.get(11)) if by_id.get(11) else None

    base_fees, gas_utilizations, tx_counts = [], [], []
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
                "block":           hex_to_int(b.get("number")),
                "base_fee_gwei":   round(wei_to_gwei(bf), 6),
                "gas_used":        gu,
                "gas_limit":       gl,
                "utilization_pct": round(gu / gl * 100, 2) if gl > 0 else 0,
                "tx_count":        txc,
            }

    avg_base_fee    = sum(base_fees) / len(base_fees) if base_fees else 0
    avg_utilization = sum(gas_utilizations) / len(gas_utilizations) if gas_utilizations else 0
    avg_tx_count    = sum(tx_counts) / len(tx_counts) if tx_counts else 0

    eth_price = fetch_eth_price()

    # Standard gas limits for common Ethereum operations
    simple_transfer_gas = 21_000
    erc20_transfer_gas  = 65_000
    uniswap_swap_gas    = 150_000
    nft_mint_gas        = 200_000

    def _estimate_cost(gas: int) -> Dict[str, Any]:
        cost_wei = gas * current_gas_price
        cost_eth = wei_to_eth(cost_wei)
        entry: Dict[str, Any] = {"gas_units": gas, "cost_ETH": round(cost_eth, 10)}
        if eth_price:
            entry["cost_USD"] = round(cost_eth * eth_price, 4)
        return entry

    out: Dict[str, Any] = {
        "current_gas_price_gwei": round(wei_to_gwei(current_gas_price), 6),
        "latest_block": latest_block_info,
        "trend_10_blocks": {
            "avg_base_fee_gwei":   round(wei_to_gwei(avg_base_fee), 6),
            "avg_utilization_pct": round(avg_utilization, 2),
            "avg_tx_count":        round(avg_tx_count, 1),
            "min_base_fee_gwei":   round(wei_to_gwei(min(base_fees)), 6) if base_fees else None,
            "max_base_fee_gwei":   round(wei_to_gwei(max(base_fees)), 6) if base_fees else None,
        },
        "cost_estimates": {
            "eth_transfer":   _estimate_cost(simple_transfer_gas),
            "erc20_transfer": _estimate_cost(erc20_transfer_gas),
            "uniswap_swap":   _estimate_cost(uniswap_swap_gas),
            "nft_mint":       _estimate_cost(nft_mint_gas),
        },
        "note": "Ethereum L1 — all fees are paid directly on mainnet. "
                "Costs shown are estimates at current gas price. "
                "Set maxFeePerGas and maxPriorityFeePerGas for EIP-1559 transactions.",
    }
    if max_priority_fee is not None:
        out["suggested_priority_fee_gwei"] = round(wei_to_gwei(max_priority_fee), 4)
    if eth_price:
        out["eth_price_usd"] = eth_price
    print_json(out)


# ---------------------------------------------------------------------------
# 6. Contract Inspection
# ---------------------------------------------------------------------------

def cmd_contract(args):
    """Inspect an address: EOA vs contract, ERC type, proxy resolution."""
    addr = args.address.lower()

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

    by_id: Dict[int, Any] = {}
    for r in results:
        if "error" not in r:
            by_id[r["id"]] = r.get("result")
        else:
            by_id[r["id"]] = None

    code        = by_id.get(0, "0x")
    eth_balance = hex_to_int(by_id.get(1))

    # Try ENS reverse resolution for the address
    ens_name = reverse_resolve_ens(args.address)

    if not code or code == "0x":
        out: Dict[str, Any] = {
            "address":     args.address,
            "is_contract": False,
            "eth_balance": round(wei_to_eth(eth_balance), 18),
            "note":        "This is an externally owned account (EOA), not a contract.",
        }
        if ens_name:
            out["ens_name"] = ens_name
        print_json(out)
        return

    code_size    = (len(code) - 2) // 2
    name         = _decode_string(by_id.get(2))
    symbol       = _decode_string(by_id.get(3))
    decimals_raw = by_id.get(4)
    supply_raw   = by_id.get(5)
    is_erc20     = bool(symbol and decimals_raw and decimals_raw != "0x")

    erc721_result  = by_id.get(6)
    erc1155_result = by_id.get(7)
    is_erc721  = erc721_result is not None and _decode_uint(erc721_result) == 1
    is_erc1155 = erc1155_result is not None and _decode_uint(erc1155_result) == 1

    # EIP-1967 proxy detection
    impl_slot    = "0x360894a13ba1a3210667c828492db98dca3e2076cc3735a920a3ca505d382bbc"
    impl_result  = _rpc_call("eth_getStorageAt", [addr, impl_slot, "latest"])
    is_proxy     = False
    impl_address = None
    if impl_result and impl_result != "0x" + "0" * 64:
        impl_address = "0x" + impl_result[-40:]
        if impl_address != "0x" + "0" * 40:
            is_proxy = True

    out = {
        "address":         args.address,
        "is_contract":     True,
        "code_size_bytes": code_size,
        "eth_balance":     round(wei_to_eth(eth_balance), 18),
    }
    if ens_name:
        out["ens_name"] = ens_name

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
                "hash":      tx.get("hash"),
                "from":      tx.get("from"),
                "to":        tx.get("to"),
                "value_ETH": round(wei_to_eth(value), 6),
            }
            if eth_price:
                entry["value_USD"] = round(wei_to_eth(value) * eth_price, 2)
            whales.append(entry)

    whales.sort(key=lambda x: x["value_ETH"], reverse=True)

    out: Dict[str, Any] = {
        "block":             hex_to_int(block.get("number")),
        "block_time":        hex_to_int(block.get("timestamp")),
        "min_threshold_ETH": args.min_eth,
        "large_transfers":   whales,
        "note":              "Scans latest block only — point-in-time snapshot.",
    }
    if eth_price:
        out["eth_price_usd"] = eth_price
    print_json(out)


# ---------------------------------------------------------------------------
# 8. ENS Lookup
# ---------------------------------------------------------------------------

def cmd_ens(args):
    """Resolve an ENS name to address, or reverse-resolve an address to ENS name."""
    query = args.name_or_address

    if query.startswith("0x") and len(query) == 42:
        # Reverse resolution
        ens_name = reverse_resolve_ens(query)
        if ens_name:
            out: Dict[str, Any] = {"address": query, "ens_name": ens_name}
        else:
            out = {"address": query, "ens_name": None, "note": "No ENS name registered for this address."}
    else:
        # Forward resolution
        address = resolve_ens_name(query)
        if address:
            out = {"ens_name": query, "address": address}
        else:
            out = {"ens_name": query, "address": None, "note": "ENS name not found or not registered."}

    print_json(out)


# ---------------------------------------------------------------------------
# 9. Price Lookup
# ---------------------------------------------------------------------------

def cmd_price(args):
    """Quick price lookup for a token by contract address or known symbol."""
    query = args.token
    addr  = _SYMBOL_TO_ADDRESS.get(query.upper(), query).lower()

    if addr == "eth":
        eth_price = fetch_eth_price()
        out: Dict[str, Any] = {"query": query, "token": "ETH", "name": "Ethereum"}
        out["price_usd"] = eth_price if eth_price else None
        if not eth_price:
            out["note"] = "Price not available."
        print_json(out)
        return

    token_meta = resolve_token_name(addr)
    prices     = fetch_prices([addr])

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
        prog="ethereum_client.py",
        description="Ethereum mainnet query tool for Hermes Agent",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("stats", help="Network stats: block, gas, EIP-1559 base fee, ETH price")

    p_wallet = sub.add_parser("wallet", help="ETH + ERC-20 balances with USD values (ENS supported)")
    p_wallet.add_argument("address", help="Ethereum address (0x...) or ENS name (e.g. vitalik.eth)")
    p_wallet.add_argument("--limit", type=int, default=20, help="Max tokens to display (default: 20)")
    p_wallet.add_argument("--all", action="store_true", help="Show all tokens (no limit, no dust filter)")
    p_wallet.add_argument("--no-prices", action="store_true", help="Skip price lookups (faster, RPC-only)")

    p_tx = sub.add_parser("tx", help="Transaction details by hash")
    p_tx.add_argument("hash")

    p_token = sub.add_parser("token", help="ERC-20 token metadata, price, and market cap")
    p_token.add_argument("address")

    sub.add_parser("gas", help="Gas analysis with EIP-1559 base fee and cost estimates")

    p_contract = sub.add_parser("contract", help="Contract inspection: type detection, proxy check, ENS")
    p_contract.add_argument("address")

    p_whales = sub.add_parser("whales", help="Large ETH transfers in the latest block")
    p_whales.add_argument("--min-eth", type=float, default=10.0,
                          help="Minimum ETH transfer size (default: 10.0)")

    p_ens = sub.add_parser("ens", help="Resolve ENS name <-> Ethereum address")
    p_ens.add_argument("name_or_address", help="ENS name (vitalik.eth) or address (0x...)")

    p_price = sub.add_parser("price", help="Quick price lookup by address or symbol")
    p_price.add_argument("token", help="Contract address or known symbol (ETH, USDC, WETH, ...)")

    args = parser.parse_args()

    dispatch = {
        "stats":    cmd_stats,
        "wallet":   cmd_wallet,
        "tx":       cmd_tx,
        "token":    cmd_token,
        "gas":      cmd_gas,
        "contract": cmd_contract,
        "whales":   cmd_whales,
        "ens":      cmd_ens,
        "price":    cmd_price,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
