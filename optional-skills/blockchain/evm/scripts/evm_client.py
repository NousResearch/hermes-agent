#!/usr/bin/env python3
"""
EVM Blockchain CLI Tool for Hermes Agent
-----------------------------------------
Queries EVM-compatible JSON-RPC APIs and CoinGecko for enriched on-chain data.
Uses only Python standard library — no external packages required.

Supported chains: Ethereum, BNB Chain (BSC), Base, Arbitrum One, Polygon.

Usage:
  python3 evm_client.py stats                                    [--chain CHAIN]
  python3 evm_client.py wallet   <address> [--limit N] [--no-prices] [--chain CHAIN]
  python3 evm_client.py tx       <hash>                          [--chain CHAIN]
  python3 evm_client.py token    <contract_address>              [--chain CHAIN]
  python3 evm_client.py activity <address> [--limit N]           [--chain CHAIN]
  python3 evm_client.py gas                                      [--chain CHAIN]
  python3 evm_client.py price    <symbol_or_address>             [--chain CHAIN]

Environment:
  EVM_RPC_URL  Override the default RPC endpoint for the selected chain
"""

import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Chain configurations
# ---------------------------------------------------------------------------
CHAIN_CONFIGS: Dict[str, Dict[str, Any]] = {
    "ethereum": {
        "name": "Ethereum",
        "rpc_url": "https://ethereum-rpc.publicnode.com",
        "chain_id": 1,
        "native_symbol": "ETH",
        "native_name": "Ether",
        "native_decimals": 18,
        "coingecko_id": "ethereum",
        "coingecko_platform": "ethereum",
        "explorer": "https://etherscan.io",
    },
    "bsc": {
        "name": "BNB Chain",
        "rpc_url": "https://bsc-dataseed1.binance.org",
        "chain_id": 56,
        "native_symbol": "BNB",
        "native_name": "BNB",
        "native_decimals": 18,
        "coingecko_id": "binancecoin",
        "coingecko_platform": "binance-smart-chain",
        "explorer": "https://bscscan.com",
    },
    "base": {
        "name": "Base",
        "rpc_url": "https://mainnet.base.org",
        "chain_id": 8453,
        "native_symbol": "ETH",
        "native_name": "Ether",
        "native_decimals": 18,
        "coingecko_id": "ethereum",
        "coingecko_platform": "base",
        "explorer": "https://basescan.org",
    },
    "arbitrum": {
        "name": "Arbitrum One",
        "rpc_url": "https://arb1.arbitrum.io/rpc",
        "chain_id": 42161,
        "native_symbol": "ETH",
        "native_name": "Ether",
        "native_decimals": 18,
        "coingecko_id": "ethereum",
        "coingecko_platform": "arbitrum-one",
        "explorer": "https://arbiscan.io",
    },
    "polygon": {
        "name": "Polygon",
        "rpc_url": "https://polygon-rpc.com",
        "chain_id": 137,
        "native_symbol": "POL",
        "native_name": "POL",
        "native_decimals": 18,
        "coingecko_id": "matic-network",
        "coingecko_platform": "polygon-pos",
        "explorer": "https://polygonscan.com",
    },
}

# Active chain config (set in main())
_chain: Dict[str, Any] = CHAIN_CONFIGS["ethereum"]
_rpc_url: str = CHAIN_CONFIGS["ethereum"]["rpc_url"]

# ---------------------------------------------------------------------------
# Well-known ERC-20 tokens per chain: contract_address → (symbol, name)
# ---------------------------------------------------------------------------
KNOWN_TOKENS_ETH: Dict[str, Tuple[str, str]] = {
    "0xdAC17F958D2ee523a2206206994597C13D831ec7": ("USDT",  "Tether"),
    "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48": ("USDC",  "USD Coin"),
    "0x6B175474E89094C44Da98b954EedeAC495271d0F": ("DAI",   "Dai"),
    "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2": ("WETH",  "Wrapped Ether"),
    "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599": ("WBTC",  "Wrapped Bitcoin"),
    "0x514910771AF9Ca656af840dff83E8264EcF986CA": ("LINK",  "Chainlink"),
    "0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984": ("UNI",   "Uniswap"),
    "0x7Fc66500c84A76Ad7e9c93437bFc5Ac33E2DDaE9": ("AAVE",  "Aave"),
    "0x6982508145454Ce325dDbE47a25d4ec3d2311933": ("PEPE",  "Pepe"),
    "0x95aD61b0a150d79219dCF64E1E6Cc01f0B64C4cE": ("SHIB",  "Shiba Inu"),
    "0x9f8F72aA9304c8B593d555F12eF6589cC3A579A2": ("MKR",   "Maker"),
    "0x5A98FcBEA516Cf06857215779Fd812CA3beF1B32": ("LDO",   "Lido DAO"),
    "0xD533a949740bb3306d119CC777fa900bA034cd52": ("CRV",   "Curve DAO"),
    "0xc00e94Cb662C3520282E6f5717214004A7f26888": ("COMP",  "Compound"),
    "0xC011a73ee8576Fb46F5E1c5751cA3B9Fe0af2a6F": ("SNX",   "Synthetix"),
    "0x111111111117dC0aa78b770fA6A738034120C302": ("1INCH", "1inch"),
    "0x7D1AfA7B718fb893dB30A3aBc0Cfc608AaCfeBB0": ("MATIC", "Polygon"),
    "0xB8c77482e45F1F44dE1745F52C74426C631bDD52": ("BNB",   "BNB"),
    "0x4d224452801ACEd8B2F0aebE155379bb5D594381": ("APE",   "ApeCoin"),
    "0x3845badAde8e6dFF049820680d1F14bD3903a5d0": ("SAND",  "The Sandbox"),
}

KNOWN_TOKENS_BSC: Dict[str, Tuple[str, str]] = {
    "0x55d398326f99059fF775485246999027B3197955": ("USDT",  "Tether"),
    "0x8AC76a51cc950d9822D68b83fE1Ad97B32Cd580d": ("USDC",  "USD Coin"),
    "0xe9e7CEA3DedcA5984780Bafc599bD69ADd087D56": ("BUSD",  "Binance USD"),
    "0xbb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c": ("WBNB",  "Wrapped BNB"),
    "0x2170Ed0880ac9A755fd29B2688956BD959F933F8": ("ETH",   "Ethereum"),
    "0x0E09FaBB73Bd3Ade0a17ECC321fD13a19e81cE82": ("CAKE",  "PancakeSwap"),
    "0x7083609fCE4d1d8Dc0C979AAb8c869Ea2C873402": ("DOT",   "Polkadot"),
    "0x1D2F0da169ceB9fC7B3144628dB156f3F6c60dBE": ("XRP",   "XRP"),
    "0xbA2aE424d960c26247Dd6c32edC70B295c744C43": ("DOGE",  "Dogecoin"),
    "0x3EE2200Efb3400fAbB9AacF31297cBdD1d435D47": ("ADA",   "Cardano"),
}

# Map chain key → known tokens dict
_CHAIN_TOKENS: Dict[str, Dict[str, Tuple[str, str]]] = {
    "ethereum": KNOWN_TOKENS_ETH,
    "bsc": KNOWN_TOKENS_BSC,
    "base": KNOWN_TOKENS_ETH,       # Base uses many of the same tokens
    "arbitrum": KNOWN_TOKENS_ETH,    # Arbitrum uses many of the same tokens
    "polygon": KNOWN_TOKENS_ETH,     # Polygon uses many of the same tokens
}

# Symbol → CoinGecko ID for price lookups
SYMBOL_TO_COINGECKO: Dict[str, str] = {
    "ETH":   "ethereum",
    "BNB":   "binancecoin",
    "USDC":  "usd-coin",
    "USDT":  "tether",
    "DAI":   "dai",
    "WBTC":  "wrapped-bitcoin",
    "BTC":   "bitcoin",
    "LINK":  "chainlink",
    "UNI":   "uniswap",
    "AAVE":  "aave",
    "PEPE":  "pepe",
    "SHIB":  "shiba-inu",
    "ARB":   "arbitrum",
    "OP":    "optimism",
    "MATIC": "matic-network",
    "POL":   "matic-network",
    "DOGE":  "dogecoin",
    "MKR":   "maker",
    "LDO":   "lido-dao",
    "CRV":   "curve-dao-token",
    "COMP":  "compound-governance-token",
    "SNX":   "havven",
    "CAKE":  "pancakeswap-token",
    "SOL":   "solana",
    "DOT":   "polkadot",
    "ADA":   "cardano",
    "XRP":   "ripple",
    "AVAX":  "avalanche-2",
    "APE":   "apecoin",
    "SAND":  "the-sandbox",
    "1INCH": "1inch",
}


# ---------------------------------------------------------------------------
# HTTP / RPC helpers
# ---------------------------------------------------------------------------
def _http_get_json(url: str, timeout: int = 10, retries: int = 2) -> Any:
    """GET JSON from a URL with retry on 429 rate-limit."""
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
    """Send a JSON-RPC request to the active EVM chain with retry on 429."""
    payload = json.dumps({
        "jsonrpc": "2.0", "id": 1,
        "method": method, "params": params or [],
    }).encode()

    for attempt in range(retries + 1):
        req = urllib.request.Request(
            _rpc_url, data=payload,
            headers={"Content-Type": "application/json", "User-Agent": "HermesAgent/1.0"},
            method="POST",
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


def _rpc_batch(calls: list) -> list:
    """Send a batch of JSON-RPC requests (with retry on 429)."""
    payload = json.dumps([
        {"jsonrpc": "2.0", "id": i, "method": c["method"], "params": c.get("params", [])}
        for i, c in enumerate(calls)
    ]).encode()

    for attempt in range(3):
        req = urllib.request.Request(
            _rpc_url, data=payload,
            headers={"Content-Type": "application/json", "User-Agent": "HermesAgent/1.0"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                result = json.load(resp)
                if isinstance(result, list):
                    return result
                # Some RPCs don't support batch — fall back to individual calls
                return [_rpc_call(c["method"], c.get("params", [])) for c in calls]
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt < 2:
                time.sleep(1.5 * (attempt + 1))
                continue
            # Fallback to individual calls
            return [_rpc_call(c["method"], c.get("params", [])) for c in calls]
        except Exception:
            return [_rpc_call(c["method"], c.get("params", [])) for c in calls]
    return []


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
def hex_to_int(hex_str: Optional[str]) -> int:
    """Convert hex string (0x...) to integer, or 0 if None."""
    if hex_str is None:
        return 0
    return int(hex_str, 16)


def wei_to_native(wei: int, decimals: int = 18) -> float:
    """Convert wei to native token units."""
    return wei / (10 ** decimals)


def gwei_from_wei(wei: int) -> float:
    """Convert wei to gwei."""
    return wei / 1e9


def print_json(obj: Any) -> None:
    print(json.dumps(obj, indent=2))


def _short_addr(addr: str) -> str:
    """Abbreviate an address: first 6 + last 4."""
    if len(addr) <= 14:
        return addr
    return f"{addr[:6]}...{addr[-4:]}"


def _get_known_tokens() -> Dict[str, Tuple[str, str]]:
    """Get known tokens for the active chain."""
    chain_key = _chain.get("_key", "ethereum")
    return _CHAIN_TOKENS.get(chain_key, KNOWN_TOKENS_ETH)


def _token_label(addr: str) -> str:
    """Return a human-readable label: symbol if known, else abbreviated address."""
    known = _get_known_tokens()
    # Case-insensitive lookup
    for k, v in known.items():
        if k.lower() == addr.lower():
            return v[0]
    return _short_addr(addr)


# ---------------------------------------------------------------------------
# CoinGecko pricing
# ---------------------------------------------------------------------------
def fetch_native_price() -> Optional[float]:
    """Fetch current native token price in USD via CoinGecko."""
    cg_id = _chain["coingecko_id"]
    data = _http_get_json(
        f"https://api.coingecko.com/api/v3/simple/price?ids={cg_id}&vs_currencies=usd"
    )
    if data and cg_id in data:
        return data[cg_id].get("usd")
    return None


def fetch_token_prices(contracts: List[str], max_lookups: int = 20) -> Dict[str, float]:
    """Fetch USD prices for contract addresses via CoinGecko."""
    platform = _chain["coingecko_platform"]
    prices: Dict[str, float] = {}

    for i, addr in enumerate(contracts[:max_lookups]):
        url = (
            f"https://api.coingecko.com/api/v3/simple/token_price/{platform}"
            f"?contract_addresses={addr}&vs_currencies=usd"
        )
        data = _http_get_json(url, timeout=10)
        if data and isinstance(data, dict):
            for contract, info in data.items():
                if isinstance(info, dict) and "usd" in info:
                    prices[addr.lower()] = info["usd"]
                    break
        if i < len(contracts[:max_lookups]) - 1:
            time.sleep(1.0)
    return prices


def fetch_price_by_id(cg_id: str) -> Optional[float]:
    """Fetch price by CoinGecko ID."""
    data = _http_get_json(
        f"https://api.coingecko.com/api/v3/simple/price?ids={cg_id}&vs_currencies=usd"
    )
    if data and cg_id in data:
        return data[cg_id].get("usd")
    return None


# ---------------------------------------------------------------------------
# ERC-20 helpers
# ---------------------------------------------------------------------------
def _erc20_call(contract: str, data: str) -> Optional[str]:
    """Call an ERC-20 view function via eth_call."""
    result = rpc("eth_call", [{"to": contract, "data": data}, "latest"])
    if result and result != "0x":
        return result
    return None


def _erc20_name(contract: str) -> Optional[str]:
    """Get ERC-20 token name."""
    # name() = 0x06fdde03
    result = _erc20_call(contract, "0x06fdde03")
    if result:
        try:
            # Decode ABI-encoded string
            data = bytes.fromhex(result[2:])
            if len(data) >= 64:
                offset = int.from_bytes(data[0:32], "big")
                length = int.from_bytes(data[offset:offset+32], "big")
                return data[offset+32:offset+32+length].decode("utf-8", errors="replace").strip("\x00")
        except Exception:
            pass
    return None


def _erc20_symbol(contract: str) -> Optional[str]:
    """Get ERC-20 token symbol."""
    # symbol() = 0x95d89b41
    result = _erc20_call(contract, "0x95d89b41")
    if result:
        try:
            data = bytes.fromhex(result[2:])
            if len(data) >= 64:
                offset = int.from_bytes(data[0:32], "big")
                length = int.from_bytes(data[offset:offset+32], "big")
                return data[offset+32:offset+32+length].decode("utf-8", errors="replace").strip("\x00")
        except Exception:
            pass
    return None


def _erc20_decimals(contract: str) -> int:
    """Get ERC-20 token decimals."""
    # decimals() = 0x313ce567
    result = _erc20_call(contract, "0x313ce567")
    if result:
        try:
            return int(result, 16)
        except Exception:
            pass
    return 18


def _erc20_total_supply(contract: str) -> Optional[int]:
    """Get ERC-20 total supply in raw units."""
    # totalSupply() = 0x18160ddd
    result = _erc20_call(contract, "0x18160ddd")
    if result:
        try:
            return int(result, 16)
        except Exception:
            pass
    return None


def _erc20_balance_of(contract: str, address: str) -> int:
    """Get ERC-20 token balance for an address."""
    # balanceOf(address) = 0x70a08231 + padded address
    addr_padded = address.lower().replace("0x", "").zfill(64)
    result = _erc20_call(contract, f"0x70a08231{addr_padded}")
    if result:
        try:
            return int(result, 16)
        except Exception:
            pass
    return 0


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def cmd_stats(_args):
    """Live network stats: block number, gas price, native token price."""
    block_hex = rpc("eth_blockNumber")
    gas_hex = rpc("eth_gasPrice")

    block_number = hex_to_int(block_hex)
    gas_wei = hex_to_int(gas_hex)
    gas_gwei = round(gwei_from_wei(gas_wei), 2)

    native_price = fetch_native_price()

    out = {
        "chain": _chain["name"],
        "chain_id": _chain["chain_id"],
        "native_token": _chain["native_symbol"],
        "latest_block": block_number,
        "gas_price_gwei": gas_gwei,
        "explorer": _chain["explorer"],
    }
    if native_price is not None:
        out[f"{_chain['native_symbol'].lower()}_price_usd"] = native_price

        # Estimate costs
        transfer_cost = gas_gwei * 21000 / 1e9
        out["estimated_transfer_cost"] = {
            f"{_chain['native_symbol']}": round(transfer_cost, 6),
            "USD": round(transfer_cost * native_price, 4),
        }

    print_json(out)


def cmd_wallet(args):
    """Native balance + ERC-20 token holdings with USD values."""
    address = args.address
    limit = getattr(args, "limit", 20) or 20
    skip_prices = getattr(args, "no_prices", False)

    # Fetch native balance
    balance_hex = rpc("eth_getBalance", [address, "latest"])
    native_balance = wei_to_native(hex_to_int(balance_hex))

    # Fetch native price
    native_price = None
    if not skip_prices:
        native_price = fetch_native_price()

    # Check known token balances
    known = _get_known_tokens()
    tokens = []

    for contract, (symbol, name) in known.items():
        raw_balance = _erc20_balance_of(contract, address)
        if raw_balance > 0:
            decimals = _erc20_decimals(contract)
            human_balance = raw_balance / (10 ** decimals)
            tokens.append({
                "contract": contract,
                "symbol": symbol,
                "name": name,
                "balance": human_balance,
                "decimals": decimals,
            })

    # Fetch prices for tokens
    prices: Dict[str, float] = {}
    if not skip_prices and tokens:
        contracts_to_price = [t["contract"] for t in tokens]
        prices = fetch_token_prices(contracts_to_price, max_lookups=20)

    # Enrich with USD values
    enriched = []
    for t in tokens:
        entry = {
            "token": t["symbol"],
            "name": t["name"],
            "contract": t["contract"],
            "balance": t["balance"],
        }
        usd_price = prices.get(t["contract"].lower())
        if usd_price is not None:
            entry["price_usd"] = usd_price
            entry["value_usd"] = round(usd_price * t["balance"], 2)
        enriched.append(entry)

    # Sort by value descending
    enriched.sort(
        key=lambda x: (x.get("value_usd") is not None, x.get("value_usd") or 0),
        reverse=True,
    )

    # Apply limit
    total_tokens = len(enriched)
    if len(enriched) > limit:
        enriched = enriched[:limit]

    # Portfolio total
    total_usd = sum(t.get("value_usd", 0) for t in enriched)
    native_value_usd = round(native_price * native_balance, 2) if native_price else None
    if native_value_usd:
        total_usd += native_value_usd

    output = {
        "chain": _chain["name"],
        "address": address,
        f"{_chain['native_symbol'].lower()}_balance": round(native_balance, 8),
    }
    if native_price:
        output[f"{_chain['native_symbol'].lower()}_price_usd"] = native_price
        output[f"{_chain['native_symbol'].lower()}_value_usd"] = native_value_usd
    output["tokens_shown"] = len(enriched)
    if total_tokens > len(enriched):
        output["tokens_hidden"] = total_tokens - len(enriched)
    output["erc20_tokens"] = enriched
    if total_usd > 0:
        output["portfolio_total_usd"] = round(total_usd, 2)
    output["note"] = (
        "Only checks balances for well-known tokens. "
        "Use a block explorer for complete token lists."
    )

    print_json(output)


def cmd_tx(args):
    """Full transaction details by hash."""
    tx = rpc("eth_getTransactionByHash", [args.hash])
    if tx is None:
        sys.exit("Transaction not found.")

    receipt = rpc("eth_getTransactionReceipt", [args.hash])

    value_wei = hex_to_int(tx.get("value"))
    value_native = wei_to_native(value_wei)
    gas_price_wei = hex_to_int(tx.get("gasPrice", "0x0"))
    gas_used = hex_to_int(receipt.get("gasUsed", "0x0")) if receipt else None
    gas_cost_native = wei_to_native(gas_price_wei * gas_used) if gas_used else None

    native_price = fetch_native_price()

    # Get block timestamp
    block_hex = tx.get("blockNumber")
    timestamp = None
    if block_hex:
        block = rpc("eth_getBlockByNumber", [block_hex, False])
        if block:
            timestamp = hex_to_int(block.get("timestamp"))

    out = {
        "chain": _chain["name"],
        "hash": args.hash,
        "block_number": hex_to_int(block_hex) if block_hex else None,
        "timestamp": timestamp,
        "from": tx.get("from"),
        "to": tx.get("to"),
        f"value_{_chain['native_symbol']}": round(value_native, 8),
        "gas_price_gwei": round(gwei_from_wei(gas_price_wei), 2),
    }

    if gas_used is not None:
        out["gas_used"] = gas_used
        out[f"gas_cost_{_chain['native_symbol']}"] = round(gas_cost_native, 8)

    if receipt:
        out["status"] = "success" if receipt.get("status") == "0x1" else "failed"
        out["logs_count"] = len(receipt.get("logs", []))
        if receipt.get("contractAddress"):
            out["contract_created"] = receipt["contractAddress"]

    if native_price:
        out[f"value_USD"] = round(value_native * native_price, 2)
        if gas_cost_native is not None:
            out["gas_cost_USD"] = round(gas_cost_native * native_price, 4)

    # Input data preview
    input_data = tx.get("input", "0x")
    if input_data and input_data != "0x":
        out["input_preview"] = input_data[:74] + ("..." if len(input_data) > 74 else "")
        out["is_contract_call"] = True
    else:
        out["is_contract_call"] = False

    print_json(out)


def cmd_token(args):
    """ERC-20 token metadata, supply, price, market cap."""
    contract = args.contract

    name = _erc20_name(contract)
    symbol = _erc20_symbol(contract)
    decimals = _erc20_decimals(contract)
    total_supply_raw = _erc20_total_supply(contract)

    if name is None and symbol is None:
        sys.exit("Contract not found or not an ERC-20 token.")

    total_supply_human = total_supply_raw / (10 ** decimals) if total_supply_raw else None

    # Fetch price
    prices = fetch_token_prices([contract])
    price = prices.get(contract.lower())

    out = {
        "chain": _chain["name"],
        "contract": contract,
    }
    if name:
        out["name"] = name
    if symbol:
        out["symbol"] = symbol
    out["decimals"] = decimals
    if total_supply_human is not None:
        out["total_supply"] = round(total_supply_human, min(decimals, 6))
    if price is not None:
        out["price_usd"] = price
        if total_supply_human:
            out["market_cap_usd"] = round(price * total_supply_human, 0)

    out["explorer_url"] = f"{_chain['explorer']}/token/{contract}"

    print_json(out)


def cmd_activity(args):
    """Recent transactions for an address by scanning recent blocks."""
    address = args.address.lower()
    limit = min(args.limit, 25)

    # Get latest block number
    latest_hex = rpc("eth_blockNumber")
    latest = hex_to_int(latest_hex)

    txs_found = []
    blocks_scanned = 0
    max_blocks = 50  # Scan at most 50 blocks

    for block_num in range(latest, max(latest - max_blocks, 0), -1):
        if len(txs_found) >= limit:
            break

        block = rpc("eth_getBlockByNumber", [hex(block_num), True])
        if not block:
            continue
        blocks_scanned += 1

        timestamp = hex_to_int(block.get("timestamp"))
        for tx in (block.get("transactions") or []):
            tx_from = (tx.get("from") or "").lower()
            tx_to = (tx.get("to") or "").lower()
            if tx_from == address or tx_to == address:
                value = wei_to_native(hex_to_int(tx.get("value")))
                txs_found.append({
                    "hash": tx["hash"],
                    "block": block_num,
                    "timestamp": timestamp,
                    "from": tx.get("from"),
                    "to": tx.get("to"),
                    f"value_{_chain['native_symbol']}": round(value, 8),
                    "direction": "OUT" if tx_from == address else "IN",
                })
                if len(txs_found) >= limit:
                    break

    print_json({
        "chain": _chain["name"],
        "address": args.address,
        "blocks_scanned": blocks_scanned,
        "transactions_found": len(txs_found),
        "transactions": txs_found,
        "note": (
            f"Scanned last {blocks_scanned} blocks only. "
            "For full history, use a block explorer."
        ),
    })


def cmd_gas(_args):
    """Current gas prices in gwei with cost estimates."""
    gas_hex = rpc("eth_gasPrice")
    gas_wei = hex_to_int(gas_hex)
    gas_gwei = gwei_from_wei(gas_wei)

    native_price = fetch_native_price()

    # Gas estimates for common operations
    operations = {
        "ETH/BNB Transfer": 21000,
        "ERC-20 Transfer": 65000,
        "ERC-20 Approve": 46000,
        "Uniswap/DEX Swap": 150000,
        "NFT Mint": 120000,
        "NFT Transfer": 85000,
    }

    estimates = {}
    for op_name, gas_units in operations.items():
        cost_native = gas_gwei * gas_units / 1e9
        entry = {
            "gas_units": gas_units,
            f"cost_{_chain['native_symbol']}": round(cost_native, 6),
        }
        if native_price:
            entry["cost_USD"] = round(cost_native * native_price, 4)
        estimates[op_name] = entry

    out = {
        "chain": _chain["name"],
        "gas_price_gwei": round(gas_gwei, 2),
        "gas_price_wei": gas_wei,
    }
    if native_price:
        out[f"{_chain['native_symbol'].lower()}_price_usd"] = native_price
    out["estimated_costs"] = estimates

    print_json(out)


def cmd_price(args):
    """Quick price lookup for a token by symbol or contract address."""
    query = args.token

    # Check if it's a known symbol
    symbol_upper = query.upper()
    if symbol_upper in SYMBOL_TO_COINGECKO:
        cg_id = SYMBOL_TO_COINGECKO[symbol_upper]
        price = fetch_price_by_id(cg_id)
        out = {"query": query, "symbol": symbol_upper, "coingecko_id": cg_id}
        if price is not None:
            out["price_usd"] = price
        else:
            out["price_usd"] = None
            out["note"] = "Price not available from CoinGecko."
        print_json(out)
        return

    # Assume it's a contract address
    if query.startswith("0x") and len(query) == 42:
        name = _erc20_name(query)
        symbol = _erc20_symbol(query)
        prices = fetch_token_prices([query])
        price = prices.get(query.lower())

        out = {"query": query, "contract": query}
        if name:
            out["name"] = name
        if symbol:
            out["symbol"] = symbol
        if price is not None:
            out["price_usd"] = price
        else:
            out["price_usd"] = None
            out["note"] = "Price not available — token may not be listed on CoinGecko."
        print_json(out)
        return

    # Unknown query
    print_json({
        "query": query,
        "price_usd": None,
        "note": f"Unknown symbol '{query}'. Use a known symbol (ETH, BNB, USDC, ...) or a contract address (0x...).",
    })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # Common parent parser for --chain argument
    chain_parent = argparse.ArgumentParser(add_help=False)
    chain_parent.add_argument(
        "--chain", type=str, default="ethereum",
        choices=list(CHAIN_CONFIGS.keys()),
        help="Blockchain to query (default: ethereum)",
    )

    parser = argparse.ArgumentParser(
        prog="evm_client.py",
        description="EVM blockchain query tool for Hermes Agent",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("stats", parents=[chain_parent],
                   help="Network stats: block number, gas price, native token price")

    p_wallet = sub.add_parser("wallet", parents=[chain_parent],
                              help="Native balance + ERC-20 tokens with USD values")
    p_wallet.add_argument("address")
    p_wallet.add_argument("--limit", type=int, default=20,
                          help="Max tokens to display (default: 20)")
    p_wallet.add_argument("--no-prices", action="store_true",
                          help="Skip price lookups (faster, RPC-only)")

    p_tx = sub.add_parser("tx", parents=[chain_parent],
                          help="Transaction details by hash")
    p_tx.add_argument("hash")

    p_token = sub.add_parser("token", parents=[chain_parent],
                             help="ERC-20 token metadata, price, and market cap")
    p_token.add_argument("contract")

    p_activity = sub.add_parser("activity", parents=[chain_parent],
                                help="Recent transactions for an address")
    p_activity.add_argument("address")
    p_activity.add_argument("--limit", type=int, default=10,
                            help="Number of transactions (max 25, default 10)")

    sub.add_parser("gas", parents=[chain_parent],
                   help="Current gas prices with cost estimates")

    p_price = sub.add_parser("price", parents=[chain_parent],
                             help="Quick price lookup by symbol or contract address")
    p_price.add_argument("token", help="Symbol (ETH, BNB, USDC, ...) or contract address (0x...)")

    args = parser.parse_args()

    # Configure chain
    global _chain, _rpc_url
    _chain = CHAIN_CONFIGS[args.chain]
    _chain["_key"] = args.chain

    # Allow RPC override via environment
    env_rpc = os.environ.get("EVM_RPC_URL")
    _rpc_url = env_rpc if env_rpc else _chain["rpc_url"]

    dispatch = {
        "stats":    cmd_stats,
        "wallet":   cmd_wallet,
        "tx":       cmd_tx,
        "token":    cmd_token,
        "activity": cmd_activity,
        "gas":      cmd_gas,
        "price":    cmd_price,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
