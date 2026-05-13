#!/usr/bin/env python3
"""
evm_client.py — EVM blockchain CLI tool for the Hermes Agent project.
Zero external dependencies. Uses stdlib only: urllib, json, argparse, time, os, sys, typing.
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Chain registry
# ---------------------------------------------------------------------------

CHAINS: Dict[str, Dict[str, Any]] = {
    "ethereum": {
        "chain_id": 1,
        "rpc": "https://ethereum-rpc.publicnode.com",
        "native": "ETH",
        "coingecko": "ethereum",
        "explorer": "https://etherscan.io",
        "decimals": 18,
    },
    "bsc": {
        "chain_id": 56,
        "rpc": "https://bsc-dataseed1.binance.org",
        "native": "BNB",
        "coingecko": "binancecoin",
        "explorer": "https://bscscan.com",
        "decimals": 18,
    },
    "base": {
        "chain_id": 8453,
        "rpc": "https://mainnet.base.org",
        "native": "ETH",
        "coingecko": "ethereum",
        "explorer": "https://basescan.org",
        "decimals": 18,
    },
    "arbitrum": {
        "chain_id": 42161,
        "rpc": "https://arb1.arbitrum.io/rpc",
        "native": "ETH",
        "coingecko": "ethereum",
        "explorer": "https://arbiscan.io",
        "decimals": 18,
    },
    "polygon": {
        "chain_id": 137,
        "rpc": "https://polygon-rpc.com",
        "native": "MATIC",
        "coingecko": "matic-network",
        "explorer": "https://polygonscan.com",
        "decimals": 18,
    },
    "optimism": {
        "chain_id": 10,
        "rpc": "https://mainnet.optimism.io",
        "native": "ETH",
        "coingecko": "ethereum",
        "explorer": "https://optimistic.etherscan.io",
        "decimals": 18,
    },
    "avalanche": {
        "chain_id": 43114,
        "rpc": "https://api.avax.network/ext/bc/C/rpc",
        "native": "AVAX",
        "coingecko": "avalanche-2",
        "explorer": "https://snowtrace.io",
        "decimals": 18,
    },
    "zksync": {
        "chain_id": 324,
        "rpc": "https://mainnet.era.zksync.io",
        "native": "ETH",
        "coingecko": "ethereum",
        "explorer": "https://explorer.zksync.io",
        "decimals": 18,
    },
}

DEFAULT_CHAIN = "ethereum"

# ---------------------------------------------------------------------------
# Known ERC-20 token registry  {chain -> {symbol -> address}}
# ---------------------------------------------------------------------------

KNOWN_TOKENS: Dict[str, Dict[str, str]] = {
    "ethereum": {
        "USDT":  "0xdAC17F958D2ee523a2206206994597C13D831ec7",
        "USDC":  "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
        "DAI":   "0x6B175474E89094C44Da98b954EedeAC495271d0F",
        "WETH":  "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
        "WBTC":  "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599",
        "LINK":  "0x514910771AF9Ca656af840dff83E8264EcF986CA",
        "UNI":   "0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984",
        "AAVE":  "0x7Fc66500c84A76Ad7e9c93437bFc5Ac33E2DDaE9",
        "MKR":   "0x9f8F72aA9304c8B593d555F12eF6589cC3A579A2",
        "COMP":  "0xc00e94Cb662C3520282E6f5717214004A7f26888",
        "SNX":   "0xC011a73ee8576Fb46F5E1c5751cA3B9Fe0af2a6F",
        "CRV":   "0xD533a949740bb3306d119CC777fa900bA034cd52",
        "LDO":   "0x5A98FcBEA516Cf06857215779Fd812CA3beF1B32",
        "RPL":   "0xD33526068D116cE69F19A9ee46F0bd304F21A51f",
        "MATIC": "0x7D1AfA7B718fb893dB30A3aBc0Cfc608AaCfeBB0",
        "SHIB":  "0x95aD61b0a150d79219dCF64E1E6Cc01f0B64C4cE",
        "APE":   "0x4d224452801ACEd8B2F0aebE155379bb5D594381",
        "GRT":   "0xc944E90C64B2c07662A292be6244BDf05Cda44a7",
        "FXS":   "0x3432B6A60D23Ca0dFCa7761B7ab56459D9C964D0",
        "FRAX":  "0x853d955aCEf822Db058eb8505911ED77F175b99e",
        "BAL":   "0xba100000625a3754423978a60c9317c58a424e3D",
        "SUSHI": "0x6B3595068778DD592e39A122f4f5a5cF09C90fE2",
        "YFI":   "0x0bc529c00C6401aEF6D220BE8C6Ea1667F6Ad93e",
        "1INCH": "0x111111111117dC0aa78b770fA6A738034120C302",
        "ENS":   "0xC18360217D8F7Ab5e7c516566761Ea12Ce7F9D72",
        "IMX":   "0xF57e7e7C23978C3cAEC3C3548E3D615c346e79fF",
        "SAND":  "0x3845badAde8e6dFF049820680d1F14bD3903a5d0",
        "MANA":  "0x0F5D2fB29fb7d3CFeE444a200298f468908cC942",
        "AXS":   "0xBB0E17EF65F82Ab018d8EDd776e8DD940327B28b",
        "CHZ":   "0x3506424F91fD33084466F402d5D97f05F8e3b4AF",
        "PEPE":  "0x6982508145454Ce325dDbE47a25d4ec3d2311933",
    },
    "bsc": {
        "USDT":  "0x55d398326f99059fF775485246999027B3197955",
        "USDC":  "0x8AC76a51cc950d9822D68b83fE1Ad97B32Cd580d",
        "BUSD":  "0xe9e7CEA3DedcA5984780Bafc599bD69ADd087D56",
        "WBNB":  "0xbb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c",
        "CAKE":  "0x0E09FaBB73Bd3Ade0a17ECC321fD13a19e81cE82",
        "XVS":   "0xcF6BB5389c92Bdda8a3747Ddb454cB7a64626C63",
        "ALPACA":"0x8F0528cE5eF7B51152A59745bEfDD91D97091d2F",
        "BAKE":  "0xE02dF9e3e622DeBdD69fb838bB799E3F168902c5",
        "BURGER":"0xAe9269f27437f0fcBC232d39Ec814844a51d6b8f",
        "DOGE":  "0xbA2aE424d960c26247Dd6c32edC70B295c744C43",
    },
    "base": {
        # Stables + wrapped
        "USDC":   "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
        "DAI":    "0x50c5725949A6F0c72E6C4a641F24049A917DB0Cb",
        "WETH":   "0x4200000000000000000000000000000000000006",
        # Liquid-staked ETH variants
        "cbETH":  "0x2Ae3F1Ec7F1F5012CFEab0185bfc7aa3cF0DEc22",
        "wstETH": "0xc1CBa3fCea344f92D9239c08C0568f6F2F0ee452",
        "rETH":   "0xB6fe221Fe9EeF5aBa221c348bA20A1Bf5e73624c",
        "cbBTC":  "0xcbB7C0000aB88B473b1f5aFd9ef808440eed33Bf",
        # Base-native DeFi + meme tokens (carried over from the standalone base/ skill)
        "AERO":   "0x940181a94A35A4569E4529A3CDfB74e38FD98631",
        "DEGEN":  "0x4ed4E862860beD51a9570b96d89aF5E1B0Efefed",
        "TOSHI":  "0xAC1Bd2486aAf3B5C0fc3Fd868558b082a531B2B4",
        "BRETT":  "0x532f27101965dd16442E59d40670FaF5eBB142E4",
        "WELL":   "0xA88594D404727625A9437C3f886C7643872296AE",
    },
    "arbitrum": {
        "USDC":  "0xaf88d065e77c8cC2239327C5EDb3A432268e5831",
        "USDT":  "0xFd086bC7CD5C481DCC9C85ebE478A1C0b69FCbb9",
        "WETH":  "0x82aF49447D8a07e3bd95BD0d56f35241523fBab1",
        "ARB":   "0x912CE59144191C1204E64559FE8253a0e49E6548",
    },
    "optimism": {
        "USDC":  "0x0b2C639c533813f4Aa9D7837CAf62653d097Ff85",
        "USDT":  "0x94b008aA00579c1307B0EF2c499aD98a8ce58e58",
        "WETH":  "0x4200000000000000000000000000000000000006",
        "OP":    "0x4200000000000000000000000000000000000042",
    },
    "polygon": {
        "USDC":  "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174",
        "USDT":  "0xc2132D05D31c914a87C6611C10748AEb04B58e8F",
        "WMATIC":"0x0d500B1d8E8eF31E21C99d1Db9A6444d3ADf1270",
        "WETH":  "0x7ceB23fD6bC0adD59E62ac25578270cFf1b9f619",
        "DAI":   "0x8f3Cf7ad23Cd3CaDbD9735AFf958023239c6A063",
    },
    "avalanche": {
        "USDC":  "0xB97EF9Ef8734C71904D8002F8b6Bc66Dd9c48a6E",
        "USDT":  "0x9702230A8Ea53601f5cD2dc00fDBc13d4dF4A8c7",
        "WAVAX": "0xB31f66AA3C1e785363F0875A1B74E27b85FD66c7",
    },
}

# Gas estimates (units) for common operations
GAS_ESTIMATES = {
    "transfer":     21_000,
    "erc20":        65_000,
    "approve":      46_000,
    "swap":        180_000,
    "nft_mint":    150_000,
    "nft_transfer": 85_000,
}

# CoinGecko symbol -> id map for common tokens
COINGECKO_IDS: Dict[str, str] = {
    "ETH":   "ethereum",
    "BTC":   "bitcoin",
    "BNB":   "binancecoin",
    "MATIC": "matic-network",
    "AVAX":  "avalanche-2",
    "USDT":  "tether",
    "USDC":  "usd-coin",
    "DAI":   "dai",
    "WBTC":  "wrapped-bitcoin",
    "WETH":  "weth",
    "LINK":  "chainlink",
    "UNI":   "uniswap",
    "AAVE":  "aave",
    "MKR":   "maker",
    "COMP":  "compound-governance-token",
    "SNX":   "havven",
    "CRV":   "curve-dao-token",
    "LDO":   "lido-dao",
    "RPL":   "rocket-pool",
    "SHIB":  "shiba-inu",
    "APE":   "apecoin",
    "GRT":   "the-graph",
    "BAL":   "balancer",
    "SUSHI": "sushi",
    "YFI":   "yearn-finance",
    "1INCH": "1inch",
    "ENS":   "ethereum-name-service",
    "IMX":   "immutable-x",
    "SAND":  "the-sandbox",
    "MANA":  "decentraland",
    "AXS":   "axie-infinity",
    "ARB":   "arbitrum",
    "OP":    "optimism",
    "CAKE":  "pancakeswap-token",
    "PEPE":  "pepe",
    "CHZ":   "chiliz",
}

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def hex_to_int(h: str) -> int:
    if not h or h == "0x":
        return 0
    return int(h, 16)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def is_valid_address(s: str) -> bool:
    """Return True if `s` looks like a 20-byte hex Ethereum address.

    Does NOT validate EIP-55 checksum — RPC endpoints accept any-case hex.
    Just guards against typos / wrong-length input before we burn an RPC call.
    """
    if not isinstance(s, str):
        return False
    if not s.startswith("0x") and not s.startswith("0X"):
        return False
    if len(s) != 42:
        return False
    try:
        int(s, 16)
    except ValueError:
        return False
    return True


def is_valid_txhash(s: str) -> bool:
    """Return True if `s` looks like a 32-byte hex transaction hash."""
    if not isinstance(s, str):
        return False
    if not s.startswith("0x") and not s.startswith("0X"):
        return False
    if len(s) != 66:
        return False
    try:
        int(s, 16)
    except ValueError:
        return False
    return True


def require_address(s: str, *, field: str = "address") -> str:
    """Return `s` lowercased if valid, else exit with an error message.

    Centralizing validation here means every subcommand fails fast on bad input
    instead of bubbling up an opaque RPC error 30 seconds later.
    """
    if not is_valid_address(s):
        sys.stderr.write(
            f"error: invalid {field} {s!r}: expected 0x-prefixed 40-hex-char address\n"
        )
        sys.exit(2)
    return s.lower()


def require_txhash(s: str, *, field: str = "tx hash") -> str:
    """Return `s` lowercased if valid, else exit with an error message."""
    if not is_valid_txhash(s):
        sys.stderr.write(
            f"error: invalid {field} {s!r}: expected 0x-prefixed 64-hex-char tx hash\n"
        )
        sys.exit(2)
    return s.lower()


def wei_to_native(wei: int, decimals: int = 18) -> float:
    return wei / (10 ** decimals)


def gwei_from_wei(wei: int) -> float:
    return wei / 1e9

def _short_addr(addr: str) -> str:
    if addr and len(addr) >= 10:
        return addr[:6] + "..." + addr[-4:]
    return addr or ""

def print_json(data: Any) -> None:
    print(json.dumps(data, indent=2, default=str))

# ---------------------------------------------------------------------------
# HTTP / JSON-RPC layer
# ---------------------------------------------------------------------------

def _http_post(url: str, payload: Any, retries: int = 5, timeout: int = 20) -> Any:
    body = json.dumps(payload).encode()
    headers = {
        "Content-Type": "application/json",
        "Accept":       "application/json",
        "User-Agent":   "Mozilla/5.0 (compatible; evm_client/1.0)",
    }
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    delay = 1.0
    last_err: Exception = RuntimeError("No attempts made")
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(delay)
                delay = min(delay * 2, 30)
                last_err = e
                continue
            body_text = ""
            try:
                body_text = e.read().decode()
            except Exception:
                pass
            raise RuntimeError(f"HTTP {e.code}: {body_text}") from e
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(delay)
                delay = min(delay * 2, 30)
    raise RuntimeError(f"Request failed after {retries} retries: {last_err}") from last_err

def _http_get(url: str, retries: int = 5, timeout: int = 20) -> Any:
    headers = {"Accept": "application/json", "User-Agent": "evm_client/1.0"}
    req = urllib.request.Request(url, headers=headers, method="GET")
    delay = 1.0
    last_err: Exception = RuntimeError("No attempts made")
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(delay)
                delay = min(delay * 2, 30)
                last_err = e
                continue
            body_text = ""
            try:
                body_text = e.read().decode()
            except Exception:
                pass
            raise RuntimeError(f"HTTP {e.code}: {body_text}") from e
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(delay)
                delay = min(delay * 2, 30)
    raise RuntimeError(f"Request failed after {retries} retries: {last_err}") from last_err

# ---------------------------------------------------------------------------
# RPC helpers
# ---------------------------------------------------------------------------

def get_rpc_url(chain: str) -> str:
    env = os.environ.get("EVM_RPC_URL", "")
    if env:
        return env
    cfg = CHAINS.get(chain)
    if not cfg:
        raise ValueError(f"Unknown chain '{chain}'. Available: {', '.join(CHAINS)}")
    return cfg["rpc"]

def rpc_call(chain: str, method: str, params: List[Any], req_id: int = 1) -> Any:
    url = get_rpc_url(chain)
    payload = {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params}
    resp = _http_post(url, payload)
    if "error" in resp:
        raise RuntimeError(f"RPC error: {resp['error']}")
    return resp.get("result")

def rpc_batch(chain: str, calls: List[Tuple[str, List[Any]]], batch_limit: int = 10) -> List[Any]:
    """Send a batch of JSON-RPC calls; returns list of results in same order.

    Auto-chunks at `batch_limit` (default 10) so we stay under per-RPC limits.
    Base's public RPC caps batches at 10 — exceeding that returns a single error
    dict instead of a results list, which would mask all our calls.
    """
    url = get_rpc_url(chain)

    # Build the full payload, preserving order via JSON-RPC `id`
    items = [
        {"jsonrpc": "2.0", "id": i, "method": m, "params": p}
        for i, (m, p) in enumerate(calls)
    ]

    out: List[Any] = [None] * len(items)
    for start in range(0, len(items), batch_limit):
        chunk = items[start:start + batch_limit]
        resp = _http_post(url, chunk)
        if not isinstance(resp, list):
            # Single error response (e.g. batch-too-large) — leave this chunk as None
            continue
        for r in resp:
            rid = r.get("id")
            if isinstance(rid, int) and 0 <= rid < len(out):
                if "error" in r:
                    out[rid] = None
                else:
                    out[rid] = r.get("result")
    return out

# ---------------------------------------------------------------------------
# ABI encoding helpers (minimal, for ERC-20 calls)
# ---------------------------------------------------------------------------

def _encode_address(addr: str) -> str:
    """Pad address to 32 bytes."""
    return addr.lower().replace("0x", "").zfill(64)

def _keccak256(data: bytes) -> bytes:
    """Pure Python Keccak-256 (Ethereum's hash, NOT SHA3-256)."""
    # Keccak-256 round constants
    RC = [
        0x0000000000000001, 0x0000000000008082, 0x800000000000808A, 0x8000000080008000,
        0x000000000000808B, 0x0000000080000001, 0x8000000080008081, 0x8000000000008009,
        0x000000000000008A, 0x0000000000000088, 0x0000000080008009, 0x000000008000000A,
        0x000000008000808B, 0x800000000000008B, 0x8000000000008089, 0x8000000000008003,
        0x8000000000008002, 0x8000000000000080, 0x000000000000800A, 0x800000008000000A,
        0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008,
    ]
    ROT = [
        [0, 36, 3, 41, 18], [1, 44, 10, 45, 2], [62, 6, 43, 15, 61],
        [28, 55, 25, 21, 56], [27, 20, 39, 8, 14],
    ]
    def rot64(x, n): return ((x << n) | (x >> (64 - n))) & 0xFFFFFFFFFFFFFFFF
    rate = 136  # 1088 bits for keccak-256
    # Padding
    msg = bytearray(data)
    msg.append(0x01)
    while len(msg) % rate != 0:
        msg.append(0x00)
    msg[-1] |= 0x80
    # Absorb
    state = [0] * 25
    for block_start in range(0, len(msg), rate):
        block = msg[block_start:block_start + rate]
        for i in range(rate // 8):
            state[i] ^= int.from_bytes(block[i*8:(i+1)*8], "little")
        # Keccak-f[1600]
        for rnd in range(24):
            # Theta
            C = [state[x] ^ state[x+5] ^ state[x+10] ^ state[x+15] ^ state[x+20] for x in range(5)]
            D = [C[(x-1) % 5] ^ rot64(C[(x+1) % 5], 1) for x in range(5)]
            state = [state[i] ^ D[i % 5] for i in range(25)]
            # Rho + Pi
            B = [0] * 25
            for x in range(5):
                for y in range(5):
                    B[y*5 + ((2*x+3*y) % 5)] = rot64(state[x + 5*y], ROT[x][y])
            # Chi
            state = [B[i] ^ ((~B[(i//5)*5 + (i%5+1)%5]) & B[(i//5)*5 + (i%5+2)%5]) for i in range(25)]
            # Iota
            state[0] ^= RC[rnd]
    # Squeeze
    out = b"".join(state[i].to_bytes(8, "little") for i in range(4))
    return out


def _selector(sig: str) -> str:
    """Compute 4-byte function selector via keccak-256."""
    return "0x" + _keccak256(sig.encode()).hex()[:8]

# Precomputed selectors for ERC-20 functions
ERC20_SELECTORS: Dict[str, str] = {
    "name()":                  "0x06fdde03",
    "symbol()":                "0x95d89b41",
    "decimals()":              "0x313ce567",
    "totalSupply()":           "0x18160ddd",
    "balanceOf(address)":      "0x70a08231",
}

def eth_call_erc20(chain: str, contract: str, fn: str, arg_addr: Optional[str] = None) -> str:
    selector = ERC20_SELECTORS[fn]
    data = selector
    if arg_addr:
        data += _encode_address(arg_addr)
    params = [{"to": contract, "data": data}, "latest"]
    return rpc_call(chain, "eth_call", params) or "0x"

def decode_string(hex_data: str) -> str:
    """Decode ABI-encoded string from eth_call result."""
    try:
        raw = hex_data[2:] if hex_data.startswith("0x") else hex_data
        if len(raw) < 128:
            # Try decoding as raw bytes (some tokens return non-ABI strings)
            b = bytes.fromhex(raw)
            return b.rstrip(b"\x00").decode("utf-8", errors="replace").strip()
        # offset (skip 32 bytes), length, data
        length = int(raw[64:128], 16)
        chars = raw[128:128 + length * 2]
        return bytes.fromhex(chars).decode("utf-8", errors="replace").strip()
    except Exception:
        return ""

def decode_uint256(hex_data: str) -> int:
    try:
        raw = hex_data[2:] if hex_data.startswith("0x") else hex_data
        if not raw:
            return 0
        return int(raw, 16)
    except Exception:
        return 0

def decode_uint8(hex_data: str) -> int:
    return decode_uint256(hex_data)

# ---------------------------------------------------------------------------
# CoinGecko price fetching
# ---------------------------------------------------------------------------

COINGECKO_BASE = "https://api.coingecko.com/api/v3"

def cg_price_by_id(cg_id: str) -> Optional[float]:
    try:
        url = f"{COINGECKO_BASE}/simple/price?ids={cg_id}&vs_currencies=usd"
        data = _http_get(url)
        return data.get(cg_id, {}).get("usd")
    except Exception:
        return None

def cg_price_by_ids(cg_ids: List[str]) -> Dict[str, float]:
    """Fetch multiple prices in one request."""
    if not cg_ids:
        return {}
    try:
        joined = ",".join(cg_ids)
        url = f"{COINGECKO_BASE}/simple/price?ids={joined}&vs_currencies=usd"
        data = _http_get(url)
        return {k: v.get("usd", 0.0) for k, v in data.items() if "usd" in v}
    except Exception:
        return {}

def cg_price_by_contract(chain: str, contract: str) -> Optional[float]:
    cg_platform_map = {
        "ethereum": "ethereum",
        "bsc":      "binance-smart-chain",
        "base":     "base",
        "arbitrum": "arbitrum-one",
        "polygon":  "polygon-pos",
        "optimism": "optimistic-ethereum",
        "avalanche":"avalanche",
        "zksync":   "zksync",
    }
    platform = cg_platform_map.get(chain)
    if not platform:
        return None
    try:
        url = (
            f"{COINGECKO_BASE}/simple/token_price/{platform}"
            f"?contract_addresses={contract}&vs_currencies=usd"
        )
        data = _http_get(url)
        addr_lower = contract.lower()
        for k, v in data.items():
            if k.lower() == addr_lower:
                return v.get("usd")
        return None
    except Exception:
        return None

def get_native_price(chain: str) -> Optional[float]:
    cg_id = CHAINS[chain]["coingecko"]
    return cg_price_by_id(cg_id)

# ---------------------------------------------------------------------------
# Command implementations
# ---------------------------------------------------------------------------

def cmd_stats(args: argparse.Namespace) -> None:
    chain = args.chain
    cfg = CHAINS[chain]

    # Batch: blockNumber + gasPrice
    results = rpc_batch(chain, [
        ("eth_blockNumber", []),
        ("eth_gasPrice",    []),
    ])
    block_num = hex_to_int(results[0] or "0x0")
    gas_price_wei = hex_to_int(results[1] or "0x0")

    # TPS estimate: compare latest block timestamp with parent
    tps: Optional[float] = None
    try:
        latest_block = rpc_call(chain, "eth_getBlockByNumber", ["latest", False])
        if latest_block:
            parent_hex = latest_block.get("parentHash")
            parent_block = rpc_call(chain, "eth_getBlockByHash", [parent_hex, False])
            if parent_block:
                t1 = hex_to_int(latest_block.get("timestamp", "0x0"))
                t0 = hex_to_int(parent_block.get("timestamp", "0x0"))
                tx_count = len(latest_block.get("transactions", []))
                if t1 > t0:
                    tps = round(tx_count / (t1 - t0), 2)
    except Exception:
        pass

    native_price = get_native_price(chain)

    print_json({
        "chain":           chain,
        "block_number":    block_num,
        "gas_price_gwei":  round(gwei_from_wei(gas_price_wei), 4),
        "gas_price_wei":   gas_price_wei,
        "native_token":    cfg["native"],
        "native_price_usd": native_price,
        "tps_estimate":    tps,
        "explorer":        cfg["explorer"],
    })


def cmd_wallet(args: argparse.Namespace) -> None:
    address = require_address(args.address)
    chain   = args.chain
    limit   = args.limit
    no_prices = args.no_prices
    cfg     = CHAINS[chain]

    # Native balance
    balance_hex = rpc_call(chain, "eth_getBalance", [address, "latest"])
    native_wei  = hex_to_int(balance_hex or "0x0")
    native_val  = wei_to_native(native_wei, cfg["decimals"])

    native_usd_price: Optional[float] = None
    native_usd: Optional[float] = None
    if not no_prices:
        native_usd_price = get_native_price(chain)
        if native_usd_price is not None:
            native_usd = round(native_val * native_usd_price, 4)

    # ERC-20 tokens
    token_list = list((KNOWN_TOKENS.get(chain) or {}).items())[:limit]
    tokens_out = []
    portfolio_usd = native_usd or 0.0

    if token_list:
        # Batch balanceOf calls
        balance_calls = [
            ("eth_call", [{"to": addr, "data": ERC20_SELECTORS["balanceOf(address)"] + _encode_address(address)}, "latest"])
            for _, addr in token_list
        ]
        balances = rpc_batch(chain, balance_calls)

        for idx, (symbol, addr) in enumerate(token_list):
            raw_bal = decode_uint256(balances[idx] or "0x0")
            if raw_bal == 0:
                continue

            # Fetch decimals
            dec_hex = eth_call_erc20(chain, addr, "decimals()")
            decimals = decode_uint8(dec_hex) if dec_hex and dec_hex != "0x" else 18
            bal_human = wei_to_native(raw_bal, decimals)

            token_price: Optional[float] = None
            token_usd: Optional[float] = None
            if not no_prices:
                try:
                    cg_id = COINGECKO_IDS.get(symbol)
                    if cg_id:
                        token_price = cg_price_by_id(cg_id)
                    if token_price is None:
                        token_price = cg_price_by_contract(chain, addr)
                    if token_price is not None:
                        token_usd = round(bal_human * token_price, 4)
                        portfolio_usd += token_usd
                except Exception:
                    pass

            tokens_out.append({
                "symbol":       symbol,
                "contract":     addr,
                "balance":      round(bal_human, 8),
                "price_usd":    token_price,
                "value_usd":    token_usd,
            })

    print_json({
        "chain":             chain,
        "address":           address,
        "native_token":      cfg["native"],
        "native_balance":    round(native_val, 8),
        "native_price_usd":  native_usd_price,
        "native_value_usd":  native_usd,
        "erc20_tokens":      tokens_out,
        "portfolio_total_usd": round(portfolio_usd, 4) if not no_prices else None,
    })


def cmd_tx(args: argparse.Namespace) -> None:
    tx_hash = require_txhash(args.hash)
    chain   = args.chain
    cfg     = CHAINS[chain]

    results = rpc_batch(chain, [
        ("eth_getTransactionByHash",       [tx_hash]),
        ("eth_getTransactionReceipt",      [tx_hash]),
    ])
    tx      = results[0]
    receipt = results[1]

    if not tx:
        print_json({"error": f"Transaction {tx_hash} not found on {chain}"})
        return

    block_num = hex_to_int(tx.get("blockNumber") or "0x0")
    timestamp: Optional[int] = None
    try:
        blk = rpc_call(chain, "eth_getBlockByNumber", [hex(block_num), False])
        if blk:
            timestamp = hex_to_int(blk.get("timestamp", "0x0"))
    except Exception:
        pass

    value_wei  = hex_to_int(tx.get("value", "0x0"))
    value_eth  = wei_to_native(value_wei, cfg["decimals"])
    gas_price  = hex_to_int(tx.get("gasPrice") or "0x0")
    gas_limit  = hex_to_int(tx.get("gas", "0x0"))
    gas_used   = hex_to_int((receipt or {}).get("gasUsed", "0x0")) if receipt else None
    status     = None
    if receipt:
        status = "success" if hex_to_int(receipt.get("status", "0x0")) == 1 else "failed"

    input_data = tx.get("input", "0x")
    input_preview = input_data[:66] + ("..." if len(input_data) > 66 else "")

    native_price = get_native_price(chain)
    value_usd = round(value_eth * native_price, 4) if native_price else None

    fee_eth: Optional[float] = None
    fee_usd: Optional[float] = None
    if gas_used is not None:
        fee_eth = wei_to_native(gas_used * gas_price, cfg["decimals"])
        if native_price:
            fee_usd = round(fee_eth * native_price, 6)

    print_json({
        "chain":          chain,
        "hash":           tx_hash,
        "block":          block_num,
        "timestamp":      timestamp,
        "from":           tx.get("from"),
        "to":             tx.get("to"),
        "value":          round(value_eth, 8),
        "value_usd":      value_usd,
        "native_token":   cfg["native"],
        "gas_limit":      gas_limit,
        "gas_used":       gas_used,
        "gas_price_gwei": round(gwei_from_wei(gas_price), 4),
        "fee_native":     round(fee_eth, 8) if fee_eth is not None else None,
        "fee_usd":        fee_usd,
        "status":         status,
        "input_preview":  input_preview,
        "nonce":          hex_to_int(tx.get("nonce", "0x0")),
        "explorer_url":   f"{cfg['explorer']}/tx/{tx_hash}",
    })


def cmd_token(args: argparse.Namespace) -> None:
    contract = require_address(args.contract, field="contract address")
    chain    = args.chain

    # Batch all ERC-20 metadata calls
    calls = [
        ("eth_call", [{"to": contract, "data": ERC20_SELECTORS["name()"]},        "latest"]),
        ("eth_call", [{"to": contract, "data": ERC20_SELECTORS["symbol()"]},       "latest"]),
        ("eth_call", [{"to": contract, "data": ERC20_SELECTORS["decimals()"]},     "latest"]),
        ("eth_call", [{"to": contract, "data": ERC20_SELECTORS["totalSupply()"]},  "latest"]),
    ]
    results  = rpc_batch(chain, calls)
    name     = decode_string(results[0] or "0x")
    symbol   = decode_string(results[1] or "0x")
    decimals = decode_uint8(results[2] or "0x0")
    supply_raw = decode_uint256(results[3] or "0x0")
    supply   = wei_to_native(supply_raw, decimals)

    price: Optional[float] = None
    market_cap: Optional[float] = None
    cg_id = COINGECKO_IDS.get(symbol.upper())
    if cg_id:
        price = cg_price_by_id(cg_id)
    if price is None:
        price = cg_price_by_contract(chain, contract)
    if price is not None and supply > 0:
        market_cap = round(price * supply, 2)

    cfg = CHAINS[chain]
    print_json({
        "chain":        chain,
        "contract":     contract,
        "name":         name,
        "symbol":       symbol,
        "decimals":     decimals,
        "total_supply": round(supply, 4),
        "price_usd":    price,
        "market_cap_usd": market_cap,
        "explorer_url": f"{cfg['explorer']}/token/{contract}",
    })


def cmd_activity(args: argparse.Namespace) -> None:
    address = require_address(args.address)
    chain   = args.chain
    limit   = args.limit
    cfg     = CHAINS[chain]

    # Get current block
    block_hex = rpc_call(chain, "eth_blockNumber", [])
    latest    = hex_to_int(block_hex or "0x0")

    txs_out: List[Dict[str, Any]] = []
    scan_range = min(200, latest)
    blocks_checked = 0

    for bn in range(latest, max(0, latest - scan_range), -1):
        if len(txs_out) >= limit:
            break
        try:
            blk = rpc_call(chain, "eth_getBlockByNumber", [hex(bn), True])
        except Exception:
            continue
        if not blk:
            continue
        blocks_checked += 1
        timestamp = hex_to_int(blk.get("timestamp", "0x0"))
        for tx in blk.get("transactions", []):
            if len(txs_out) >= limit:
                break
            frm = (tx.get("from") or "").lower()
            to  = (tx.get("to")   or "").lower()
            addr_lower = address.lower()
            if frm == addr_lower or to == addr_lower:
                value_wei = hex_to_int(tx.get("value", "0x0"))
                value_eth = wei_to_native(value_wei, cfg["decimals"])
                gas_price = hex_to_int(tx.get("gasPrice") or "0x0")
                txs_out.append({
                    "hash":           tx.get("hash"),
                    "block":          bn,
                    "timestamp":      timestamp,
                    "from":           tx.get("from"),
                    "to":             tx.get("to"),
                    "value":          round(value_eth, 8),
                    "native_token":   cfg["native"],
                    "gas_price_gwei": round(gwei_from_wei(gas_price), 4),
                    "direction":      "out" if frm == addr_lower else "in",
                })

    print_json({
        "chain":          chain,
        "address":        address,
        "blocks_scanned": blocks_checked,
        "tx_count":       len(txs_out),
        "transactions":   txs_out,
    })


def cmd_gas(args: argparse.Namespace) -> None:
    chain = args.chain
    cfg   = CHAINS[chain]

    gas_price_hex = rpc_call(chain, "eth_gasPrice", [])
    gas_wei       = hex_to_int(gas_price_hex or "0x0")
    gas_gwei      = gwei_from_wei(gas_wei)

    native_price  = get_native_price(chain)

    estimates: Dict[str, Any] = {}
    for op, gas_units in GAS_ESTIMATES.items():
        cost_wei   = gas_wei * gas_units
        cost_native = wei_to_native(cost_wei, cfg["decimals"])
        cost_usd    = round(cost_native * native_price, 6) if native_price else None
        estimates[op] = {
            "gas_units":   gas_units,
            "cost_native": round(cost_native, 8),
            "cost_usd":    cost_usd,
        }

    print_json({
        "chain":           chain,
        "native_token":    cfg["native"],
        "gas_price_gwei":  round(gas_gwei, 4),
        "gas_price_wei":   gas_wei,
        "native_price_usd": native_price,
        "estimates":       estimates,
    })


def cmd_price(args: argparse.Namespace) -> None:
    token = args.token
    chain = args.chain

    price: Optional[float] = None
    source = "unknown"

    # Check if it's a contract address
    if token.startswith("0x") and len(token) >= 10:
        price = cg_price_by_contract(chain, token)
        source = "coingecko_contract"
        if price is None:
            print_json({"error": f"Could not find price for contract {token} on {chain}"})
            return
    else:
        symbol = token.upper()
        cg_id  = COINGECKO_IDS.get(symbol)
        if cg_id:
            price  = cg_price_by_id(cg_id)
            source = f"coingecko:{cg_id}"
        if price is None:
            # Try known tokens on given chain
            contract = (KNOWN_TOKENS.get(chain) or {}).get(symbol)
            if contract:
                price  = cg_price_by_contract(chain, contract)
                source = f"coingecko_contract:{contract}"
        if price is None:
            print_json({"error": f"Could not find price for '{token}'. Try a contract address."})
            return

    print_json({
        "token":     token,
        "chain":     chain,
        "price_usd": price,
        "source":    source,
    })


def _fetch_chain_stats(chain: str) -> Dict[str, Any]:
    """Fetch gas price + native price for a single chain (used in compare)."""
    try:
        gas_hex = rpc_call(chain, "eth_gasPrice", [])
        gas_wei = hex_to_int(gas_hex or "0x0")
        gas_gwei = round(gwei_from_wei(gas_wei), 4)
    except Exception:
        gas_gwei = None

    cg_id = CHAINS[chain]["coingecko"]
    native_price = cg_price_by_id(cg_id)

    transfer_usd: Optional[float] = None
    if gas_gwei is not None and native_price is not None:
        gas_wei_val = int(gas_gwei * 1e9)
        cost_wei    = gas_wei_val * GAS_ESTIMATES["transfer"]
        cost_native = wei_to_native(cost_wei, CHAINS[chain]["decimals"])
        transfer_usd = round(cost_native * native_price, 6)

    return {
        "chain":             chain,
        "native_token":      CHAINS[chain]["native"],
        "gas_price_gwei":    gas_gwei,
        "native_price_usd":  native_price,
        "transfer_cost_usd": transfer_usd,
    }


def cmd_compare(_args: argparse.Namespace) -> None:
    """Compare gas prices and native token prices across all chains simultaneously."""
    import threading

    results: Dict[str, Any] = {}
    errors:  Dict[str, str] = {}
    lock = threading.Lock()

    def fetch(chain: str) -> None:
        try:
            data = _fetch_chain_stats(chain)
            with lock:
                results[chain] = data
        except Exception as e:
            with lock:
                errors[chain] = str(e)

    threads = [threading.Thread(target=fetch, args=(c,), daemon=True) for c in CHAINS]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    sorted_by_gas = sorted(
        results.values(),
        key=lambda x: x.get("gas_price_gwei") or float("inf"),
    )

    print_json({
        "comparison":       sorted_by_gas,
        "errors":           errors,
        "cheapest_gas":     sorted_by_gas[0]["chain"] if sorted_by_gas else None,
        "most_expensive_gas": sorted_by_gas[-1]["chain"] if sorted_by_gas else None,
    })


def cmd_whale(args: argparse.Namespace) -> None:
    chain    = args.chain
    blocks   = args.blocks
    min_usd  = args.min_usd
    cfg      = CHAINS[chain]

    native_price = get_native_price(chain)
    if native_price is None:
        print_json({"error": "Could not fetch native token price for USD conversion."})
        return

    block_hex = rpc_call(chain, "eth_blockNumber", [])
    latest    = hex_to_int(block_hex or "0x0")

    whales: List[Dict[str, Any]] = []
    blocks_scanned = 0

    for bn in range(latest, max(0, latest - blocks), -1):
        try:
            blk = rpc_call(chain, "eth_getBlockByNumber", [hex(bn), True])
        except Exception:
            continue
        if not blk:
            continue
        blocks_scanned += 1
        timestamp = hex_to_int(blk.get("timestamp", "0x0"))

        for tx in blk.get("transactions", []):
            value_wei = hex_to_int(tx.get("value", "0x0"))
            if value_wei == 0:
                continue
            value_native = wei_to_native(value_wei, cfg["decimals"])
            value_usd    = value_native * native_price
            if value_usd >= min_usd:
                whales.append({
                    "hash":         tx.get("hash"),
                    "block":        bn,
                    "timestamp":    timestamp,
                    "from":         tx.get("from"),
                    "from_short":   _short_addr(tx.get("from") or ""),
                    "to":           tx.get("to"),
                    "to_short":     _short_addr(tx.get("to") or ""),
                    "value_native": round(value_native, 6),
                    "native_token": cfg["native"],
                    "value_usd":    round(value_usd, 2),
                })

    whales.sort(key=lambda x: x["value_usd"], reverse=True)

    print_json({
        "chain":           chain,
        "blocks_scanned":  blocks_scanned,
        "latest_block":    latest,
        "min_usd":         min_usd,
        "native_price_usd": native_price,
        "whale_count":     len(whales),
        "transfers":       whales,
    })


# ---------------------------------------------------------------------------
# New commands: multichain, allowance, decode, ens, contract
# ---------------------------------------------------------------------------

def cmd_multichain(args: argparse.Namespace) -> None:
    """Scan same wallet across all 8 chains simultaneously."""
    import threading

    address = require_address(args.address)
    results: Dict[str, Any] = {}
    lock = threading.Lock()

    def scan_chain(chain: str) -> None:
        cfg = CHAINS[chain]
        try:
            bal_hex = rpc_call(chain, "eth_getBalance", [address, "latest"])
            native_bal = int(bal_hex, 16) / 1e18 if bal_hex else 0.0
            native_price = get_native_price(chain)
            native_usd = round(native_bal * native_price, 2) if native_price else None
            entry: Dict[str, Any] = {
                "native_symbol": cfg["native"],
                "native_balance": round(native_bal, 8),
                "native_price_usd": native_price,
                "native_value_usd": native_usd,
                "tokens": [],
                "total_usd": native_usd or 0.0,
            }
            # Check known tokens for this chain.
            # KNOWN_TOKENS[chain] maps {symbol: contract_address}, not {addr: (sym, name)}.
            known = KNOWN_TOKENS.get(chain, {})
            for symbol, contract in known.items():
                raw = eth_call_erc20(chain, contract, "balanceOf(address)", address)
                if not raw or raw == "0x":
                    continue
                try:
                    bal_int = int(raw, 16)
                except Exception:
                    continue
                if bal_int == 0:
                    continue
                dec_raw = eth_call_erc20(chain, contract, "decimals()")
                decimals = decode_uint8(dec_raw) if dec_raw else 18
                human = bal_int / (10 ** decimals)
                tok_price = cg_price_by_contract(chain, contract)
                tok_usd = round(human * tok_price, 2) if tok_price else None
                entry["tokens"].append({
                    "symbol": symbol,
                    "balance": round(human, 6),
                    "value_usd": tok_usd,
                })
                if tok_usd:
                    entry["total_usd"] = round(entry["total_usd"] + tok_usd, 2)
            with lock:
                results[chain] = entry
        except Exception as exc:
            with lock:
                results[chain] = {"error": str(exc)}

    threads = [threading.Thread(target=scan_chain, args=(c,)) for c in CHAINS]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    grand_total = sum(
        v.get("total_usd", 0) for v in results.values() if isinstance(v, dict)
    )
    print_json({
        "address": address,
        "chains": results,
        "grand_total_usd": round(grand_total, 2),
    })


def cmd_allowance(args: argparse.Namespace) -> None:
    """Check dangerous ERC-20 approvals for a wallet (known spenders)."""
    address = require_address(args.address)
    chain = args.chain

    # Well-known spender contracts (DEXes, bridges, etc.)
    KNOWN_SPENDERS = {
        "0x000000000022D473030F116dDEE9F6B43aC78BA3": "Permit2 (Uniswap)",
        "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D": "Uniswap V2 Router",
        "0xE592427A0AEce92De3Edee1F18E0157C05861564": "Uniswap V3 Router",
        "0x68b3465833fb72A70ecDF485E0e4C7bD8665Fc45": "Uniswap Universal Router",
        "0x1111111254EEB25477B68fb85Ed929f73A960582": "1inch Router V5",
        "0x6131B5fae19EA4f9D964eAc0408E4408b66337b5": "KyberSwap Router",
        "0xDef1C0ded9bec7F1a1670819833240f027b25EfF": "0x Exchange Proxy",
        "0x3fc91a3afd70395cd496c647d5a6cc9d4b2b7fad": "Uniswap Universal Router 2",
    }

    known = KNOWN_TOKENS.get(chain, {})
    approvals = []

    # KNOWN_TOKENS[chain] is {symbol: contract_address}, not {addr: (sym, name)}.
    for symbol, contract in known.items():
        for spender_addr, spender_name in KNOWN_SPENDERS.items():
            # allowance(owner, spender) = 0xdd62ed3e
            owner_pad  = address.lower().replace("0x", "").zfill(64)
            spender_pad = spender_addr.lower().replace("0x", "").zfill(64)
            data = "0xdd62ed3e" + owner_pad + spender_pad
            raw = rpc_call(chain, "eth_call", [{"to": contract, "data": data}, "latest"])
            if not raw or raw == "0x":
                continue
            try:
                allowance_int = int(raw, 16)
            except Exception:
                continue
            if allowance_int == 0:
                continue

            dec_raw = eth_call_erc20(chain, contract, "decimals()")
            decimals = decode_uint8(dec_raw) if dec_raw else 18
            max_uint = 2**256 - 1
            is_unlimited = allowance_int >= max_uint // 2

            approvals.append({
                "token": symbol,
                "contract": contract,
                "spender": spender_name,
                "spender_address": spender_addr,
                "allowance": "UNLIMITED" if is_unlimited else str(round(allowance_int / 10**decimals, 4)),
                "risk": "HIGH" if is_unlimited else "LOW",
            })

    print_json({
        "chain": chain,
        "address": address,
        "approvals_found": len(approvals),
        "approvals": approvals,
        "note": "Only checks known DEX/bridge spenders. Use a full allowance checker for complete coverage.",
    })


def cmd_decode(args: argparse.Namespace) -> None:
    """Decode transaction input data using 4byte.directory."""
    chain = args.chain
    tx_hash = require_txhash(args.hash)

    tx = rpc_call(chain, "eth_getTransactionByHash", [tx_hash])
    if not tx:
        print_json({"error": "Transaction not found"})
        return

    input_data: str = tx.get("input", "0x")
    if not input_data or input_data == "0x":
        print_json({
            "chain": chain,
            "hash": tx_hash,
            "decoded": None,
            "note": "No input data (plain ETH transfer)",
        })
        return

    selector = input_data[:10]  # 0x + 4 bytes = 10 chars

    # Query 4byte.directory
    url = f"https://www.4byte.directory/api/v1/signatures/?hex_signature={selector}"
    data = _http_get(url)

    signatures = []
    if data and data.get("results"):
        signatures = [r["text_signature"] for r in data["results"]]

    # Decode known transfer(address,uint256) manually as fallback
    decoded_args: Optional[Dict] = None
    if signatures and len(input_data) >= 74:
        sig = signatures[0]
        if sig == "transfer(address,uint256)" and len(input_data) == 138:
            to_addr = "0x" + input_data[34:74]
            amount_hex = input_data[74:]
            try:
                amount = int(amount_hex, 16)
                decoded_args = {"to": to_addr, "amount_raw": amount}
            except Exception:
                pass

    print_json({
        "chain": chain,
        "hash": tx_hash,
        "selector": selector,
        "input_length_bytes": (len(input_data) - 2) // 2,
        "from": tx.get("from"),
        "to": tx.get("to"),
        "signatures": signatures,
        "primary_signature": signatures[0] if signatures else None,
        "decoded_args": decoded_args,
        "raw_input_preview": input_data[:74] + ("..." if len(input_data) > 74 else ""),
        "source": "4byte.directory",
    })


def cmd_ens(args: argparse.Namespace) -> None:
    """Resolve ENS name <-> address via ensideas.com public API (no key needed)."""
    query = args.name_or_address

    # ensideas.com handles both forward (name->address) and reverse (address->name)
    try:
        data = _http_get(f"https://api.ensideas.com/ens/resolve/{query}")
    except Exception as exc:
        print_json({"error": str(exc), "note": "ENS API unavailable"})
        return

    if not data:
        print_json({"query": query, "address": None, "ens_name": None, "note": "Not found"})
        return

    print_json({
        "query":      query,
        "address":    data.get("address"),
        "ens_name":   data.get("name"),
        "avatar":     data.get("avatar"),
        "display":    data.get("displayName"),
        "twitter":    data.get("twitter"),
        "github":     data.get("github"),
        "source":     "ensideas.com",
    })


def cmd_contract(args: argparse.Namespace) -> None:
    """Inspect a smart contract: bytecode size, proxy detection, creation info."""
    chain = args.chain
    address = require_address(args.address)

    # Get bytecode
    code_hex = rpc_call(chain, "eth_getCode", [address, "latest"])
    if not code_hex or code_hex == "0x":
        print_json({"chain": chain, "address": address, "is_contract": False, "note": "EOA (externally owned account)"})
        return

    bytecode_bytes = (len(code_hex) - 2) // 2

    # Proxy detection patterns
    # EIP-1967: implementation slot 0x360894a13ba1a3210667c828492db98dca3e2076cc3735a920a3ca505d382bbc
    impl_slot = "0x360894a13ba1a3210667c828492db98dca3e2076cc3735a920a3ca505d382bbc"
    impl_raw = rpc_call(chain, "eth_getStorageAt", [address, impl_slot, "latest"])
    implementation = None
    is_proxy = False
    if impl_raw and impl_raw != "0x" and int(impl_raw, 16) != 0:
        is_proxy = True
        implementation = "0x" + impl_raw[-40:]

    # EIP-1167 minimal proxy detection (starts with 0x363d3d37)
    if code_hex[2:10] == "363d3d37" or code_hex[2:18] == "3d602d80600a3d39":
        is_proxy = True

    # supportsInterface check: ERC-165
    supports_erc165 = False
    try:
        erc165_data = "0x01ffc9a701ffc9a700000000000000000000000000000000000000000000000000000000"
        erc165_raw = rpc_call(chain, "eth_call", [{"to": address, "data": erc165_data}, "latest"])
        supports_erc165 = bool(erc165_raw and erc165_raw != "0x" and int(erc165_raw, 16) == 1)
    except Exception:
        pass

    # Try to detect ERC-20 (has totalSupply)
    is_erc20 = False
    try:
        ts_raw = eth_call_erc20(chain, address, "totalSupply()")
        is_erc20 = ts_raw is not None and ts_raw != "0x" and int(ts_raw, 16) > 0
    except Exception:
        pass

    # Try to detect ERC-721 (supportsInterface 0x80ac58cd)
    is_erc721 = False
    try:
        erc721_data = "0x01ffc9a780ac58cd00000000000000000000000000000000000000000000000000000000"
        erc721_raw = rpc_call(chain, "eth_call", [{"to": address, "data": erc721_data}, "latest"])
        is_erc721 = bool(erc721_raw and erc721_raw != "0x" and int(erc721_raw, 16) == 1)
    except Exception:
        pass

    detected_standards = []
    if is_erc20:
        detected_standards.append("ERC-20")
    if is_erc721:
        detected_standards.append("ERC-721")
    if supports_erc165:
        detected_standards.append("ERC-165")

    print_json({
        "chain": chain,
        "address": address,
        "is_contract": True,
        "bytecode_size_bytes": bytecode_bytes,
        "is_proxy": is_proxy,
        "implementation": implementation,
        "detected_standards": detected_standards,
        "explorer_url": f"{CHAINS[chain]['explorer']}/address/{address}",
        "note": "Proxy detected via EIP-1967 storage slot. Standards via EIP-165 + heuristics." if is_proxy else None,
    })


# ---------------------------------------------------------------------------
# Argument parsing & dispatch
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    chain_choices = list(CHAINS.keys())

    parser = argparse.ArgumentParser(
        prog="evm_client",
        description="EVM blockchain CLI — stdlib only, zero dependencies.",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # -- stats --
    p_stats = sub.add_parser("stats", help="Chain stats: block, gas price, native price, TPS")
    p_stats.add_argument("--chain", default=DEFAULT_CHAIN, choices=chain_choices)

    # -- wallet --
    p_wallet = sub.add_parser("wallet", help="Wallet balance + ERC-20 portfolio")
    p_wallet.add_argument("address", help="Wallet address (0x...)")
    p_wallet.add_argument("--limit",     type=int, default=20, metavar="N",
                          help="Max number of known tokens to check (default: 20)")
    p_wallet.add_argument("--no-prices", action="store_true",
                          help="Skip USD price lookups (faster)")
    p_wallet.add_argument("--chain", default=DEFAULT_CHAIN, choices=chain_choices)

    # -- tx --
    p_tx = sub.add_parser("tx", help="Transaction details")
    p_tx.add_argument("hash", help="Transaction hash (0x...)")
    p_tx.add_argument("--chain", default=DEFAULT_CHAIN, choices=chain_choices)

    # -- token --
    p_token = sub.add_parser("token", help="ERC-20 token metadata + price")
    p_token.add_argument("contract", help="Token contract address (0x...)")
    p_token.add_argument("--chain", default=DEFAULT_CHAIN, choices=chain_choices)

    # -- activity --
    p_act = sub.add_parser("activity", help="Recent transactions for an address")
    p_act.add_argument("address", help="Wallet address (0x...)")
    p_act.add_argument("--limit", type=int, default=10, metavar="N",
                       help="Max transactions to return (default: 10)")
    p_act.add_argument("--chain", default=DEFAULT_CHAIN, choices=chain_choices)

    # -- gas --
    p_gas = sub.add_parser("gas", help="Gas prices and cost estimates")
    p_gas.add_argument("--chain", default=DEFAULT_CHAIN, choices=chain_choices)

    # -- price --
    p_price = sub.add_parser("price", help="Token price by symbol or contract address")
    p_price.add_argument("token", help="Symbol (e.g. ETH, USDC) or contract address")
    p_price.add_argument("--chain", default=DEFAULT_CHAIN, choices=chain_choices)

    # -- compare --
    sub.add_parser("compare", help="Gas + native prices across ALL chains simultaneously")

    # -- whale --
    p_whale = sub.add_parser("whale", help="Scan for large value transfers in recent blocks")
    p_whale.add_argument("--blocks",  type=int, default=20, metavar="N",
                         help="Number of recent blocks to scan (default: 20)")
    p_whale.add_argument("--min-usd", type=float, default=10_000.0, metavar="N",
                         help="Minimum USD value to report (default: 10000)")
    p_whale.add_argument("--chain", default=DEFAULT_CHAIN, choices=chain_choices)

    # -- multichain --
    p_multi = sub.add_parser("multichain", help="Scan same wallet across ALL chains simultaneously")
    p_multi.add_argument("address", help="Wallet address (0x...)")

    # -- allowance --
    p_allow = sub.add_parser("allowance", help="Check dangerous ERC-20 approvals (known DEX/bridge spenders)")
    p_allow.add_argument("address", help="Wallet address (0x...)")
    p_allow.add_argument("--chain", default=DEFAULT_CHAIN, choices=chain_choices)

    # -- decode --
    p_decode = sub.add_parser("decode", help="Decode transaction input data via 4byte.directory")
    p_decode.add_argument("hash", help="Transaction hash (0x...)")
    p_decode.add_argument("--chain", default=DEFAULT_CHAIN, choices=chain_choices)

    # -- ens --
    p_ens = sub.add_parser("ens", help="Resolve ENS name <-> address (Ethereum only)")
    p_ens.add_argument("name_or_address", help="ENS name (vitalik.eth) or address (0x...)")

    # -- contract --
    p_contract = sub.add_parser("contract", help="Inspect a smart contract: proxy, standards, bytecode size")
    p_contract.add_argument("address", help="Contract address (0x...)")
    p_contract.add_argument("--chain", default=DEFAULT_CHAIN, choices=chain_choices)

    return parser


DISPATCH = {
    "stats":      cmd_stats,
    "wallet":     cmd_wallet,
    "tx":         cmd_tx,
    "token":      cmd_token,
    "activity":   cmd_activity,
    "gas":        cmd_gas,
    "price":      cmd_price,
    "compare":    cmd_compare,
    "whale":      cmd_whale,
    "multichain": cmd_multichain,
    "allowance":  cmd_allowance,
    "decode":     cmd_decode,
    "ens":        cmd_ens,
    "contract":   cmd_contract,
}


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    # Validate chain exists (argparse choices already handles this, but for ENV override)
    if hasattr(args, "chain") and args.chain not in CHAINS:
        print_json({"error": f"Unknown chain '{args.chain}'. Available: {list(CHAINS.keys())}"})
        sys.exit(1)

    cmd_fn = DISPATCH.get(args.command)
    if cmd_fn is None:
        print_json({"error": f"Unknown command '{args.command}'"})
        sys.exit(1)

    try:
        cmd_fn(args)
    except KeyboardInterrupt:
        print_json({"error": "Interrupted by user"})
        sys.exit(130)
    except Exception as e:
        print_json({"error": str(e)})
        sys.exit(1)


if __name__ == "__main__":
    main()
