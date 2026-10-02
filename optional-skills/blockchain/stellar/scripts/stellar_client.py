#!/usr/bin/env python3
"""
Stellar CLI Tool for Hermes Agent
---------------------------------
Read-only queries against Stellar Horizon and Stellar RPC, with USD values
taken from the on-chain DEX (XLM/USDC order book). Standard library only.

Usage:
  python3 stellar_client.py stats
  python3 stellar_client.py account  <G... | name*domain> [--no-prices] [--limit N]
  python3 stellar_client.py tx       <hash>
  python3 stellar_client.py activity <G...> [--limit N]
  python3 stellar_client.py asset    <CODE:ISSUER | CODE>
  python3 stellar_client.py price    <XLM | CODE | CODE:ISSUER>
  python3 stellar_client.py contract <C...>
  python3 stellar_client.py events   <C...> [--ledgers N] [--limit N]
  python3 stellar_client.py toml     <domain>
  python3 stellar_client.py fund     <G...>              (testnet only)

Every command accepts --network mainnet|testnet and --json.

Environment:
  STELLAR_NETWORK      mainnet (default) or testnet
  STELLAR_HORIZON_URL  Override the Horizon base URL
  STELLAR_RPC_URL      Override the Stellar RPC URL
"""

from __future__ import annotations

import argparse
import base64
import binascii
import json
from decimal import Decimal, InvalidOperation
import os
import struct
import sys
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

USER_AGENT = "HermesAgent/1.0"
STROOPS_PER_XLM = 10_000_000
BASE_RESERVE_XLM = 0.5
COINGECKO_XLM_URL = "https://api.coingecko.com/api/v3/simple/price?ids=stellar&vs_currencies=usd"

NETWORKS: Dict[str, Dict[str, Optional[str]]] = {
    "mainnet": {
        "horizon": "https://horizon.stellar.org",
        "rpc": "https://mainnet.sorobanrpc.com",
        "passphrase": "Public Global Stellar Network ; September 2015",
        "friendbot": None,
        "explorer": "https://stellar.expert/explorer/public",
    },
    "testnet": {
        "horizon": "https://horizon-testnet.stellar.org",
        "rpc": "https://soroban-testnet.stellar.org",
        "passphrase": "Test SDF Network ; September 2015",
        "friendbot": "https://friendbot.stellar.org",
        "explorer": "https://stellar.expert/explorer/testnet",
    },
}

# Well-known classic assets (code -> issuer). Verified against Horizon /assets.
KNOWN_ASSETS: Dict[str, Dict[str, str]] = {
    "mainnet": {
        "USDC": "GA5ZSEJYB37JRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVN",
        "EURC": "GDHU6WRG4IEQXM5NZ4BMPKOXHW76MZM4Y2IEMFDVXBSDP6SJY4ITNPP2",
        "AQUA": "GBNZILSTVQZ4R7IKQDGHYGY2QXL5QOFJYQMXPKWRRM5PAV7Y4M67AQUA",
        "YXLM": "GARDNV3Q7YGT4AKSDF25LT32YSCCW4EV22Y2TV3I2PU2MMXJTEDL5T55",
        "YUSDC": "GDGTVWSM4MGS4T7Z6W4RPWOCHE2I6RDFCIFZGS3DOA63LWQTRNZNTTFF",
        "SHX": "GDSTRSHXHGJ7ZIVRBXEYE5Q74XUVCUSEKEBR7UCHEUUEK72N7I7KJ6JH",
        "ARST": "GCSAZVWXZKWS4XS223M5F54H2B6XPIIXZZGP7KEAIU6YSL5HDRGCI3DG",
        "VELO": "GDM4RQUQQUVSKQA7S6EM7XBZP3FCGH4Q7CL6TABQ7B2BEJ5ERARM2M5M",
    },
    "testnet": {
        "USDC": "GBBD47IF6LWK7P7MDEVSCWR7DPUWV3NY3DTQEVFL4NAT4AQH3ZLLFLA5",
    },
}


class ClientError(Exception):
    """User-facing failure. Carries the HTTP status when one applies."""

    def __init__(self, message: str, status: Optional[int] = None):
        super().__init__(message)
        self.status = status


# ---------------------------------------------------------------------------
# Environment / network configuration
# ---------------------------------------------------------------------------

def _hermes_home() -> Path:
    return Path(os.environ.get("HERMES_HOME", "~/.hermes")).expanduser()


def _dotenv_paths() -> List[Path]:
    paths: List[Path] = []
    project_env = Path.cwd() / ".env"
    if project_env.exists():
        paths.append(project_env)
    user_env = _hermes_home() / ".env"
    if user_env.exists():
        paths.append(user_env)
    return paths


def _load_dotenv_values() -> Dict[str, str]:
    """Later files win: the user's HERMES_HOME/.env overrides a project .env."""
    values: Dict[str, str] = {}
    for env_path in _dotenv_paths():
        try:
            lines = env_path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            lines = env_path.read_text(encoding="latin-1").splitlines()
        for raw_line in lines:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = raw_line.partition("=")
            key, value = key.strip(), value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            values[key] = value
    return values


def _env_lookup(key: str, default: str = "") -> str:
    value = os.environ.get(key, "").strip()
    if value:
        return value
    dotenv_value = _load_dotenv_values().get(key, "").strip()
    return dotenv_value or default


def resolve_config(network: Optional[str]) -> Dict[str, Any]:
    name = (network or _env_lookup("STELLAR_NETWORK", "mainnet")).strip().lower()
    if name in ("pubnet", "public"):
        name = "mainnet"
    if name not in NETWORKS:
        raise ClientError(f"Unknown network {name!r}. Use mainnet or testnet.")
    cfg: Dict[str, Any] = dict(NETWORKS[name])
    cfg["name"] = name
    cfg["horizon"] = _env_lookup("STELLAR_HORIZON_URL", cfg["horizon"]).rstrip("/")
    cfg["rpc"] = _env_lookup("STELLAR_RPC_URL", cfg["rpc"]).rstrip("/")
    cfg["known_assets"] = KNOWN_ASSETS.get(name, {})
    return cfg


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _describe_http_error(url: str, code: int, body: str) -> str:
    host = urllib.parse.urlsplit(url).netloc
    try:
        problem = json.loads(body) if body else {}
    except ValueError:
        problem = {}
    if isinstance(problem, dict) and problem.get("title"):
        detail = problem.get("detail") or ""
        extras = problem.get("extras") or {}
        codes = extras.get("result_codes") if isinstance(extras, dict) else None
        suffix = f" result_codes={json.dumps(codes)}" if codes else ""
        return f"HTTP {code} from {host}: {problem['title']}. {detail}{suffix}".strip()
    return f"HTTP {code} from {host}"


def _http_raw(url: str, payload: Optional[Dict[str, Any]] = None, timeout: int = 20, retries: int = 2) -> bytes:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Accept": "application/json, text/plain;q=0.9, */*;q=0.5", "User-Agent": USER_AGENT}
    if data is not None:
        headers["Content-Type"] = "application/json"
    for attempt in range(retries + 1):
        request = urllib.request.Request(url, data=data, headers=headers, method="POST" if data else "GET")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace") if exc.fp else ""
            if exc.code in (429, 502, 503, 504) and attempt < retries:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise ClientError(_describe_http_error(url, exc.code, body), status=exc.code) from None
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if attempt < retries:
                time.sleep(1.0 * (attempt + 1))
                continue
            reason = getattr(exc, "reason", exc)
            raise ClientError(f"Network error reaching {url}: {reason}") from None
    raise ClientError(f"Gave up reaching {url}")


def _http_json(url: str, payload: Optional[Dict[str, Any]] = None, timeout: int = 20, retries: int = 2) -> Any:
    raw = _http_raw(url, payload=payload, timeout=timeout, retries=retries)
    if not raw:
        return {}
    try:
        return json.loads(raw.decode("utf-8"))
    except ValueError:
        raise ClientError(f"Non-JSON response from {urllib.parse.urlsplit(url).netloc}") from None


def horizon_get(cfg: Dict[str, Any], path: str, **params: Any) -> Any:
    query = {k: v for k, v in params.items() if v is not None}
    url = cfg["horizon"] + path
    if query:
        url += "?" + urllib.parse.urlencode(query)
    return _http_json(url)


def rpc_call(cfg: Dict[str, Any], method: str, params: Optional[Dict[str, Any]] = None) -> Any:
    body = _http_json(cfg["rpc"], payload={"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}})
    if not isinstance(body, dict):
        raise ClientError(f"Unexpected RPC response for {method}")
    if "error" in body:
        err = body["error"]
        if isinstance(err, dict):
            raise ClientError(f"RPC {method} failed ({err.get('code')}): {err.get('message')}")
        raise ClientError(f"RPC {method} failed: {err}")
    return body.get("result")


def _records(page: Any) -> List[Dict[str, Any]]:
    if isinstance(page, dict):
        return list((page.get("_embedded") or {}).get("records") or [])
    return []


# ---------------------------------------------------------------------------
# StrKey (SEP-23) encoding
# ---------------------------------------------------------------------------

_STRKEY_VERSIONS = {
    "G": 6 << 3,    # ed25519 public key (account)
    "M": 12 << 3,   # muxed account
    "C": 2 << 3,    # contract
    "L": 11 << 3,   # liquidity pool
    "B": 1 << 3,    # claimable balance
    "S": 18 << 3,   # ed25519 secret seed (never printed; recognised only)
    "T": 19 << 3,   # pre-auth tx hash
    "X": 23 << 3,   # sha256 hash
    "P": 15 << 3,   # signed payload
}
_STRKEY_PREFIX_BY_VERSION = {v: k for k, v in _STRKEY_VERSIONS.items()}
_B32_ALPHABET = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ234567")


def crc16_xmodem(data: bytes) -> int:
    crc = 0
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) if crc & 0x8000 else (crc << 1)
            crc &= 0xFFFF
    return crc


def strkey_encode(prefix: str, payload: bytes) -> str:
    body = bytes([_STRKEY_VERSIONS[prefix]]) + payload
    checksum = struct.pack("<H", crc16_xmodem(body))
    return base64.b32encode(body + checksum).decode("ascii").rstrip("=")


def strkey_decode(value: str) -> Tuple[str, bytes]:
    text = (value or "").strip().upper()
    if not text or any(ch not in _B32_ALPHABET for ch in text):
        raise ClientError(f"Invalid StrKey {value!r}: not base32")
    try:
        raw = base64.b32decode(text + "=" * (-len(text) % 8))
    except (binascii.Error, ValueError):
        raise ClientError(f"Invalid StrKey {value!r}: bad length") from None
    if len(raw) < 3:
        raise ClientError(f"Invalid StrKey {value!r}: too short")
    body, checksum = raw[:-2], raw[-2:]
    if struct.pack("<H", crc16_xmodem(body)) != checksum:
        raise ClientError(f"Invalid StrKey {value!r}: checksum mismatch (typo?)")
    prefix = _STRKEY_PREFIX_BY_VERSION.get(body[0])
    if prefix is None:
        raise ClientError(f"Invalid StrKey {value!r}: unknown version byte {body[0]}")
    return prefix, body[1:]


def is_account_id(value: str) -> bool:
    try:
        prefix, payload = strkey_decode(value)
    except ClientError:
        return False
    return prefix == "G" and len(payload) == 32


def is_contract_id(value: str) -> bool:
    try:
        prefix, payload = strkey_decode(value)
    except ClientError:
        return False
    return prefix == "C" and len(payload) == 32


def require_account_id(value: str) -> str:
    value = (value or "").strip()
    if value.startswith("S") and len(value) == 56:
        raise ClientError("That is a SECRET seed (S...). Never paste secrets; use the public G... address.")
    if not is_account_id(value):
        raise ClientError(f"{value!r} is not a valid Stellar account id (G..., 56 chars)")
    return value.upper()


def require_contract_id(value: str) -> str:
    value = (value or "").strip()
    if not is_contract_id(value):
        raise ClientError(f"{value!r} is not a valid Stellar contract id (C..., 56 chars)")
    return value.upper()


# ---------------------------------------------------------------------------
# XDR decoding (the subset needed for ScVal, ledger entries, tx results)
# ---------------------------------------------------------------------------

class XdrReader:
    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0

    def take(self, n: int) -> bytes:
        if self.pos + n > len(self.data):
            raise ClientError("Truncated XDR")
        chunk = self.data[self.pos:self.pos + n]
        self.pos += n
        return chunk

    def int32(self) -> int:
        return struct.unpack(">i", self.take(4))[0]

    def uint32(self) -> int:
        return struct.unpack(">I", self.take(4))[0]

    def int64(self) -> int:
        return struct.unpack(">q", self.take(8))[0]

    def uint64(self) -> int:
        return struct.unpack(">Q", self.take(8))[0]

    def boolean(self) -> bool:
        return self.int32() != 0

    def opaque(self) -> bytes:
        length = self.uint32()
        data = self.take(length)
        self.take((-length) % 4)
        return data


SC_ERROR_TYPES = ["CONTRACT", "WASM_VM", "CONTEXT", "STORAGE", "OBJECT", "CRYPTO", "EVENTS", "BUDGET", "VALUE", "AUTH"]
SC_ERROR_CODES = [
    "ARITH_DOMAIN", "INDEX_BOUNDS", "INVALID_INPUT", "MISSING_VALUE", "EXISTING_VALUE",
    "EXCEEDED_LIMIT", "INVALID_ACTION", "INTERNAL_ERROR", "UNEXPECTED_TYPE", "UNEXPECTED_SIZE",
]


def _name(names: List[str], index: int) -> str:
    return names[index] if 0 <= index < len(names) else str(index)


def decode_scaddress(reader: XdrReader) -> str:
    kind = reader.int32()
    if kind == 0:  # SC_ADDRESS_TYPE_ACCOUNT -> PublicKey (type + 32 bytes)
        reader.int32()
        return strkey_encode("G", reader.take(32))
    if kind == 1:  # SC_ADDRESS_TYPE_CONTRACT
        return strkey_encode("C", reader.take(32))
    if kind == 2:  # SC_ADDRESS_TYPE_MUXED_ACCOUNT { uint64 id; uint256 ed25519 }
        muxed_id = reader.uint64()
        key = reader.take(32)
        return strkey_encode("M", key + struct.pack(">Q", muxed_id))
    if kind == 3:  # SC_ADDRESS_TYPE_CLAIMABLE_BALANCE
        balance_type = reader.int32()
        return strkey_encode("B", bytes([balance_type]) + reader.take(32))
    if kind == 4:  # SC_ADDRESS_TYPE_LIQUIDITY_POOL
        return strkey_encode("L", reader.take(32))
    raise ClientError(f"Unknown ScAddress type {kind}")


def _key_label(key: Any) -> Optional[str]:
    """Render a map key as a short label: `"METADATA"`, `Balance(GABC...)`. None if not representable."""
    if isinstance(key, str):
        return key
    if isinstance(key, (int, bool)) or key is None:
        return str(key)
    if isinstance(key, list) and key and isinstance(key[0], str):
        if len(key) == 1:
            return key[0]
        if all(isinstance(part, (str, int, bool)) for part in key[1:]):
            return f"{key[0]}({', '.join(str(part) for part in key[1:])})"
    return None


def _pairs_to_map(pairs: List[Tuple[Any, Any]]) -> Any:
    labels = [_key_label(key) for key, _ in pairs]
    if all(label is not None for label in labels) and len(set(labels)) == len(labels):
        return {label: value for label, (_, value) in zip(labels, pairs)}
    return [{"key": key, "value": value} for key, value in pairs]


def decode_scval(reader: XdrReader) -> Any:
    kind = reader.int32()
    if kind == 0:
        return reader.boolean()
    if kind == 1:
        return None
    if kind == 2:
        err_type = reader.int32()
        if err_type == 0:
            return f"Error(Contract, #{reader.uint32()})"
        return f"Error({_name(SC_ERROR_TYPES, err_type)}, {_name(SC_ERROR_CODES, reader.int32())})"
    if kind == 3:
        return reader.uint32()
    if kind == 4:
        return reader.int32()
    if kind in (5, 7, 8):  # u64, timepoint, duration
        return reader.uint64()
    if kind == 6:
        return reader.int64()
    if kind == 9:  # u128 {hi, lo}
        hi, lo = reader.uint64(), reader.uint64()
        return (hi << 64) | lo
    if kind == 10:  # i128 {int64 hi, uint64 lo}
        hi, lo = reader.int64(), reader.uint64()
        return (hi << 64) + lo
    if kind == 11:  # u256
        parts = [reader.uint64() for _ in range(4)]
        return (parts[0] << 192) | (parts[1] << 128) | (parts[2] << 64) | parts[3]
    if kind == 12:  # i256 {int64 hi_hi, uint64 ...}
        hi_hi = reader.int64()
        rest = [reader.uint64() for _ in range(3)]
        return (hi_hi << 192) + (rest[0] << 128) + (rest[1] << 64) + rest[2]
    if kind == 13:
        return "0x" + reader.opaque().hex()
    if kind in (14, 15):  # string, symbol
        return reader.opaque().decode("utf-8", "replace")
    if kind == 16:  # vec (optional)
        if not reader.boolean():
            return None
        return [decode_scval(reader) for _ in range(reader.uint32())]
    if kind == 17:  # map (optional)
        if not reader.boolean():
            return None
        count = reader.uint32()
        return _pairs_to_map([(decode_scval(reader), decode_scval(reader)) for _ in range(count)])
    if kind == 18:
        return decode_scaddress(reader)
    if kind == 19:
        return decode_contract_instance(reader)
    if kind == 20:
        return "<ledger-key-contract-instance>"
    if kind == 21:
        return {"nonce": reader.int64()}
    raise ClientError(f"Unknown ScVal type {kind}")


def decode_contract_instance(reader: XdrReader) -> Dict[str, Any]:
    executable_type = reader.int32()
    if executable_type == 0:
        executable: Dict[str, Any] = {"type": "wasm", "wasm_hash": reader.take(32).hex()}
    else:
        executable = {"type": "stellar_asset"}
    storage = None
    if reader.boolean():
        count = reader.uint32()
        storage = _pairs_to_map([(decode_scval(reader), decode_scval(reader)) for _ in range(count)])
    return {"executable": executable, "storage": storage}


def decode_scval_b64(value: str) -> Any:
    try:
        return decode_scval(XdrReader(base64.b64decode(value)))
    except (ClientError, binascii.Error, ValueError):
        return {"undecoded_xdr": value}


CONTRACT_CODE_COST_FIELDS = [
    "n_instructions", "n_functions", "n_globals", "n_table_entries", "n_types",
    "n_data_segments", "n_elem_segments", "n_imports", "n_exports", "n_data_segment_bytes",
]


def decode_ledger_entry_data(value: str) -> Dict[str, Any]:
    """Decode the `xdr` field of a getLedgerEntries entry (LedgerEntryData)."""
    reader = XdrReader(base64.b64decode(value))
    kind = reader.int32()
    if kind == 6:  # CONTRACT_DATA
        reader.int32()  # ExtensionPoint v0
        contract = decode_scaddress(reader)
        key = decode_scval(reader)
        durability = "temporary" if reader.int32() == 0 else "persistent"
        val = decode_scval(reader)
        return {"type": "contract_data", "contract": contract, "key": key, "durability": durability, "value": val}
    if kind == 7:  # CONTRACT_CODE
        ext_version = reader.int32()
        cost_inputs = None
        if ext_version == 1:
            reader.int32()  # v1.ext ExtensionPoint
            reader.int32()  # costInputs.ext ExtensionPoint
            cost_inputs = {field: reader.uint32() for field in CONTRACT_CODE_COST_FIELDS}
        wasm_hash = reader.take(32).hex()
        code = reader.opaque()
        return {"type": "contract_code", "wasm_hash": wasm_hash, "code_bytes": len(code), "cost_inputs": cost_inputs}
    return {"type": f"ledger_entry_type_{kind}"}


def contract_instance_key(contract_id: str) -> str:
    _, contract_hash = strkey_decode(require_contract_id(contract_id))
    key = struct.pack(">i", 6) + struct.pack(">i", 1) + contract_hash + struct.pack(">i", 20) + struct.pack(">i", 1)
    return base64.b64encode(key).decode("ascii")


def contract_code_key(wasm_hash_hex: str) -> str:
    return base64.b64encode(struct.pack(">i", 7) + bytes.fromhex(wasm_hash_hex)).decode("ascii")


TX_RESULT_CODES = {
    0: "txSUCCESS", 1: "txFEE_BUMP_INNER_SUCCESS", -1: "txFAILED", -2: "txTOO_EARLY", -3: "txTOO_LATE",
    -4: "txMISSING_OPERATION", -5: "txBAD_SEQ", -6: "txBAD_AUTH", -7: "txINSUFFICIENT_BALANCE",
    -8: "txNO_ACCOUNT", -9: "txINSUFFICIENT_FEE", -10: "txBAD_AUTH_EXTRA", -11: "txINTERNAL_ERROR",
    -12: "txNOT_SUPPORTED", -13: "txFEE_BUMP_INNER_FAILED", -14: "txBAD_SPONSORSHIP",
    -15: "txBAD_MIN_SEQ_AGE_OR_GAP", -16: "txMALFORMED", -17: "txSOROBAN_INVALID",
}
OP_RESULT_CODES = {
    0: "opINNER", -1: "opBAD_AUTH", -2: "opNO_ACCOUNT", -3: "opNOT_SUPPORTED",
    -4: "opTOO_MANY_SUBENTRIES", -5: "opEXCEEDED_WORK_LIMIT", -6: "opTOO_MANY_SPONSORING",
}
OPERATION_TYPES = [
    "create_account", "payment", "path_payment_strict_receive", "manage_sell_offer",
    "create_passive_sell_offer", "set_options", "change_trust", "allow_trust", "account_merge",
    "inflation", "manage_data", "bump_sequence", "manage_buy_offer", "path_payment_strict_send",
    "create_claimable_balance", "claim_claimable_balance", "begin_sponsoring_future_reserves",
    "end_sponsoring_future_reserves", "revoke_sponsorship", "clawback", "clawback_claimable_balance",
    "set_trust_line_flags", "liquidity_pool_deposit", "liquidity_pool_withdraw",
    "invoke_host_function", "extend_footprint_ttl", "restore_footprint",
]
# Per-operation failure names for the most common operation types.
OP_FAILURE_NAMES: Dict[int, List[str]] = {
    0: ["MALFORMED", "UNDERFUNDED", "LOW_RESERVE", "ALREADY_EXIST"],
    1: ["MALFORMED", "UNDERFUNDED", "SRC_NO_TRUST", "NO_DESTINATION", "NO_TRUST", "NOT_AUTHORIZED", "LINE_FULL", "NO_ISSUER"],
    2: ["MALFORMED", "UNDERFUNDED", "SRC_NO_TRUST", "SRC_NOT_AUTHORIZED", "NO_DESTINATION", "NO_TRUST",
        "NOT_AUTHORIZED", "LINE_FULL", "NO_ISSUER", "TOO_FEW_OFFERS", "OFFER_CROSS_SELF", "OVER_SENDMAX"],
    6: ["MALFORMED", "NO_ISSUER", "INVALID_LIMIT", "LOW_RESERVE", "SELF_NOT_ALLOWED", "TRUST_LINE_MISSING",
        "CANNOT_DELETE", "NOT_AUTH_MAINTAIN_LIABILITIES"],
    13: ["MALFORMED", "UNDERFUNDED", "SRC_NO_TRUST", "SRC_NOT_AUTHORIZED", "NO_DESTINATION", "NO_TRUST",
         "NOT_AUTHORIZED", "LINE_FULL", "NO_ISSUER", "TOO_FEW_OFFERS", "OFFER_CROSS_SELF", "UNDER_DESTMIN"],
    24: ["MALFORMED", "TRAPPED", "RESOURCE_LIMIT_EXCEEDED", "ENTRY_ARCHIVED", "INSUFFICIENT_REFUNDABLE_FEE"],
    25: ["MALFORMED", "RESOURCE_LIMIT_EXCEEDED", "INSUFFICIENT_REFUNDABLE_FEE"],
    26: ["MALFORMED", "RESOURCE_LIMIT_EXCEEDED", "INSUFFICIENT_REFUNDABLE_FEE"],
}
# Operation types whose *success* result body we can skip to keep reading.
_VOID_SUCCESS_OPS = {0, 1, 5, 6, 7, 10, 11, 15, 16, 17, 18, 19, 20, 21, 25, 26}


def _decode_operation_results(reader: XdrReader) -> List[str]:
    results: List[str] = []
    count = reader.uint32()
    for index in range(count):
        code = reader.int32()
        if code != 0:
            results.append(f"op{index}: {OP_RESULT_CODES.get(code, code)}")
            break
        op_type = reader.int32()
        op_name = _name(OPERATION_TYPES, op_type)
        inner = reader.int32()
        if inner == 0:
            results.append(f"op{index} {op_name}: SUCCESS")
            if op_type == 8:
                reader.int64()  # account_merge: source balance
            elif op_type == 24:
                reader.take(32)  # invoke_host_function: return value hash
            elif op_type not in _VOID_SUCCESS_OPS:
                break  # offer/path-payment/LP bodies are not worth decoding here
            continue
        names = OP_FAILURE_NAMES.get(op_type, [])
        label = names[-inner - 1] if 0 < -inner <= len(names) else str(inner)
        results.append(f"op{index} {op_name}: {label}")
        break
    return results


def decode_tx_result(result_xdr: Optional[str]) -> Dict[str, Any]:
    """Decode fee, tx-level code and (best effort) per-operation codes from a TransactionResult."""
    if not result_xdr:
        return {}
    try:
        reader = XdrReader(base64.b64decode(result_xdr))
        fee_charged = reader.int64()
        code = reader.int32()
        out: Dict[str, Any] = {"fee_charged_stroops": fee_charged, "code": TX_RESULT_CODES.get(code, f"code {code}")}
        if code in (1, -13):  # fee bump: inner result follows
            reader.take(32)
            reader.int64()
            inner_code = reader.int32()
            out["inner_code"] = TX_RESULT_CODES.get(inner_code, f"code {inner_code}")
            code = inner_code
        if code in (0, -1):
            out["operations"] = _decode_operation_results(reader)
        return out
    except (ClientError, binascii.Error, ValueError, struct.error):
        return {"undecoded_xdr": result_xdr}


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def fmt_amount(value: Any, max_decimals: int = 7) -> str:
    try:
        number = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return str(value)
    text = f"{number:,.{max_decimals}f}".rstrip("0").rstrip(".")
    return text if text not in ("", "-0") else "0"


def fmt_usd(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    if abs(value) >= 1:
        return f"${value:,.2f}"
    return f"${value:,.6f}".rstrip("0").rstrip(".")


def fmt_stroops(stroops: Any) -> str:
    try:
        return fmt_amount(int(stroops) / STROOPS_PER_XLM) + " XLM"
    except (TypeError, ValueError):
        return str(stroops)


def short(value: Any, head: int = 6, tail: int = 4) -> str:
    text = str(value or "")
    return text if len(text) <= head + tail + 3 else f"{text[:head]}...{text[-tail:]}"


def asset_label(record: Dict[str, Any], prefix: str = "") -> str:
    asset_type = record.get(f"{prefix}asset_type") or record.get("asset_type")
    if asset_type == "native":
        return "XLM"
    if asset_type == "liquidity_pool_shares":
        return "LP:" + short(record.get("liquidity_pool_id"), 6, 4)
    code = record.get(f"{prefix}asset_code") or record.get("asset_code") or "?"
    issuer = record.get(f"{prefix}asset_issuer") or record.get("asset_issuer") or ""
    return f"{code}:{short(issuer, 4, 4)}" if issuer else code


def parse_asset(text: str, cfg: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    """Return (code, issuer); issuer is None for XLM."""
    raw = (text or "").strip()
    if not raw:
        raise ClientError("Asset is required (XLM, CODE, or CODE:ISSUER)")
    if raw.upper() in ("XLM", "NATIVE"):
        return "XLM", None
    sep = ":" if ":" in raw else ("-" if "-" in raw else None)
    if sep:
        code, issuer = raw.split(sep, 1)
        code, issuer = code.strip(), issuer.strip()
        if not (1 <= len(code) <= 12) or not code.isalnum():
            raise ClientError(f"Invalid asset code {code!r} (1-12 alphanumeric chars)")
        return code, require_account_id(issuer)
    known = cfg["known_assets"].get(raw.upper())
    if not known:
        names = ", ".join(sorted(cfg["known_assets"])) or "none on this network"
        raise ClientError(f"Unknown asset {raw!r}. Use CODE:ISSUER. Known codes: {names}")
    # keep the canonical casing of the known code (e.g. yXLM)
    canonical = next((k for k in cfg["known_assets"] if k.upper() == raw.upper()), raw.upper())
    canonical = {"YXLM": "yXLM", "YUSDC": "yUSDC"}.get(canonical, canonical)
    return canonical, known


def _asset_query(prefix: str, code: str, issuer: Optional[str]) -> Dict[str, str]:
    if issuer is None:
        return {f"{prefix}_asset_type": "native"}
    asset_type = "credit_alphanum4" if len(code) <= 4 else "credit_alphanum12"
    return {f"{prefix}_asset_type": asset_type, f"{prefix}_asset_code": code, f"{prefix}_asset_issuer": issuer}


# ---------------------------------------------------------------------------
# Pricing via the on-chain DEX (Horizon order book)
# ---------------------------------------------------------------------------

class Pricer:
    """Prices come from Horizon order books: XLM/USDC for XLM, ASSET/XLM for everything else."""

    def __init__(self, cfg: Dict[str, Any], enabled: bool = True):
        self.cfg = cfg
        self.enabled = enabled
        self._xlm_usd: Optional[float] = None
        self._xlm_source = "none"
        self._cache: Dict[Tuple[str, Optional[str]], Optional[Dict[str, Any]]] = {}

    def order_book_mid(self, base: Tuple[str, Optional[str]], counter: Tuple[str, Optional[str]]) -> Optional[Dict[str, Any]]:
        params = {**_asset_query("selling", *base), **_asset_query("buying", *counter), "limit": 1}
        try:
            book = horizon_get(self.cfg, "/order_book", **params)
        except ClientError:
            return None
        bids, asks = book.get("bids") or [], book.get("asks") or []
        bid = float(bids[0]["price"]) if bids else None
        ask = float(asks[0]["price"]) if asks else None
        if bid is None and ask is None:
            return None
        if bid is not None and ask is not None:
            mid = (bid + ask) / 2
            spread = (ask - bid) / mid * 100 if mid else None
        else:
            mid, spread = (bid if bid is not None else ask), None
        return {"mid": mid, "bid": bid, "ask": ask, "spread_pct": spread}

    def xlm_usd(self) -> Optional[float]:
        if not self.enabled:
            return None
        if self._xlm_usd is not None:
            return self._xlm_usd
        usdc_issuer = self.cfg["known_assets"].get("USDC")
        if usdc_issuer:
            book = self.order_book_mid(("XLM", None), ("USDC", usdc_issuer))
            if book and book["mid"]:
                self._xlm_usd = book["mid"]
                self._xlm_source = "dex XLM/USDC" if self.cfg["name"] == "mainnet" else "testnet dex (not a real price)"
                return self._xlm_usd
        if self.cfg["name"] == "mainnet":
            try:
                data = _http_json(COINGECKO_XLM_URL, timeout=10, retries=1)
                self._xlm_usd = float(data["stellar"]["usd"])
                self._xlm_source = "coingecko"
            except (ClientError, KeyError, TypeError, ValueError):
                self._xlm_usd = None
        return self._xlm_usd

    @property
    def xlm_source(self) -> str:
        return self._xlm_source

    def asset_price(self, code: str, issuer: Optional[str]) -> Optional[Dict[str, Any]]:
        """Return {price_xlm, price_usd, spread_pct, bid, ask} or None."""
        if not self.enabled:
            return None
        key = (code, issuer)
        if key in self._cache:
            return self._cache[key]
        xlm_usd = self.xlm_usd()
        if issuer is None:
            result = {"price_xlm": 1.0, "price_usd": xlm_usd, "spread_pct": None, "source": self._xlm_source} if xlm_usd else None
        else:
            book = self.order_book_mid((code, issuer), ("XLM", None))
            if book is None:
                result = None
            else:
                result = {
                    "price_xlm": book["mid"],
                    "price_usd": book["mid"] * xlm_usd if xlm_usd else None,
                    "spread_pct": book["spread_pct"],
                    "bid_xlm": book["bid"],
                    "ask_xlm": book["ask"],
                    "source": f"dex {code}/XLM",
                }
        self._cache[key] = result
        return result


# ---------------------------------------------------------------------------
# SEP-1 stellar.toml and SEP-2 federation
# ---------------------------------------------------------------------------

SEP_ENDPOINTS = {
    "FEDERATION_SERVER": "SEP-2 federation",
    "TRANSFER_SERVER": "SEP-6 deposit/withdraw API",
    "TRANSFER_SERVER_SEP0024": "SEP-24 hosted deposit/withdraw",
    "WEB_AUTH_ENDPOINT": "SEP-10 web auth",
    "WEB_AUTH_FOR_CONTRACTS_ENDPOINT": "SEP-45 web auth for contracts",
    "KYC_SERVER": "SEP-12 KYC",
    "DIRECT_PAYMENT_SERVER": "SEP-31 cross-border payments",
    "ANCHOR_QUOTE_SERVER": "SEP-38 quotes",
    "AUTH_SERVER": "SEP-3 compliance (legacy)",
}


def normalize_domain(text: str) -> str:
    raw = (text or "").strip()
    if "://" in raw:
        raw = urllib.parse.urlsplit(raw).netloc
    raw = raw.split("/")[0].strip().lower().rstrip(".")
    if not raw or " " in raw or "." not in raw:
        raise ClientError(f"{text!r} is not a domain name")
    return raw


def fetch_stellar_toml(domain: str) -> Dict[str, Any]:
    url = f"https://{domain}/.well-known/stellar.toml"
    try:
        raw = _http_raw(url, timeout=15, retries=1)
    except ClientError as exc:
        if exc.status == 404:
            raise ClientError(f"No stellar.toml published at {url}", status=404) from None
        raise
    try:
        return tomllib.loads(raw.decode("utf-8", "replace"))
    except tomllib.TOMLDecodeError as exc:
        raise ClientError(f"{url} is not valid TOML: {exc}") from None


def federation_lookup(address: str) -> Dict[str, Any]:
    if address.count("*") != 1:
        raise ClientError(f"{address!r} is not a federation address (name*domain)")
    _, domain = address.split("*", 1)
    domain = normalize_domain(domain)
    toml_data = fetch_stellar_toml(domain)
    server = toml_data.get("FEDERATION_SERVER")
    if not server:
        raise ClientError(f"{domain} publishes no FEDERATION_SERVER in its stellar.toml")
    url = str(server).rstrip("/") + "?" + urllib.parse.urlencode({"q": address, "type": "name"})
    result = _http_json(url, timeout=15, retries=1)
    account = result.get("account_id") if isinstance(result, dict) else None
    if not account or not is_account_id(account):
        raise ClientError(f"Federation server at {domain} returned no account for {address}")
    return {"address": address, "account_id": account, "memo_type": result.get("memo_type"), "memo": result.get("memo")}


# ---------------------------------------------------------------------------
# Horizon operation summaries
# ---------------------------------------------------------------------------

def _host_function_name(record: Dict[str, Any]) -> str:
    text = str(record.get("function") or "")
    return text.replace("HostFunctionTypeHostFunctionType", "") or "unknown"


def summarize_operation(op: Dict[str, Any]) -> Dict[str, Any]:
    kind = op.get("type", "?")
    summary: Dict[str, Any] = {"type": kind, "id": op.get("id"), "source": op.get("source_account")}
    if kind == "payment":
        summary["text"] = f"{short(op.get('from'))} -> {short(op.get('to'))}: {fmt_amount(op.get('amount'))} {asset_label(op)}"
    elif kind == "create_account":
        summary["text"] = f"{short(op.get('funder'))} funded {short(op.get('account'))} with {fmt_amount(op.get('starting_balance'))} XLM"
    elif kind in ("path_payment_strict_send", "path_payment_strict_receive"):
        summary["text"] = (
            f"{short(op.get('from'))} -> {short(op.get('to'))}: sent {fmt_amount(op.get('source_amount'))} "
            f"{asset_label(op, 'source_')} received {fmt_amount(op.get('amount'))} {asset_label(op)}"
        )
    elif kind == "change_trust":
        action = "removed trustline" if str(op.get("limit")) in ("0", "0.0000000") else "trustline"
        summary["text"] = f"{short(op.get('trustor'))} {action} {asset_label(op)} limit {fmt_amount(op.get('limit'))}"
    elif kind in ("manage_sell_offer", "manage_buy_offer", "create_passive_sell_offer"):
        summary["text"] = (
            f"{fmt_amount(op.get('amount'))} {asset_label(op, 'selling_')} for {asset_label(op, 'buying_')} "
            f"@ {op.get('price')} (offer {op.get('offer_id')})"
        )
    elif kind == "invoke_host_function":
        function = _host_function_name(op)
        params = [decode_scval_b64(p.get("value", "")) for p in op.get("parameters") or []]
        summary["function"] = function
        if function == "InvokeContract" and len(params) >= 2:
            summary["contract"] = params[0]
            summary["method"] = params[1]
            summary["args"] = params[2:]
            summary["text"] = f"{short(params[0], 4, 4)}.{params[1]}({', '.join(short(a, 8, 6) if isinstance(a, str) else str(a) for a in params[2:])})"
        else:
            summary["args"] = params
            summary["text"] = function
        changes = []
        for change in op.get("asset_balance_changes") or []:
            changes.append(
                f"{change.get('type')} {fmt_amount(change.get('amount'))} {asset_label(change)} "
                f"{short(change.get('from'))} -> {short(change.get('to'))}"
            )
        if changes:
            summary["balance_changes"] = changes
    elif kind == "extend_footprint_ttl":
        summary["text"] = f"extend TTL to +{op.get('extend_to')} ledgers"
    elif kind == "manage_data":
        summary["text"] = f"data {op.get('name')!r} = {op.get('value')!r}"
    elif kind == "account_merge":
        summary["text"] = f"{short(op.get('account'))} merged into {short(op.get('into'))}"
    elif kind == "set_options":
        changed = {k: v for k, v in op.items() if k in (
            "home_domain", "inflation_dest", "master_key_weight", "low_threshold", "med_threshold",
            "high_threshold", "signer_key", "signer_weight", "set_flags_s", "clear_flags_s") and v not in (None, [], "")}
        summary["text"] = "set_options " + json.dumps(changed, default=str)
    else:
        skip = {"_links", "id", "paging_token", "type", "type_i", "transaction_hash", "transaction_successful",
                "source_account", "created_at", "sponsor"}
        summary["text"] = kind + " " + json.dumps({k: v for k, v in op.items() if k not in skip}, default=str)[:160]
    return summary


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def _ledger_seconds(health: Dict[str, Any]) -> float:
    try:
        ledgers = int(health["latestLedger"]) - int(health["oldestLedger"])
        seconds = int(health["latestLedgerCloseTime"]) - int(health["oldestLedgerCloseTime"])
        if ledgers > 0 and seconds > 0:
            return seconds / ledgers
    except (KeyError, TypeError, ValueError):
        pass
    return 5.0


def _ttl_summary(live_until: Optional[int], latest: int, seconds_per_ledger: float) -> Dict[str, Any]:
    if live_until is None:
        return {"live_until_ledger": None}
    remaining = int(live_until) - int(latest)
    return {
        "live_until_ledger": int(live_until),
        "ledgers_remaining": remaining,
        "days_remaining": round(remaining * seconds_per_ledger / 86400, 1),
        "expired": remaining <= 0,
    }


def run_stats(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = resolve_config(args.network)
    root = horizon_get(cfg, "/")
    fees = horizon_get(cfg, "/fee_stats")
    health: Dict[str, Any] = {}
    rpc_error = None
    try:
        health = rpc_call(cfg, "getHealth") or {}
    except ClientError as exc:
        rpc_error = str(exc)
    pricer = Pricer(cfg)
    xlm_usd = pricer.xlm_usd()
    charged = fees.get("fee_charged") or {}
    return {
        "network": cfg["name"],
        "passphrase": root.get("network_passphrase"),
        "protocol_version": root.get("current_protocol_version"),
        "latest_ledger": root.get("history_latest_ledger"),
        "horizon_version": str(root.get("horizon_version") or "").split("-")[0],
        "core_version": root.get("core_version"),
        "base_fee_stroops": fees.get("last_ledger_base_fee"),
        "ledger_capacity_usage": fees.get("ledger_capacity_usage"),
        "fee_charged_p50_stroops": charged.get("p50"),
        "fee_charged_p99_stroops": charged.get("p99"),
        "fee_charged_max_stroops": charged.get("max"),
        "rpc_url": cfg["rpc"],
        "rpc_status": health.get("status"),
        "rpc_latest_ledger": health.get("latestLedger"),
        "rpc_oldest_ledger": health.get("oldestLedger"),
        "rpc_retention_ledgers": health.get("ledgerRetentionWindow"),
        "seconds_per_ledger": round(_ledger_seconds(health), 2) if health else None,
        "rpc_error": rpc_error,
        "xlm_usd": xlm_usd,
        "xlm_usd_source": pricer.xlm_source,
        "horizon_url": cfg["horizon"],
    }


def render_stats(data: Dict[str, Any]) -> str:
    lines = [
        f"Stellar {data['network']}  protocol {data['protocol_version']}  ledger {data['latest_ledger']}",
        f"  Horizon {data['horizon_version']} @ {data['horizon_url']}",
        f"  Core    {data['core_version']}",
        f"  XLM/USD {fmt_usd(data['xlm_usd'])}  ({data['xlm_usd_source']})",
        f"  Fees    base {fmt_stroops(data['base_fee_stroops'])}, charged p50 {fmt_stroops(data['fee_charged_p50_stroops'])}, "
        f"p99 {fmt_stroops(data['fee_charged_p99_stroops'])}, max {fmt_stroops(data['fee_charged_max_stroops'])}",
        f"  Ledger capacity usage {data['ledger_capacity_usage']}",
    ]
    if data.get("rpc_status"):
        lines.append(
            f"  RPC     {data['rpc_status']} @ {data['rpc_url']}  ledgers {data['rpc_oldest_ledger']}..{data['rpc_latest_ledger']} "
            f"(retention {data['rpc_retention_ledgers']}, ~{data['seconds_per_ledger']}s/ledger)"
        )
    else:
        lines.append(f"  RPC     unavailable: {data.get('rpc_error')}")
    return "\n".join(lines)


def run_account(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = resolve_config(args.network)
    target = (args.address or "").strip()
    federation = None
    if "*" in target:
        federation = federation_lookup(target)
        target = federation["account_id"]
    account_id = require_account_id(target)
    try:
        acct = horizon_get(cfg, f"/accounts/{account_id}")
    except ClientError as exc:
        if exc.status == 404:
            raise ClientError(
                f"Account {account_id} does not exist on {cfg['name']}. "
                "Unfunded accounts are not on the ledger (it needs a create_account / first XLM deposit)."
            ) from None
        raise
    pricer = Pricer(cfg, enabled=not args.no_prices)
    xlm_usd = pricer.xlm_usd()
    native = next((b for b in acct.get("balances", []) if b.get("asset_type") == "native"), {})
    native_balance = float(native.get("balance") or 0)
    subentries = int(acct.get("subentry_count") or 0)
    sponsoring = int(acct.get("num_sponsoring") or 0)
    sponsored = int(acct.get("num_sponsored") or 0)
    min_balance = (2 + subentries + sponsoring - sponsored) * BASE_RESERVE_XLM
    spendable = max(native_balance - min_balance - float(native.get("selling_liabilities") or 0), 0.0)
    holdings: List[Dict[str, Any]] = []
    total_usd = native_balance * xlm_usd if xlm_usd else None
    priced = 0
    for balance in acct.get("balances", []):
        if balance.get("asset_type") == "native":
            continue
        entry: Dict[str, Any] = {
            "asset": asset_label(balance),
            "code": balance.get("asset_code") or ("LP" if balance.get("asset_type") == "liquidity_pool_shares" else "?"),
            "issuer": balance.get("asset_issuer"),
            "balance": balance.get("balance"),
            "limit": balance.get("limit"),
            "authorized": balance.get("is_authorized"),
            "price_usd": None,
            "value_usd": None,
        }
        amount = float(balance.get("balance") or 0)
        if pricer.enabled and balance.get("asset_issuer") and amount > 0 and priced < args.limit:
            priced += 1
            quote = pricer.asset_price(balance["asset_code"], balance["asset_issuer"])
            if quote and quote.get("price_usd") is not None:
                entry["price_usd"] = quote["price_usd"]
                entry["value_usd"] = amount * quote["price_usd"]
                if total_usd is not None:
                    total_usd += entry["value_usd"]
        holdings.append(entry)
    holdings.sort(key=lambda h: (h["value_usd"] is None, -(h["value_usd"] or 0), -float(h["balance"] or 0)))
    return {
        "network": cfg["name"],
        "account_id": account_id,
        "federation": federation,
        "xlm_balance": native_balance,
        "xlm_min_balance": min_balance,
        "xlm_spendable": spendable,
        "xlm_usd": xlm_usd,
        "xlm_value_usd": native_balance * xlm_usd if xlm_usd else None,
        "total_value_usd": total_usd,
        "trustlines": holdings,
        "subentry_count": subentries,
        "num_sponsoring": sponsoring,
        "num_sponsored": sponsored,
        "sequence": acct.get("sequence"),
        "last_modified_ledger": acct.get("last_modified_ledger"),
        "home_domain": acct.get("home_domain"),
        "thresholds": acct.get("thresholds"),
        "signers": acct.get("signers"),
        "flags": {k: v for k, v in (acct.get("flags") or {}).items() if v},
        "data_entries": sorted((acct.get("data") or {}).keys()),
        "explorer_url": f"{cfg['explorer']}/account/{account_id}",
    }


def render_account(data: Dict[str, Any]) -> str:
    lines = [f"Account {data['account_id']}  ({data['network']})"]
    if data.get("federation"):
        lines.append(f"  resolved from {data['federation']['address']}")
    lines.append(
        f"  XLM {fmt_amount(data['xlm_balance'])}  (spendable {fmt_amount(data['xlm_spendable'])}, "
        f"reserve {fmt_amount(data['xlm_min_balance'])})  {fmt_usd(data['xlm_value_usd'])}"
    )
    if data["trustlines"]:
        lines.append(f"  Trustlines ({len(data['trustlines'])}):")
        for h in data["trustlines"]:
            flag = "" if h["authorized"] in (None, True) else "  [not authorized]"
            value = f"  {fmt_usd(h['value_usd'])}" if h["value_usd"] is not None else ""
            lines.append(f"    {h['asset']:<22} {fmt_amount(h['balance']):>22}{value}{flag}")
    else:
        lines.append("  Trustlines: none")
    if data["total_value_usd"] is not None:
        lines.append(f"  Total value ~{fmt_usd(data['total_value_usd'])} (XLM/USD {fmt_usd(data['xlm_usd'])})")
    th = data.get("thresholds") or {}
    signer_text = ", ".join(f"{short(s.get('key'))}(w{s.get('weight')})" for s in data.get("signers") or [])
    lines.append(f"  Signers: {signer_text}  thresholds low/med/high {th.get('low_threshold')}/{th.get('med_threshold')}/{th.get('high_threshold')}")
    extras = []
    if data.get("home_domain"):
        extras.append(f"home_domain={data['home_domain']}")
    if data.get("flags"):
        extras.append("flags=" + ",".join(data["flags"]))
    if data.get("data_entries"):
        extras.append(f"data_entries={len(data['data_entries'])}")
    extras.append(f"subentries={data['subentry_count']}")
    if data["num_sponsoring"] or data["num_sponsored"]:
        extras.append(f"sponsoring={data['num_sponsoring']} sponsored={data['num_sponsored']}")
    lines.append("  " + "  ".join(extras))
    lines.append(f"  Sequence {data['sequence']}  last modified ledger {data['last_modified_ledger']}")
    lines.append(f"  {data['explorer_url']}")
    return "\n".join(lines)


def _require_tx_hash(value: str) -> str:
    text = (value or "").strip().lower()
    if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
        raise ClientError(f"{value!r} is not a transaction hash (64 hex chars)")
    return text


def run_tx(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = resolve_config(args.network)
    tx_hash = _require_tx_hash(args.hash)
    try:
        tx = horizon_get(cfg, f"/transactions/{tx_hash}")
    except ClientError as exc:
        if exc.status == 404:
            raise ClientError(f"Transaction {tx_hash} not found on {cfg['name']} Horizon") from None
        raise
    ops = _records(horizon_get(cfg, f"/transactions/{tx_hash}/operations", limit=200, include_failed="true"))
    fee_bump = tx.get("fee_bump_transaction") or {}
    inner = tx.get("inner_transaction") or {}
    return {
        "network": cfg["name"],
        "hash": tx.get("hash"),
        "successful": tx.get("successful"),
        "ledger": tx.get("ledger"),
        "created_at": tx.get("created_at"),
        "source_account": tx.get("source_account"),
        "fee_charged_stroops": tx.get("fee_charged"),
        "max_fee_stroops": tx.get("max_fee"),
        "fee_bump": {"fee_account": tx.get("fee_account"), "outer_hash": fee_bump.get("hash"), "inner_hash": inner.get("hash")} if fee_bump else None,
        "memo_type": tx.get("memo_type"),
        "memo": tx.get("memo"),
        "operation_count": tx.get("operation_count"),
        "operations": [summarize_operation(op) for op in ops],
        "result": decode_tx_result(tx.get("result_xdr")),
        "explorer_url": f"{cfg['explorer']}/tx/{tx_hash}",
    }


def render_tx(data: Dict[str, Any]) -> str:
    status = "SUCCESS" if data["successful"] else "FAILED"
    lines = [
        f"Transaction {data['hash']}  ({data['network']})  {status}",
        f"  Ledger {data['ledger']}  at {data['created_at']}",
        f"  Source {data['source_account']}",
        f"  Fee charged {fmt_stroops(data['fee_charged_stroops'])} (max {fmt_stroops(data['max_fee_stroops'])})",
    ]
    if data.get("fee_bump"):
        lines.append(f"  Fee bump: paid by {data['fee_bump']['fee_account']}  inner {short(data['fee_bump']['inner_hash'], 8, 8)}")
    if data.get("memo_type") and data["memo_type"] != "none":
        lines.append(f"  Memo ({data['memo_type']}): {data.get('memo')}")
    result = data.get("result") or {}
    final_code = result.get("inner_code") or result.get("code")
    if final_code and final_code != "txSUCCESS":
        detail = f" (inner {result['inner_code']})" if result.get("inner_code") else ""
        lines.append(f"  Result: {result['code']}{detail}")
    for op_result in result.get("operations") or []:
        if "SUCCESS" not in op_result:
            lines.append(f"  Result: {op_result}")
    lines.append(f"  Operations ({data['operation_count']}):")
    for index, op in enumerate(data["operations"], 1):
        lines.append(f"    {index}. [{op['type']}] {op.get('text', '')}")
        for change in op.get("balance_changes") or []:
            lines.append(f"         {change}")
    lines.append(f"  {data['explorer_url']}")
    return "\n".join(lines)


def run_activity(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = resolve_config(args.network)
    account_id = require_account_id(args.address)
    limit = max(1, min(int(args.limit), 200))
    try:
        ops = _records(horizon_get(cfg, f"/accounts/{account_id}/operations", order="desc", limit=limit, include_failed="true"))
    except ClientError as exc:
        if exc.status == 404:
            raise ClientError(f"Account {account_id} does not exist on {cfg['name']}") from None
        raise
    rows = []
    for op in ops:
        summary = summarize_operation(op)
        summary.update({
            "created_at": op.get("created_at"),
            "tx_hash": op.get("transaction_hash"),
            "successful": op.get("transaction_successful", True),
        })
        rows.append(summary)
    return {"network": cfg["name"], "account_id": account_id, "count": len(rows), "operations": rows}


def render_activity(data: Dict[str, Any]) -> str:
    lines = [f"Recent operations for {data['account_id']}  ({data['network']}, newest first, {data['count']} shown)"]
    for op in data["operations"]:
        flag = "" if op.get("successful", True) else "  [FAILED]"
        lines.append(f"  {op.get('created_at')}  {op['type']:<28} {op.get('text', '')}  tx {short(op.get('tx_hash'), 6, 4)}{flag}")
    if not data["operations"]:
        lines.append("  (no operations)")
    return "\n".join(lines)


def run_asset(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = resolve_config(args.network)
    code, issuer = parse_asset(args.asset, cfg)
    if issuer is None:
        raise ClientError("XLM is the native asset; use `stats` or `price XLM` instead")
    records = _records(horizon_get(cfg, "/assets", asset_code=code, asset_issuer=issuer))
    if not records:
        raise ClientError(f"Asset {code}:{issuer} not found on {cfg['name']} (no trustlines exist for it)")
    rec = records[0]
    toml_url = ((rec.get("_links") or {}).get("toml") or {}).get("href")
    org_name = None
    currency: Dict[str, Any] = {}
    if toml_url:
        try:
            toml_data = fetch_stellar_toml(normalize_domain(toml_url))
            org_name = (toml_data.get("DOCUMENTATION") or {}).get("ORG_NAME")
            for entry in toml_data.get("CURRENCIES") or []:
                if entry.get("code") == code and entry.get("issuer") == issuer:
                    currency = {k: entry.get(k) for k in ("name", "desc", "anchor_asset", "anchor_asset_type", "is_asset_anchored") if entry.get(k) is not None}
                    break
        except ClientError:
            pass
    pricer = Pricer(cfg, enabled=not args.no_prices)
    quote = pricer.asset_price(code, issuer)
    accounts = rec.get("accounts") or {}
    balances = rec.get("balances") or {}
    return {
        "network": cfg["name"],
        "code": code,
        "issuer": issuer,
        "sac_contract_id": rec.get("contract_id"),
        "holders_authorized": accounts.get("authorized"),
        "holders_unauthorized": accounts.get("unauthorized"),
        "holders_maintain_liabilities": accounts.get("authorized_to_maintain_liabilities"),
        "amount_authorized": balances.get("authorized"),
        "amount_in_contracts": rec.get("contracts_amount"),
        "num_contracts": rec.get("num_contracts"),
        "num_liquidity_pools": rec.get("num_liquidity_pools"),
        "amount_in_liquidity_pools": rec.get("liquidity_pools_amount"),
        "num_claimable_balances": rec.get("num_claimable_balances"),
        "flags": {k: v for k, v in (rec.get("flags") or {}).items() if v},
        "toml_url": toml_url,
        "org_name": org_name,
        "currency": currency,
        "price": quote,
        "xlm_usd": pricer.xlm_usd() if not args.no_prices else None,
        "explorer_url": f"{cfg['explorer']}/asset/{code}-{issuer}",
    }


def render_asset(data: Dict[str, Any]) -> str:
    name = data["currency"].get("name")
    lines = [f"Asset {data['code']}:{data['issuer']}  ({data['network']})" + (f"  {name}" if name else "")]
    if data.get("org_name"):
        lines.append(f"  Issuer org: {data['org_name']}  ({data['toml_url']})")
    if data["currency"].get("anchor_asset"):
        lines.append(f"  Anchored to {data['currency']['anchor_asset']} ({data['currency'].get('anchor_asset_type')})")
    lines.append(f"  Holders: {data['holders_authorized']:,} authorized" + (f", {data['holders_unauthorized']:,} unauthorized" if data.get("holders_unauthorized") else ""))
    lines.append(f"  Supply held in accounts: {fmt_amount(data['amount_authorized'])}  in contracts: {fmt_amount(data['amount_in_contracts'])} ({data['num_contracts']} contracts)")
    lines.append(f"  Liquidity pools: {data['num_liquidity_pools']} holding {fmt_amount(data['amount_in_liquidity_pools'])}  claimable balances: {data['num_claimable_balances']}")
    lines.append(f"  Issuer flags: {', '.join(data['flags']) or 'none'}")
    lines.append(f"  SAC contract id: {data['sac_contract_id']}")
    price = data.get("price")
    if price:
        spread = f", spread {price['spread_pct']:.2f}%" if price.get("spread_pct") is not None else ""
        lines.append(f"  DEX price: {fmt_amount(price['price_xlm'], 7)} XLM = {fmt_usd(price.get('price_usd'))}{spread}")
    lines.append(f"  {data['explorer_url']}")
    return "\n".join(lines)


def run_price(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = resolve_config(args.network)
    code, issuer = parse_asset(args.asset, cfg)
    pricer = Pricer(cfg)
    quote = pricer.asset_price(code, issuer)
    if quote is None:
        raise ClientError(f"No order book price for {code}{':' + issuer if issuer else ''} on {cfg['name']} (empty DEX book)")
    return {"network": cfg["name"], "code": code, "issuer": issuer, "xlm_usd": pricer.xlm_usd(), **quote}


def render_price(data: Dict[str, Any]) -> str:
    label = data["code"] if data["issuer"] is None else f"{data['code']}:{short(data['issuer'], 4, 4)}"
    line = f"{label}: {fmt_usd(data.get('price_usd'))}"
    if data["issuer"] is not None:
        line += f"  ({fmt_amount(data['price_xlm'], 7)} XLM"
        if data.get("spread_pct") is not None:
            line += f", spread {data['spread_pct']:.2f}%"
        line += ")"
    line += f"  source: {data.get('source')}  XLM/USD {fmt_usd(data.get('xlm_usd'))}  [{data['network']}]"
    return line


def run_contract(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = resolve_config(args.network)
    contract_id = require_contract_id(args.contract)
    result = rpc_call(cfg, "getLedgerEntries", {"keys": [contract_instance_key(contract_id)]}) or {}
    latest = int(result.get("latestLedger") or 0)
    entries = result.get("entries") or []
    health = rpc_call(cfg, "getHealth") or {}
    seconds_per_ledger = _ledger_seconds(health)
    data: Dict[str, Any] = {
        "network": cfg["name"],
        "contract_id": contract_id,
        "latest_ledger": latest,
        "exists": bool(entries),
        "explorer_url": f"{cfg['explorer']}/contract/{contract_id}",
    }
    if not entries:
        data["note"] = ("No live instance entry: the contract was never deployed on this network, "
                        "or its instance TTL expired and the entry was archived (restore_footprint brings it back).")
        return data
    entry = entries[0]
    decoded = decode_ledger_entry_data(entry["xdr"])
    instance = decoded.get("value") or {}
    executable = instance.get("executable") or {}
    data["executable"] = executable
    data["instance_last_modified_ledger"] = entry.get("lastModifiedLedgerSeq")
    data["instance_ttl"] = _ttl_summary(entry.get("liveUntilLedgerSeq"), latest, seconds_per_ledger)
    data["instance_storage"] = instance.get("storage")
    if executable.get("type") == "stellar_asset":
        metadata = (instance.get("storage") or {}).get("METADATA") if isinstance(instance.get("storage"), dict) else None
        if isinstance(metadata, dict):
            data["asset"] = {"name": metadata.get("name"), "symbol": metadata.get("symbol"), "decimals": metadata.get("decimal")}
    elif executable.get("wasm_hash"):
        code_result = rpc_call(cfg, "getLedgerEntries", {"keys": [contract_code_key(executable["wasm_hash"])]}) or {}
        code_entries = code_result.get("entries") or []
        if code_entries:
            code_entry = code_entries[0]
            code = decode_ledger_entry_data(code_entry["xdr"])
            data["wasm"] = {
                "code_bytes": code.get("code_bytes"),
                "cost_inputs": code.get("cost_inputs"),
                "last_modified_ledger": code_entry.get("lastModifiedLedgerSeq"),
                "ttl": _ttl_summary(code_entry.get("liveUntilLedgerSeq"), latest, seconds_per_ledger),
            }
        else:
            data["wasm"] = {"note": "WASM code entry not live (archived); contract calls will fail until restored"}
    return data


def _render_ttl(ttl: Dict[str, Any]) -> str:
    if not ttl or ttl.get("live_until_ledger") is None:
        return "no TTL info"
    if ttl.get("expired"):
        return f"EXPIRED at ledger {ttl['live_until_ledger']} (archived)"
    return f"live until ledger {ttl['live_until_ledger']} (~{ttl['days_remaining']} days, {ttl['ledgers_remaining']} ledgers)"


def render_contract(data: Dict[str, Any]) -> str:
    lines = [f"Contract {data['contract_id']}  ({data['network']}, latest ledger {data['latest_ledger']})"]
    if not data.get("exists"):
        lines.append(f"  {data.get('note')}")
        lines.append(f"  {data['explorer_url']}")
        return "\n".join(lines)
    executable = data.get("executable") or {}
    if executable.get("type") == "stellar_asset":
        asset = data.get("asset") or {}
        lines.append(f"  Type: Stellar Asset Contract (SAC) for {asset.get('name')}  symbol {asset.get('symbol')}  decimals {asset.get('decimals')}")
    else:
        lines.append(f"  Type: WASM contract  hash {executable.get('wasm_hash')}")
    lines.append(f"  Instance TTL: {_render_ttl(data.get('instance_ttl') or {})}  (last modified ledger {data.get('instance_last_modified_ledger')})")
    wasm = data.get("wasm")
    if wasm:
        if wasm.get("note"):
            lines.append(f"  Code: {wasm['note']}")
        else:
            cost = wasm.get("cost_inputs") or {}
            size = f"{wasm['code_bytes']:,} bytes" if wasm.get("code_bytes") is not None else "unknown size"
            funcs = f", {cost.get('n_functions')} functions, {cost.get('n_exports')} exports" if cost else ""
            lines.append(f"  Code: {size}{funcs}  TTL: {_render_ttl(wasm.get('ttl') or {})}")
    storage = data.get("instance_storage")
    if isinstance(storage, dict) and storage:
        lines.append(f"  Instance storage keys: {', '.join(str(k) for k in list(storage)[:12])}" + (" ..." if len(storage) > 12 else ""))
    elif isinstance(storage, list) and storage:
        lines.append(f"  Instance storage entries: {len(storage)} (see --json)")
    lines.append(f"  {data['explorer_url']}")
    return "\n".join(lines)


def run_events(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = resolve_config(args.network)
    contract_id = require_contract_id(args.contract)
    health = rpc_call(cfg, "getHealth") or {}
    latest = int(health.get("latestLedger") or 0)
    oldest = int(health.get("oldestLedger") or 1)
    window = max(1, int(args.ledgers))
    start = max(oldest, latest - window + 1)
    limit = max(1, min(int(args.limit), 1000))
    collected: List[Dict[str, Any]] = []
    cursor = None
    scanned = 0
    for _ in range(5):
        params: Dict[str, Any] = {"filters": [{"type": "contract", "contractIds": [contract_id]}], "pagination": {"limit": 1000}}
        if cursor:
            params["pagination"]["cursor"] = cursor
        else:
            params["startLedger"] = start
        result = rpc_call(cfg, "getEvents", params) or {}
        page = result.get("events") or []
        scanned += len(page)
        collected.extend(page)
        collected = collected[-limit:]
        cursor = result.get("cursor")
        if len(page) < 1000 or not cursor:
            break
    events = []
    for event in collected:
        events.append({
            "ledger": event.get("ledger"),
            "closed_at": event.get("ledgerClosedAt"),
            "tx_hash": event.get("txHash"),
            "in_successful_call": event.get("inSuccessfulContractCall"),
            "topics": [decode_scval_b64(t) for t in event.get("topic") or []],
            "value": decode_scval_b64(event.get("value") or ""),
        })
    return {
        "network": cfg["name"],
        "contract_id": contract_id,
        "start_ledger": start,
        "latest_ledger": latest,
        "events_scanned": scanned,
        "count": len(events),
        "events": events,
    }


def _compact(value: Any, width: int = 60) -> str:
    if isinstance(value, str):
        if len(value) == 56 and value[0] in "GCMLB" and value.isalnum():
            return short(value, 6, 4)
        if ":" in value and len(value.rsplit(":", 1)[-1]) == 56:
            code, issuer = value.rsplit(":", 1)
            return f"{code}:{short(issuer, 4, 4)}"
        text = value
    elif isinstance(value, list):
        text = "[" + ", ".join(_compact(item, 24) for item in value) + "]"
    else:
        text = json.dumps(value, default=str)
    return text if len(text) <= width else text[: width - 3] + "..."


def render_events(data: Dict[str, Any]) -> str:
    lines = [
        f"Events for {data['contract_id']}  ({data['network']}, ledgers {data['start_ledger']}..{data['latest_ledger']}, "
        f"{data['events_scanned']} scanned, last {data['count']} shown)"
    ]
    for event in data["events"]:
        topics = " ".join(_compact(t, 40) for t in event["topics"])
        flag = "" if event.get("in_successful_call", True) else " [failed call]"
        lines.append(f"  {event['closed_at']}  L{event['ledger']}  {topics}  => {_compact(event['value'], 50)}  tx {short(event['tx_hash'], 6, 4)}{flag}")
    if not data["events"]:
        lines.append("  (no events in this window; widen with --ledgers)")
    return "\n".join(lines)


def run_toml(args: argparse.Namespace) -> Dict[str, Any]:
    domain = normalize_domain(args.domain)
    toml_data = fetch_stellar_toml(domain)
    documentation = toml_data.get("DOCUMENTATION") or {}
    endpoints = {key: toml_data[key] for key in SEP_ENDPOINTS if toml_data.get(key)}
    currencies = []
    for entry in toml_data.get("CURRENCIES") or []:
        currencies.append({k: entry.get(k) for k in ("code", "issuer", "name", "anchor_asset", "is_asset_anchored", "status") if entry.get(k) is not None})
    return {
        "domain": domain,
        "url": f"https://{domain}/.well-known/stellar.toml",
        "version": toml_data.get("VERSION"),
        "network_passphrase": toml_data.get("NETWORK_PASSPHRASE"),
        "org_name": documentation.get("ORG_NAME"),
        "org_url": documentation.get("ORG_URL"),
        "accounts": toml_data.get("ACCOUNTS") or [],
        "signing_key": toml_data.get("SIGNING_KEY"),
        "endpoints": endpoints,
        "seps": sorted({SEP_ENDPOINTS[k] for k in endpoints}),
        "currencies": currencies,
        "validators": len(toml_data.get("VALIDATORS") or []),
        "principals": [p.get("name") for p in toml_data.get("PRINCIPALS") or [] if p.get("name")],
    }


def render_toml(data: Dict[str, Any]) -> str:
    lines = [f"stellar.toml for {data['domain']}" + (f"  version {data['version']}" if data.get("version") else "")]
    if data.get("org_name"):
        lines.append(f"  Org: {data['org_name']}  {data.get('org_url') or ''}".rstrip())
    if data.get("network_passphrase"):
        lines.append(f"  Network: {data['network_passphrase']}")
    if data.get("signing_key"):
        lines.append(f"  Signing key: {data['signing_key']}")
    lines.append(f"  Accounts ({len(data['accounts'])}): " + ", ".join(short(a, 6, 4) for a in data["accounts"][:8]) + (" ..." if len(data["accounts"]) > 8 else ""))
    if data["endpoints"]:
        lines.append("  Endpoints:")
        for key, url in data["endpoints"].items():
            lines.append(f"    {SEP_ENDPOINTS[key]:<32} {url}")
    else:
        lines.append("  Endpoints: none declared (no anchor services)")
    if data["currencies"]:
        lines.append(f"  Currencies ({len(data['currencies'])}):")
        for c in data["currencies"][:15]:
            anchor = f" anchored to {c['anchor_asset']}" if c.get("anchor_asset") else ""
            lines.append(f"    {c.get('code')}:{short(c.get('issuer'), 4, 4)}  {c.get('name') or ''}{anchor}")
    if data["validators"]:
        lines.append(f"  Validators declared: {data['validators']}")
    lines.append(f"  {data['url']}")
    return "\n".join(lines)


def run_fund(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = resolve_config(args.network)
    account_id = require_account_id(args.address)
    if not cfg.get("friendbot"):
        raise ClientError(f"Friendbot only exists on testnet; {cfg['name']} accounts need a real XLM deposit. Re-run with --network testnet.")
    url = cfg["friendbot"].rstrip("/") + "?" + urllib.parse.urlencode({"addr": account_id})
    try:
        result = _http_json(url, timeout=30, retries=1)
    except ClientError as exc:
        if exc.status == 400:
            raise ClientError(f"Friendbot refused {account_id}: it is probably already funded (friendbot funds each account once)") from None
        raise
    balance = None
    try:
        acct = horizon_get(cfg, f"/accounts/{account_id}")
        native = next((b for b in acct.get("balances", []) if b.get("asset_type") == "native"), {})
        balance = native.get("balance")
    except ClientError:
        pass
    return {"network": cfg["name"], "account_id": account_id, "tx_hash": result.get("hash") if isinstance(result, dict) else None, "xlm_balance": balance}


def render_fund(data: Dict[str, Any]) -> str:
    return (
        f"Funded {data['account_id']} on {data['network']}  tx {data.get('tx_hash')}\n"
        f"  XLM balance now {fmt_amount(data['xlm_balance']) if data.get('xlm_balance') else 'unknown'}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

COMMANDS = {
    "stats": (run_stats, render_stats),
    "account": (run_account, render_account),
    "tx": (run_tx, render_tx),
    "activity": (run_activity, render_activity),
    "asset": (run_asset, render_asset),
    "price": (run_price, render_price),
    "contract": (run_contract, render_contract),
    "events": (run_events, render_events),
    "toml": (run_toml, render_toml),
    "fund": (run_fund, render_fund),
}


def _common_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--network", choices=["mainnet", "testnet"], default=None, help="Defaults to STELLAR_NETWORK or mainnet")
    parser.add_argument("--json", action="store_true", help="Machine-readable output")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="stellar_client.py", description="Read-only Stellar queries (Horizon + Stellar RPC)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("stats", help="Network status, protocol, fees, XLM price")
    _common_flags(p)

    p = sub.add_parser("account", aliases=["wallet"], help="Balances, reserves, signers, trustlines (G... or name*domain)")
    p.add_argument("address")
    p.add_argument("--no-prices", action="store_true", help="Skip DEX price lookups")
    p.add_argument("--limit", type=int, default=20, help="Max trustlines to price (default 20)")
    _common_flags(p)

    p = sub.add_parser("tx", help="Transaction details, operations, result codes")
    p.add_argument("hash")
    _common_flags(p)

    p = sub.add_parser("activity", help="Recent operations for an account")
    p.add_argument("address")
    p.add_argument("--limit", type=int, default=15)
    _common_flags(p)

    p = sub.add_parser("asset", help="Classic asset stats, issuer info, SAC id, DEX price")
    p.add_argument("asset", help="CODE:ISSUER or a known code (USDC, EURC, ...)")
    p.add_argument("--no-prices", action="store_true")
    _common_flags(p)

    p = sub.add_parser("price", help="DEX mid price for XLM or an asset")
    p.add_argument("asset", help="XLM, CODE, or CODE:ISSUER")
    _common_flags(p)

    p = sub.add_parser("contract", help="Contract instance: executable, TTLs, storage")
    p.add_argument("contract")
    _common_flags(p)

    p = sub.add_parser("events", help="Recent contract events (decoded)")
    p.add_argument("contract")
    p.add_argument("--ledgers", type=int, default=200, help="Look-back window in ledgers (~5s each; default 200)")
    p.add_argument("--limit", type=int, default=20, help="Show the last N events (default 20)")
    _common_flags(p)

    p = sub.add_parser("toml", help="Fetch and summarize a domain's stellar.toml (SEP-1)")
    p.add_argument("domain")
    _common_flags(p)

    p = sub.add_parser("fund", help="Fund a testnet account via friendbot")
    p.add_argument("address")
    _common_flags(p)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    command = "account" if args.command == "wallet" else args.command
    run, render = COMMANDS[command]
    try:
        data = run(args)
    except ClientError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(data, indent=2, default=str))
    else:
        print(render(data))
    return 0


if __name__ == "__main__":
    sys.exit(main())
