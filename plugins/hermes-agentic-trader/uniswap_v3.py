"""Direct, local-construction Uniswap V3 support for Base.

No private key is accepted. Quotes are obtained with eth_call against the
official Base QuoterV2 deployment. Execution output is an unsigned EIP-5792
wallet_sendCalls request whose calldata is constructed locally from a cached,
same-session quote.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import threading
import time
import urllib.request
from dataclasses import dataclass
from typing import Any, Optional

BASE_CHAIN_ID = 8453
BASE_CHAIN_HEX = "0x2105"
DEFAULT_BASE_RPC = "https://mainnet.base.org"

UNISWAP_DEPLOYMENT_SOURCE = "Uniswap/contracts"
UNISWAP_DEPLOYMENT_SOURCE_COMMIT = "e34ba78b05663c8c342cce69b0039067ace75691"
SWAP_ROUTER_02 = "0x2626664c2603336e57b271c5c0b26f421741e481"
QUOTER_V2 = "0x3d4e44eb1374240ce5f1b871ab261cd16335b76a"

APPROVE_SELECTOR = "095ea7b3"
EXACT_INPUT_SINGLE_SELECTOR = "04e45aaf"
QUOTE_EXACT_INPUT_SINGLE_SELECTOR = "c6a5026a"

SUPPORTED_FEES = frozenset({100, 500, 3000, 10000})
QUOTE_TTL_SECONDS = 120.0
MAX_LOCAL_QUOTES = 64


@dataclass(frozen=True)
class DirectQuote:
    quote_id: str
    session_id: str
    token_in: str
    token_out: str
    amount_in: int
    fee: int
    amount_out: int
    sqrt_price_x96_after: int
    initialized_ticks_crossed: int
    gas_estimate: int
    captured_at: float


_quote_lock = threading.RLock()
_quotes: dict[str, DirectQuote] = {}


def _normalize_address(value: str) -> str:
    text = str(value or "").strip().lower()
    if len(text) != 42 or not text.startswith("0x"):
        raise ValueError("expected a 20-byte EVM address")
    try:
        int(text[2:], 16)
    except ValueError as exc:
        raise ValueError("expected a hexadecimal EVM address") from exc
    if int(text[2:], 16) == 0:
        raise ValueError("zero address is not supported")
    return text


def _uint(value: Any, *, bits: int = 256, label: str = "value") -> int:
    try:
        number = int(str(value), 0) if isinstance(value, str) and value.startswith(("0x", "0X")) else int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if number < 0 or number >= 1 << bits:
        raise ValueError(f"{label} is outside uint{bits}")
    return number


def _word_uint(value: Any, *, bits: int = 256, label: str = "value") -> str:
    return f"{_uint(value, bits=bits, label=label):064x}"


def _word_address(value: str) -> str:
    return _normalize_address(value)[2:].rjust(64, "0")


def _validate_pair(token_in: str, token_out: str) -> tuple[str, str]:
    left = _normalize_address(token_in)
    right = _normalize_address(token_out)
    if left == right:
        raise ValueError("token_in and token_out must differ")
    return left, right


def _validate_fee(fee: Any) -> int:
    value = _uint(fee, bits=24, label="fee")
    if value not in SUPPORTED_FEES:
        raise ValueError("fee must be one of 100, 500, 3000, or 10000")
    return value


def quote_calldata(
    *,
    token_in: str,
    token_out: str,
    amount_in: Any,
    fee: Any,
    sqrt_price_limit_x96: Any = 0,
) -> str:
    token_in, token_out = _validate_pair(token_in, token_out)
    amount = _uint(amount_in, label="amount_in")
    if amount <= 0:
        raise ValueError("amount_in must be positive")
    fee_value = _validate_fee(fee)
    sqrt_limit = _uint(
        sqrt_price_limit_x96,
        bits=160,
        label="sqrt_price_limit_x96",
    )
    words = [
        _word_address(token_in),
        _word_address(token_out),
        _word_uint(amount, label="amount_in"),
        _word_uint(fee_value, bits=24, label="fee"),
        _word_uint(sqrt_limit, bits=160, label="sqrt_price_limit_x96"),
    ]
    return "0x" + QUOTE_EXACT_INPUT_SINGLE_SELECTOR + "".join(words)


def build_quote_eth_call(
    *,
    token_in: str,
    token_out: str,
    amount_in: Any,
    fee: Any,
) -> dict[str, Any]:
    return {
        "method": "eth_call",
        "params": [
            {
                "to": QUOTER_V2,
                "data": quote_calldata(
                    token_in=token_in,
                    token_out=token_out,
                    amount_in=amount_in,
                    fee=fee,
                ),
            },
            "latest",
        ],
    }


def decode_quote_result(raw: str) -> dict[str, int]:
    text = str(raw or "")
    if not text.startswith("0x"):
        raise ValueError("quote result must be 0x-prefixed")
    payload = text[2:]
    if len(payload) < 64 * 4:
        raise ValueError("quote result is shorter than four ABI words")
    try:
        words = [int(payload[i : i + 64], 16) for i in range(0, 64 * 4, 64)]
    except ValueError as exc:
        raise ValueError("quote result is not valid hex") from exc
    amount_out, sqrt_after, ticks_crossed, gas_estimate = words
    return {
        "amount_out": amount_out,
        "sqrt_price_x96_after": sqrt_after,
        "initialized_ticks_crossed": ticks_crossed,
        "gas_estimate": gas_estimate,
    }


def _rpc_url() -> str:
    return os.environ.get("EVM_RPC_URL", "").strip() or DEFAULT_BASE_RPC


def rpc_call(method: str, params: list[Any], *, timeout: float = 15.0) -> Any:
    payload = json.dumps(
        {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
    ).encode("utf-8")
    request = urllib.request.Request(
        _rpc_url(),
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    if not isinstance(body, dict):
        raise RuntimeError("RPC response must be an object")
    if "error" in body:
        raise RuntimeError(f"RPC error: {body['error']}")
    return body.get("result")


def fetch_direct_quote(
    *,
    token_in: str,
    token_out: str,
    amount_in: Any,
    fee: Any,
    session_id: Optional[str],
    now: Optional[float] = None,
) -> DirectQuote:
    if not session_id:
        raise ValueError("session_id is required for quote binding")

    chain_id = rpc_call("eth_chainId", [])
    if str(chain_id).lower() != BASE_CHAIN_HEX:
        raise RuntimeError(
            f"configured EVM RPC is not Base mainnet: expected {BASE_CHAIN_HEX}, got {chain_id!r}"
        )

    call = build_quote_eth_call(
        token_in=token_in,
        token_out=token_out,
        amount_in=amount_in,
        fee=fee,
    )
    raw = rpc_call(call["method"], call["params"])
    decoded = decode_quote_result(str(raw))

    left, right = _validate_pair(token_in, token_out)
    amount = _uint(amount_in, label="amount_in")
    fee_value = _validate_fee(fee)
    timestamp = time.monotonic() if now is None else float(now)

    quote_material = {
        "session_id": str(session_id),
        "token_in": left,
        "token_out": right,
        "amount_in": amount,
        "fee": fee_value,
        "amount_out": decoded["amount_out"],
        "captured_at_ms": int(timestamp * 1000),
        "nonce": secrets.token_hex(16),
    }
    quote_id = hashlib.sha256(
        json.dumps(
            quote_material,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    record = DirectQuote(
        quote_id=quote_id,
        session_id=str(session_id),
        token_in=left,
        token_out=right,
        amount_in=amount,
        fee=fee_value,
        amount_out=decoded["amount_out"],
        sqrt_price_x96_after=decoded["sqrt_price_x96_after"],
        initialized_ticks_crossed=decoded["initialized_ticks_crossed"],
        gas_estimate=decoded["gas_estimate"],
        captured_at=timestamp,
    )
    with _quote_lock:
        _evict_quotes(timestamp)
        _quotes[quote_id] = record
        _evict_quotes(timestamp)
    return record


def _evict_quotes(now: float) -> None:
    expired = [
        quote_id
        for quote_id, record in _quotes.items()
        if now - record.captured_at > QUOTE_TTL_SECONDS
    ]
    for quote_id in expired:
        _quotes.pop(quote_id, None)

    overflow = len(_quotes) - MAX_LOCAL_QUOTES
    if overflow > 0:
        oldest = sorted(_quotes.values(), key=lambda item: item.captured_at)
        for record in oldest[:overflow]:
            _quotes.pop(record.quote_id, None)


def resolve_direct_quote(
    quote_id: str,
    *,
    session_id: Optional[str],
    now: Optional[float] = None,
) -> Optional[DirectQuote]:
    if not quote_id or not session_id:
        return None
    timestamp = time.monotonic() if now is None else float(now)
    with _quote_lock:
        _evict_quotes(timestamp)
        record = _quotes.get(str(quote_id))
        if record is None or record.session_id != str(session_id):
            return None
        return record


def clear_direct_quotes() -> None:
    with _quote_lock:
        _quotes.clear()


def amount_out_minimum(amount_out: Any, slippage_bps: Any) -> int:
    amount = _uint(amount_out, label="amount_out")
    bps = _uint(slippage_bps, bits=16, label="slippage_bps")
    if amount <= 0:
        raise ValueError("amount_out must be positive")
    if bps <= 0 or bps >= 10_000:
        raise ValueError("slippage_bps must be between 1 and 9999")
    minimum = amount * (10_000 - bps) // 10_000
    if minimum <= 0:
        raise ValueError("amount_out_minimum would be zero")
    return minimum


def approve_calldata(*, spender: str, amount: Any) -> str:
    amount_value = _uint(amount, label="amount")
    if amount_value <= 0:
        raise ValueError("approval amount must be positive")
    return (
        "0x"
        + APPROVE_SELECTOR
        + _word_address(spender)
        + _word_uint(amount_value, label="amount")
    )


def exact_input_single_calldata(
    *,
    token_in: str,
    token_out: str,
    fee: Any,
    recipient: str,
    amount_in: Any,
    amount_out_minimum_value: Any,
    sqrt_price_limit_x96: Any = 0,
) -> str:
    token_in, token_out = _validate_pair(token_in, token_out)
    recipient_value = _normalize_address(recipient)
    fee_value = _validate_fee(fee)
    amount = _uint(amount_in, label="amount_in")
    minimum = _uint(amount_out_minimum_value, label="amount_out_minimum")
    sqrt_limit = _uint(
        sqrt_price_limit_x96,
        bits=160,
        label="sqrt_price_limit_x96",
    )
    if amount <= 0 or minimum <= 0:
        raise ValueError("swap amounts must be positive")

    words = [
        _word_address(token_in),
        _word_address(token_out),
        _word_uint(fee_value, bits=24, label="fee"),
        _word_address(recipient_value),
        _word_uint(amount, label="amount_in"),
        _word_uint(minimum, label="amount_out_minimum"),
        _word_uint(sqrt_limit, bits=160, label="sqrt_price_limit_x96"),
    ]
    return "0x" + EXACT_INPUT_SINGLE_SELECTOR + "".join(words)


def build_wallet_capabilities_request(owner: str) -> dict[str, Any]:
    owner_value = _normalize_address(owner)
    return {
        "method": "wallet_getCapabilities",
        "params": [owner_value, [BASE_CHAIN_HEX]],
    }


def build_wallet_send_calls(
    *,
    owner: str,
    quote: DirectQuote,
    slippage_bps: Any,
) -> dict[str, Any]:
    owner_value = _normalize_address(owner)
    minimum = amount_out_minimum(quote.amount_out, slippage_bps)

    approval = {
        "to": quote.token_in,
        "value": "0x0",
        "data": approve_calldata(
            spender=SWAP_ROUTER_02,
            amount=quote.amount_in,
        ),
    }
    swap = {
        "to": SWAP_ROUTER_02,
        "value": "0x0",
        "data": exact_input_single_calldata(
            token_in=quote.token_in,
            token_out=quote.token_out,
            fee=quote.fee,
            recipient=owner_value,
            amount_in=quote.amount_in,
            amount_out_minimum_value=minimum,
        ),
    }

    return {
        "method": "wallet_sendCalls",
        "params": [
            {
                "version": "2.0.0",
                "from": owner_value,
                "chainId": BASE_CHAIN_HEX,
                "atomicRequired": True,
                "calls": [approval, swap],
            }
        ],
        "quote": {
            "quote_id": quote.quote_id,
            "amount_in": str(quote.amount_in),
            "amount_out": str(quote.amount_out),
            "amount_out_minimum": str(minimum),
            "fee": quote.fee,
            "quoter": QUOTER_V2,
            "router": SWAP_ROUTER_02,
            "source_commit": UNISWAP_DEPLOYMENT_SOURCE_COMMIT,
        },
    }
