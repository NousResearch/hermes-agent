"""Local trader tools: direct Base quote + unsigned wallet bundle."""

from __future__ import annotations

from typing import Any

from tools.registry import tool_error, tool_result

from .policy import TraderPolicy
from .uniswap_v3 import (
    BASE_CHAIN_ID,
    QUOTER_V2,
    SWAP_ROUTER_02,
    UNISWAP_DEPLOYMENT_SOURCE_COMMIT,
    build_wallet_capabilities_request,
    build_wallet_send_calls,
    fetch_direct_quote,
    resolve_direct_quote,
)

_STR = {"type": "string"}
_ADDR = {
    "type": "string",
    "pattern": "^0x[0-9a-fA-F]{40}$",
}
_UINT_STRING = {
    "type": "string",
    "pattern": "^[0-9]+$",
}
_FEE = {
    "type": "integer",
    "enum": [100, 500, 3000, 10000],
}

TRADER_UNISWAP_QUOTE_SCHEMA = {
    "name": "trader_uniswap_quote",
    "description": (
        "Read-only Base mainnet Uniswap V3 single-pool quote. Uses eth_call "
        "against the official QuoterV2 deployment and caches the result briefly. "
        "Never signs or submits a transaction."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "token_in": _ADDR,
            "token_out": _ADDR,
            "amount_in": {
                **_UINT_STRING,
                "description": "Input token amount in base units.",
            },
            "fee": _FEE,
        },
        "required": ["token_in", "token_out", "amount_in", "fee"],
        "additionalProperties": False,
    },
}

TRADER_UNISWAP_BUILD_CALLS_SCHEMA = {
    "name": "trader_uniswap_build_calls",
    "description": (
        "Build an unsigned EIP-5792 wallet_sendCalls request from a fresh "
        "trader_uniswap_quote result. Produces an exact-amount ERC-20 approval "
        "plus Uniswap SwapRouter02 exactInputSingle call. Does not contact a "
        "wallet and cannot submit a transaction."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "quote_id": _STR,
            "owner": _ADDR,
            "slippage_bps": {
                "type": "integer",
                "minimum": 1,
                "maximum": 1000,
            },
        },
        "required": ["quote_id", "owner", "slippage_bps"],
        "additionalProperties": False,
    },
}


def _trader_mapping() -> dict[str, Any]:
    try:
        from hermes_cli.config import load_config

        config = load_config() or {}
    except Exception:
        return {}
    if not isinstance(config, dict):
        return {}
    trader = config.get("trader") or {}
    return trader if isinstance(trader, dict) else {}


def _quote_scope(kwargs: dict[str, Any]) -> str:
    """Use Hermes observability scope when supplied; random quote_id is fallback authority."""
    for key in ("session_id", "task_id", "turn_id", "api_request_id"):
        value = str(kwargs.get(key) or "").strip()
        if value:
            return f"{key}:{value}"
    return "unscoped"


def handle_uniswap_quote(args: dict, **kwargs) -> str:
    try:
        record = fetch_direct_quote(
            token_in=str(args.get("token_in") or ""),
            token_out=str(args.get("token_out") or ""),
            amount_in=args.get("amount_in"),
            fee=args.get("fee"),
            session_id=_quote_scope(kwargs),
        )
        return tool_result(
            {
                "success": True,
                "chain_id": BASE_CHAIN_ID,
                "quote_id": record.quote_id,
                "token_in": record.token_in,
                "token_out": record.token_out,
                "amount_in": str(record.amount_in),
                "amount_out": str(record.amount_out),
                "fee": record.fee,
                "sqrt_price_x96_after": str(record.sqrt_price_x96_after),
                "initialized_ticks_crossed": record.initialized_ticks_crossed,
                "gas_estimate": str(record.gas_estimate),
                "quoter": QUOTER_V2,
                "source_commit": UNISWAP_DEPLOYMENT_SOURCE_COMMIT,
                "read_only": True,
                "expires_in_seconds": 120,
            }
        )
    except Exception as exc:
        return tool_error(
            f"Direct Base Uniswap quote failed: {type(exc).__name__}: {exc}"
        )


def handle_uniswap_build_calls(args: dict, **kwargs) -> str:
    try:
        quote = resolve_direct_quote(
            str(args.get("quote_id") or ""),
            session_id=_quote_scope(kwargs),
        )
        if quote is None:
            return tool_error(
                "quote_id is missing, expired, or invalid for this Hermes execution scope"
            )

        policy = TraderPolicy.from_mapping(_trader_mapping())
        slippage = int(args.get("slippage_bps") or 0)
        if slippage <= 0:
            return tool_error("slippage_bps must be positive")
        if slippage > policy.max_slippage_bps:
            return tool_error(
                f"slippage_bps {slippage} exceeds trader policy maximum "
                f"{policy.max_slippage_bps}"
            )

        owner = str(args.get("owner") or "")
        payload = build_wallet_send_calls(
            owner=owner,
            quote=quote,
            slippage_bps=slippage,
        )
        return tool_result(
            {
                "success": True,
                "chain_id": BASE_CHAIN_ID,
                "capability_preflight": build_wallet_capabilities_request(owner),
                "wallet_request": payload,
                "router": SWAP_ROUTER_02,
                "unsigned": True,
                "submitted": False,
                "requires_atomic_wallet": True,
                "warning": (
                    "Review the wallet simulation before signing. Hermes has "
                    "not contacted a wallet or submitted this request."
                ),
            }
        )
    except Exception as exc:
        return tool_error(
            f"Unsigned Base Uniswap call build failed: {type(exc).__name__}: {exc}"
        )
