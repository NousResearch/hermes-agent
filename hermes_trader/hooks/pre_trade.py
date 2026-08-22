"""Synchronous MCP write-tool gate at the Hermes transport layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from hermes_trader.audit.rate_limit import reserve_write_rate_limit
from hermes_trader.config import load_trader_config
from hermes_trader.risk.gate import RiskGate, TradeIntent
from hermes_trader.tools import LIVE_WRITE_TOOLS


@dataclass(frozen=True)
class PreTradeResult:
    error: Optional[str] = None
    reservation_id: Optional[str] = None


def _intent_from_mcp_args(args: dict[str, Any]) -> TradeIntent:
    return TradeIntent.from_mapping(
        {
            "action": args.get("action", "buy"),
            "chain": args.get("chain") or "",
            "token_address": args.get("token_address") or "",
            "size_usd": args.get("amount_usd", args.get("size_usd")),
            "confidence": args.get("confidence", 1.0),
            "pool_liquidity_usd": args.get("pool_liquidity_usd"),
            "slippage_bps": args.get("slippage_bps"),
            "reasoning": args.get("reasoning", "direct MCP write"),
            "strategy_tag": args.get("strategy_tag"),
        }
    )


def prepare_mcp_tool_call(
    server_name: str, tool_name: str, args: dict[str, Any] | None = None
) -> PreTradeResult:
    """Risk-check and atomically reserve a live write before MCP dispatch."""
    if tool_name not in LIVE_WRITE_TOOLS:
        return PreTradeResult()
    cfg = load_trader_config()
    if server_name != cfg.mcp_server_name:
        return PreTradeResult()

    try:
        intent = _intent_from_mcp_args(args or {})
    except (TypeError, ValueError) as exc:
        return PreTradeResult(error=f"{tool_name} blocked: invalid order arguments ({exc})")

    decision = RiskGate(config=cfg).evaluate(intent)
    if not decision.approved:
        return PreTradeResult(error=f"{tool_name} blocked: {decision.message}")

    reservation_id, rate_msg = reserve_write_rate_limit(
        max_per_hour=cfg.max_write_tools_per_hour
    )
    if reservation_id is None:
        return PreTradeResult(error=f"{tool_name} blocked: {rate_msg}")
    return PreTradeResult(reservation_id=reservation_id)


def intercept_mcp_tool_call(
    server_name: str, tool_name: str, args: dict[str, Any] | None = None
) -> Optional[str]:
    """Compatibility wrapper used by callers that only need a gate decision."""
    result = prepare_mcp_tool_call(server_name, tool_name, args)
    if result.reservation_id is not None:
        from hermes_trader.audit.rate_limit import WriteToolRateLimiter

        cfg = load_trader_config()
        WriteToolRateLimiter(max_per_hour=cfg.max_write_tools_per_hour).reconcile(
            result.reservation_id, succeeded=False
        )
    return result.error
