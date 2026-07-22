"""Layer 4 — execute approved orders via MCP write tools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from hermes_trader.config import TraderConfig
from hermes_trader.risk.gate import OrderRequest
from hermes_trader.tools import LIVE_WRITE_TOOLS

McpCallFn = Callable[[str, str, dict[str, Any]], Any]


@dataclass(frozen=True)
class ExecutionResult:
    status: str
    tool: Optional[str] = None
    message: str = ""
    payload: Optional[Any] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "tool": self.tool,
            "message": self.message,
            "payload": self.payload,
        }


def build_swap_tool_args(order: OrderRequest) -> dict[str, Any]:
    """Map OrderRequest to defi-trading-mcp swap tool arguments."""
    return {
        "chain": order.chain,
        "token_address": order.token_address,
        "amount_usd": order.size_usd,
        "slippage_bps": order.max_slippage_bps,
        "action": order.action,
    }


class OrderExecutor:
    """Execute gate-approved orders; never bypasses paper mode or write allowlist."""

    def __init__(self, config: TraderConfig, mcp_call: McpCallFn):
        self.config = config
        self.mcp_call = mcp_call

    def execute(self, order: OrderRequest) -> ExecutionResult:
        if self.config.mode == "paper":
            return ExecutionResult(
                status="skipped",
                tool=order.tool,
                message="Paper mode — execution not submitted",
            )

        if order.tool not in LIVE_WRITE_TOOLS:
            return ExecutionResult(
                status="error",
                tool=order.tool,
                message=f"Unknown write tool {order.tool!r}",
            )

        args = build_swap_tool_args(order)
        payload = self.mcp_call(self.config.mcp_server_name, order.tool, args)
        return ExecutionResult(
            status="submitted",
            tool=order.tool,
            message="Order submitted via MCP",
            payload=payload,
        )