"""MCP tool allowlists for Hermes Agentic Trader paper vs live modes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, Iterable, Literal, Sequence

TraderMode = Literal["paper", "live"]

# defi-trading-mcp read-only / quote tools safe for paper mode.
PAPER_MODE_READ_TOOLS: FrozenSet[str] = frozenset(
    {
        "get_portfolio_tokens",
        "get_portfolio_balances",
        "get_portfolio_transactions",
        "get_trending_pools",
        "get_new_pools",
        "get_pool_ohlcv",
        "get_pool_trades",
        "get_token_price",
        "get_token_data",
        "get_token_info",
        "search_pools",
        "get_swap_price",
        "get_swap_quote",
        "get_supported_chains",
        "get_gasless_price",
        "get_gasless_quote",
        "get_gasless_status",
        "convert_wei_to_formatted",
        "convert_formatted_to_wei",
    }
)

# Write tools that move funds on-chain — live mode only (P1 adds mandate gate).
LIVE_WRITE_TOOLS: FrozenSet[str] = frozenset(
    {
        "execute_swap",
        "submit_gasless_swap",
    }
)

ALL_KNOWN_DEFI_TRADING_TOOLS: FrozenSet[str] = PAPER_MODE_READ_TOOLS | LIVE_WRITE_TOOLS


@dataclass(frozen=True)
class ToolPolicy:
    mode: TraderMode
    allowed_tools: FrozenSet[str]
    blocked_write_tools: FrozenSet[str]

    def is_tool_allowed(self, tool_name: str) -> bool:
        return tool_name in self.allowed_tools

    def mcp_tools_include(self) -> list[str]:
        """Hermes MCP server ``tools.include`` whitelist for catalog install."""
        return sorted(self.allowed_tools)


def resolve_tool_policy(mode: TraderMode) -> ToolPolicy:
    if mode == "paper":
        return ToolPolicy(
            mode="paper",
            allowed_tools=PAPER_MODE_READ_TOOLS,
            blocked_write_tools=LIVE_WRITE_TOOLS,
        )
    return ToolPolicy(
        mode="live",
        allowed_tools=ALL_KNOWN_DEFI_TRADING_TOOLS,
        blocked_write_tools=frozenset(),
    )


def paper_mode_mcp_tools_include() -> list[str]:
    """Default ``tools.include`` list for defi-trading MCP in paper mode."""
    return resolve_tool_policy("paper").mcp_tools_include()


def assert_no_live_tools(requested: Iterable[str], mode: TraderMode) -> None:
    """Raise if paper mode would invoke a live write tool."""
    if mode != "paper":
        return
    blocked = set(requested) & LIVE_WRITE_TOOLS
    if blocked:
        names = ", ".join(sorted(blocked))
        raise PermissionError(
            f"Paper mode blocks live write tools: {names}. "
            "Set mode: live and sign mandate.json (P1) to enable execution."
        )


def filter_tool_names(tool_names: Sequence[str], mode: TraderMode) -> list[str]:
    policy = resolve_tool_policy(mode)
    return [name for name in tool_names if policy.is_tool_allowed(name)]