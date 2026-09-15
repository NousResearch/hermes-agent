"""Hermes Agentic Trader safety boundary and local planning tools.

The repair track is paper-safe by default. Aggregator-backed MCP write paths
remain fail-closed. The only transaction-oriented tool exposed by this plugin
builds an unsigned EIP-5792 request from a direct on-chain Uniswap quote; it
never receives a private key and never contacts a wallet.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Optional

from . import tools as _tools
from .mandate import load_mandate
from .policy import TraderPolicy
from .quote_cache import capture_quote, resolve_execution_quote
from .risk import evaluate_bound_quote

LIVE_WRITE_TOOLS = frozenset({"execute_swap", "submit_gasless_swap"})
QUOTE_TOOLS = frozenset({"get_swap_quote", "get_gasless_quote"})
_TRUTHY = frozenset({"1", "true", "yes", "on"})


def _load_trader_config() -> Mapping[str, Any]:
    """Return the trader config mapping; malformed/unavailable config fails safe."""
    try:
        from hermes_cli.config import load_config

        config = load_config() or {}
    except Exception:
        return {}
    if not isinstance(config, dict):
        return {}
    trader = config.get("trader") or {}
    return trader if isinstance(trader, dict) else {}


def _kill_switch_active() -> bool:
    """Honor both the environment and the on-disk emergency stop."""
    if os.environ.get("HERMES_TRADER_KILL_SWITCH", "").strip().lower() in _TRUTHY:
        return True
    try:
        from hermes_constants import get_hermes_home

        return (Path(get_hermes_home()) / "trader" / "KILL_SWITCH").is_file()
    except Exception:
        return False


def _block(message: str) -> dict[str, str]:
    return {"action": "block", "message": message}


def _on_pre_tool_call(
    tool_name: str = "",
    args: Optional[dict[str, Any]] = None,
    session_id: Optional[str] = None,
    **_: Any,
) -> Optional[dict[str, str]]:
    """Fail closed for legacy aggregator-backed live swap tools."""
    if tool_name not in LIVE_WRITE_TOOLS:
        return None

    if _kill_switch_active():
        return _block(
            "Hermes Agentic Trader blocked this write: the trader kill switch is active."
        )

    trader = _load_trader_config()
    mode = str(trader.get("mode", "paper")).strip().lower()

    if mode != "live":
        return _block(
            "Hermes Agentic Trader is in paper mode. Live MCP write tools are blocked."
        )

    bound_quote = resolve_execution_quote(
        tool_name=tool_name,
        args=args or {},
        session_id=session_id,
        consume=False,
    )
    if bound_quote is None:
        return _block(
            "Hermes Agentic Trader blocked this write because quoteData was not "
            "observed from the matching quote tool in this Hermes session, or the "
            "quote expired."
        )

    try:
        policy = TraderPolicy.from_mapping(trader)
    except (TypeError, ValueError) as exc:
        return _block(f"Hermes Agentic Trader policy is invalid: {exc}")

    try:
        mandate = load_mandate()
    except (OSError, ValueError, TypeError) as exc:
        return _block(f"Hermes Agentic Trader mandate is invalid: {exc}")

    decision = evaluate_bound_quote(
        policy=policy,
        record=bound_quote,
        mandate=mandate,
        notional_usd=None,
        portfolio_value_usd=None,
        daily_loss_pct=None,
        pool_liquidity_usd=None,
    )
    reason = decision.reason.value if decision.reason is not None else "UNEXPECTED_APPROVAL"
    return _block(
        "Hermes Agentic Trader legacy MCP execution remains quarantined on "
        f"the PR #60159 repair branch. Gate {reason}: {decision.message}"
    )


def _on_post_tool_call(
    tool_name: str = "",
    args: Optional[dict[str, Any]] = None,
    result: Any = None,
    status: Optional[str] = None,
    session_id: Optional[str] = None,
    **_: Any,
) -> None:
    """Remember legacy MCP quote responses only for defense-in-depth gating."""
    if tool_name not in QUOTE_TOOLS:
        return None
    if status not in (None, "ok"):
        return None
    capture_quote(
        tool_name=tool_name,
        args=args or {},
        result=result,
        session_id=session_id,
    )
    return None


def register(ctx) -> None:
    """Register local planning tools plus legacy-MCP safety hooks."""
    ctx.register_tool(
        name="trader_uniswap_quote",
        toolset="trader",
        schema=_tools.TRADER_UNISWAP_QUOTE_SCHEMA,
        handler=_tools.handle_uniswap_quote,
        emoji="📈",
    )
    ctx.register_tool(
        name="trader_uniswap_build_calls",
        toolset="trader",
        schema=_tools.TRADER_UNISWAP_BUILD_CALLS_SCHEMA,
        handler=_tools.handle_uniswap_build_calls,
        emoji="🧾",
    )
    ctx.register_hook("pre_tool_call", _on_pre_tool_call)
    ctx.register_hook("post_tool_call", _on_post_tool_call)
