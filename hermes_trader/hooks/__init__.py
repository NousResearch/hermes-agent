"""Hermes Agentic Trader integration hooks."""

from hermes_trader.hooks.pre_trade import intercept_mcp_tool_call

__all__ = ["intercept_mcp_tool_call"]