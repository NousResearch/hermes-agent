"""Tests for hermes_trader.tools paper vs live tool policy."""

from __future__ import annotations

import pytest

from hermes_trader.tools import (
    LIVE_WRITE_TOOLS,
    PAPER_MODE_READ_TOOLS,
    assert_no_live_tools,
    filter_tool_names,
    paper_mode_mcp_tools_include,
    resolve_tool_policy,
)


def test_paper_policy_excludes_write_tools():
    policy = resolve_tool_policy("paper")
    assert policy.mode == "paper"
    assert policy.blocked_write_tools == LIVE_WRITE_TOOLS
    assert not policy.is_tool_allowed("execute_swap")
    assert not policy.is_tool_allowed("submit_gasless_swap")
    assert policy.is_tool_allowed("get_swap_quote")


def test_live_policy_includes_write_tools():
    policy = resolve_tool_policy("live")
    assert policy.mode == "live"
    assert policy.is_tool_allowed("execute_swap")
    assert policy.is_tool_allowed("get_trending_pools")


def test_assert_no_live_tools_raises_in_paper_mode():
    with pytest.raises(PermissionError, match="execute_swap"):
        assert_no_live_tools(["get_swap_quote", "execute_swap"], "paper")


def test_assert_no_live_tools_allows_live_mode():
    assert_no_live_tools(["execute_swap"], "live")


def test_filter_tool_names_paper_mode():
    names = [
        "get_portfolio_tokens",
        "execute_swap",
        "get_trending_pools",
        "submit_gasless_swap",
    ]
    assert filter_tool_names(names, "paper") == [
        "get_portfolio_tokens",
        "get_trending_pools",
    ]


def test_paper_mode_mcp_tools_include_sorted_and_complete():
    include = paper_mode_mcp_tools_include()
    assert include == sorted(PAPER_MODE_READ_TOOLS)
    assert "execute_swap" not in include
    assert "submit_gasless_swap" not in include