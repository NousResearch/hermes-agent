"""Unit tests for agent.anthropic_adapter._apply_tool_search across modes.

Covers:
  * server_side mode — preserves legacy behavior (prepend server tool,
    stubs carry defer_loading=true).
  * client_side mode (default) — no server-tool prepend, stubs carry no
    defer_loading flag, promoted_tools bypass the stub.
  * Back-compat — missing mode key defaults to client_side; "off"/None
    config returns input unchanged.
  * Safety — "all deferred" or "all eager" returns input unchanged.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from agent.anthropic_adapter import _apply_tool_search


def _tool(name: str, description: str = "x", **extra) -> Dict[str, Any]:
    """Minimal tool dict — same shape produced by convert_messages_to_anthropic."""
    out: Dict[str, Any] = {
        "name": name,
        "description": description,
        "input_schema": {"type": "object", "properties": {}, "required": []},
    }
    out.update(extra)
    return out


def _names(tools: List[Dict[str, Any]]) -> List[str]:
    return [t.get("name", "<?>") for t in tools]


# ---------------------------------------------------------------------------
# Disabled / no-config paths
# ---------------------------------------------------------------------------


def test_no_config_returns_input_unchanged():
    tools = [_tool("a"), _tool("slack_x")]
    assert _apply_tool_search(tools, None) is tools


def test_disabled_config_returns_input_unchanged():
    tools = [_tool("a"), _tool("slack_x")]
    cfg = {"enabled": False, "mcp_server_prefixes": ["slack_"]}
    assert _apply_tool_search(tools, cfg) is tools


def test_empty_tools_returns_input():
    tools: List[Dict[str, Any]] = []
    cfg = {"enabled": True, "mode": "client_side", "mcp_server_prefixes": ["slack_"]}
    assert _apply_tool_search(tools, cfg) is tools


# ---------------------------------------------------------------------------
# client_side mode (the new default)
# ---------------------------------------------------------------------------


def test_client_side_stubs_have_no_defer_loading():
    tools = [_tool("core_tool", "do thing"), _tool("slack_send", "send slack msg")]
    cfg = {
        "enabled": True,
        "mode": "client_side",
        "mcp_server_prefixes": ["slack_"],
        "defer_mcp_tools": True,
    }
    out = _apply_tool_search(tools, cfg)

    # Same names, same order, no server tool prepended.
    assert _names(out) == ["core_tool", "slack_send"]

    # slack_send was stubbed (different description, generic schema).
    stub = next(t for t in out if t["name"] == "slack_send")
    assert "defer_loading" not in stub, "client_side stubs must omit defer_loading"
    assert stub["input_schema"] == {"type": "object"}
    assert stub["description"] != "send slack msg", "stub should replace description"

    # core_tool is eager / unchanged.
    eager = next(t for t in out if t["name"] == "core_tool")
    assert eager["description"] == "do thing"


def test_client_side_no_server_tool_prepended():
    tools = [_tool("a"), _tool("slack_b")]
    cfg = {
        "enabled": True,
        "mode": "client_side",
        "mcp_server_prefixes": ["slack_"],
        "defer_mcp_tools": True,
    }
    out = _apply_tool_search(tools, cfg)
    server_tool_types = {t.get("type") for t in out if "type" in t}
    assert not any(
        v and v.startswith("tool_search_tool_") for v in server_tool_types
    ), "client_side must not prepend Anthropic server tool"


def test_client_side_promoted_tools_skip_stub():
    """Tools in promoted_tools ship their full schema even if MCP-prefixed."""
    tools = [_tool("a"), _tool("slack_promoted", "real desc"), _tool("slack_stubbed")]
    cfg = {
        "enabled": True,
        "mode": "client_side",
        "mcp_server_prefixes": ["slack_"],
        "defer_mcp_tools": True,
        "promoted_tools": {"slack_promoted"},
    }
    out = _apply_tool_search(tools, cfg)

    promoted = next(t for t in out if t["name"] == "slack_promoted")
    assert promoted["description"] == "real desc", "promoted tool keeps full schema"
    assert promoted["input_schema"]["properties"] is not None

    stubbed = next(t for t in out if t["name"] == "slack_stubbed")
    assert stubbed["input_schema"] == {"type": "object"}
    assert "defer_loading" not in stubbed


def test_default_mode_is_client_side():
    """Omitting mode entirely should behave as client_side (the safe new default)."""
    tools = [_tool("a"), _tool("slack_b")]
    cfg = {
        "enabled": True,
        # no "mode" key at all
        "mcp_server_prefixes": ["slack_"],
        "defer_mcp_tools": True,
    }
    out = _apply_tool_search(tools, cfg)

    # No server tool prepended → it's client_side.
    assert all(not t.get("type", "").startswith("tool_search_tool_") for t in out)
    stub = next(t for t in out if t["name"] == "slack_b")
    assert "defer_loading" not in stub


def test_invalid_mode_falls_back_to_client_side():
    tools = [_tool("a"), _tool("slack_b")]
    cfg = {
        "enabled": True,
        "mode": "garbage",
        "mcp_server_prefixes": ["slack_"],
        "defer_mcp_tools": True,
    }
    out = _apply_tool_search(tools, cfg)
    assert all(not t.get("type", "").startswith("tool_search_tool_") for t in out)


# ---------------------------------------------------------------------------
# server_side mode (legacy behavior preserved)
# ---------------------------------------------------------------------------


def test_server_side_prepends_server_tool():
    tools = [_tool("a"), _tool("slack_b")]
    cfg = {
        "enabled": True,
        "mode": "server_side",
        "variant": "regex",
        "mcp_server_prefixes": ["slack_"],
        "defer_mcp_tools": True,
    }
    out = _apply_tool_search(tools, cfg)
    first = out[0]
    assert first.get("type", "").startswith("tool_search_tool_regex_")
    assert first.get("name") == "tool_search_tool_regex"


def test_server_side_stubs_carry_defer_loading():
    tools = [_tool("a"), _tool("slack_b", "real desc")]
    cfg = {
        "enabled": True,
        "mode": "server_side",
        "variant": "regex",
        "mcp_server_prefixes": ["slack_"],
        "defer_mcp_tools": True,
    }
    out = _apply_tool_search(tools, cfg)
    stub = next(t for t in out if t.get("name") == "slack_b")
    assert stub.get("defer_loading") is True
    assert stub["description"] == ""  # empty in server_side mode


def test_server_side_bm25_variant():
    tools = [_tool("a"), _tool("slack_b")]
    cfg = {
        "enabled": True,
        "mode": "server_side",
        "variant": "bm25",
        "mcp_server_prefixes": ["slack_"],
        "defer_mcp_tools": True,
    }
    out = _apply_tool_search(tools, cfg)
    first = out[0]
    assert first.get("type", "").startswith("tool_search_tool_bm25_")
    assert first.get("name") == "tool_search_tool_bm25"


# ---------------------------------------------------------------------------
# Policy / safety guards (mode-independent)
# ---------------------------------------------------------------------------


def test_additional_eager_overrides_mcp_prefix():
    """additional_eager wins over defer_mcp_tools."""
    tools = [_tool("a"), _tool("slack_keep_eager", "real")]
    cfg = {
        "enabled": True,
        "mode": "client_side",
        "mcp_server_prefixes": ["slack_"],
        "defer_mcp_tools": True,
        "additional_eager": ["slack_keep_eager"],
    }
    out = _apply_tool_search(tools, cfg)
    kept = next(t for t in out if t["name"] == "slack_keep_eager")
    assert kept["description"] == "real", "additional_eager should bypass the stub"


def test_additional_deferred_works_without_mcp_prefix():
    tools = [_tool("a", "real"), _tool("b", "real")]
    cfg = {
        "enabled": True,
        "mode": "client_side",
        "mcp_server_prefixes": [],
        "defer_mcp_tools": False,
        "additional_deferred": ["b"],
    }
    out = _apply_tool_search(tools, cfg)
    stub = next(t for t in out if t["name"] == "b")
    assert stub["input_schema"] == {"type": "object"}
    eager = next(t for t in out if t["name"] == "a")
    assert eager["description"] == "real"


def test_all_deferred_returns_input_unchanged():
    """Avoids Anthropic 400 + a 100% stub array is useless anyway."""
    tools = [_tool("slack_a"), _tool("slack_b")]
    cfg = {
        "enabled": True,
        "mode": "client_side",
        "mcp_server_prefixes": ["slack_"],
        "defer_mcp_tools": True,
    }
    out = _apply_tool_search(tools, cfg)
    # All tools matched the slack_ prefix → no eager anchor → unchanged.
    assert out is tools


def test_none_deferred_returns_input_unchanged():
    """Nothing matches the deferral policy → no transformation needed."""
    tools = [_tool("a"), _tool("b")]
    cfg = {
        "enabled": True,
        "mode": "client_side",
        "mcp_server_prefixes": ["slack_"],
        "defer_mcp_tools": True,
    }
    out = _apply_tool_search(tools, cfg)
    assert out is tools


def test_cache_control_preserved_on_stub():
    tools = [
        _tool("a"),
        _tool("slack_b", cache_control={"type": "ephemeral"}),
    ]
    cfg = {
        "enabled": True,
        "mode": "client_side",
        "mcp_server_prefixes": ["slack_"],
        "defer_mcp_tools": True,
    }
    out = _apply_tool_search(tools, cfg)
    stub = next(t for t in out if t["name"] == "slack_b")
    assert stub.get("cache_control") == {"type": "ephemeral"}
