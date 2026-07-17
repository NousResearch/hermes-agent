"""Regression coverage for MCP toolsets in explicit platform toolsets."""

import json

import model_tools
from tools.registry import registry


def _schema(name: str) -> dict:
    return {
        "name": name,
        "description": f"Test schema for {name}",
        "parameters": {"type": "object", "properties": {}},
    }


def _register_tool(name: str, toolset: str) -> None:
    registry.register(
        name=name,
        toolset=toolset,
        schema=_schema(name),
        handler=lambda args, **kw: json.dumps({"ok": True}),
    )
    model_tools._clear_tool_defs_cache()


def _deregister_tools(*names: str) -> None:
    for name in names:
        registry.deregister(name)
    model_tools._clear_tool_defs_cache()


def _tool_names(defs: list[dict]) -> set[str]:
    return {td["function"]["name"] for td in defs}


def test_explicit_platform_toolsets_auto_include_registered_mcp_toolsets():
    tool_name = "mcp_gateway_auto_test_query"
    _register_tool(tool_name, "mcp-gateway-auto-test")
    try:
        defs = model_tools.get_tool_definitions(
            enabled_toolsets=["terminal"],
            quiet_mode=True,
            skip_tool_search_assembly=True,
        )
        names = _tool_names(defs)
        assert "terminal" in names
        assert tool_name in names
    finally:
        _deregister_tools(tool_name)


def test_disabled_toolsets_still_remove_auto_included_mcp_toolsets():
    tool_name = "mcp_gateway_disabled_test_query"
    _register_tool(tool_name, "mcp-gateway-disabled-test")
    try:
        defs = model_tools.get_tool_definitions(
            enabled_toolsets=["terminal"],
            disabled_toolsets=["mcp-gateway-disabled-test"],
            quiet_mode=True,
            skip_tool_search_assembly=True,
        )
        names = _tool_names(defs)
        assert "terminal" in names
        assert tool_name not in names
    finally:
        _deregister_tools(tool_name)


def test_explicit_mcp_enabled_toolset_does_not_widen_to_every_mcp_server():
    in_scope = "mcp_gateway_scoped_test_query"
    out_of_scope = "mcp_gateway_scoped_other_query"
    _register_tool(in_scope, "mcp-gateway-scoped-test")
    _register_tool(out_of_scope, "mcp-gateway-scoped-other")
    try:
        defs = model_tools.get_tool_definitions(
            enabled_toolsets=["mcp-gateway-scoped-test"],
            quiet_mode=True,
            skip_tool_search_assembly=True,
        )
        names = _tool_names(defs)
        assert in_scope in names
        assert out_of_scope not in names
    finally:
        _deregister_tools(in_scope, out_of_scope)
