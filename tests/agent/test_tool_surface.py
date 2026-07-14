"""Behavior contracts for complete tool-surface assembly."""

from copy import deepcopy

from agent.tool_surface import assemble_full_tool_surface
from tools.tool_search import ToolSearchConfig


def _tool(name: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": name,
            "parameters": {"type": "object", "properties": {}},
        },
    }


def _schema(name: str) -> dict:
    return {
        "name": name,
        "description": name,
        "parameters": {"type": "object", "properties": {}},
    }


def test_full_surface_is_non_mutating_and_deduplicates_injected_families():
    base = [_tool("read_file")]
    memory = [_schema("read_file"), _schema("memory_recall")]
    context = [_schema("context_expand")]
    original_base = deepcopy(base)
    original_memory = deepcopy(memory)
    original_context = deepcopy(context)

    surface = assemble_full_tool_surface(
        base,
        enabled_toolsets=["file", "memory", "context_engine"],
        memory_tool_schemas=memory,
        context_engine_tool_schemas=context,
        tool_search_config=ToolSearchConfig.from_raw({"enabled": "off"}),
    )

    names = [tool["function"]["name"] for tool in surface.tool_defs]
    assert names == ["read_file", "memory_recall", "context_expand"]
    assert surface.injected_names == {
        "memory": ["memory_recall"],
        "context_engine": ["context_expand"],
    }
    assert surface.skipped["memory"] == [
        {"tool": "read_file", "reason": "duplicate tool name"}
    ]
    assert base == original_base
    assert memory == original_memory
    assert context == original_context


def test_full_surface_records_toolset_gates_without_mutating_schemas():
    memory = [_schema("memory_recall")]
    context = [_schema("context_expand")]

    surface = assemble_full_tool_surface(
        [_tool("read_file")],
        enabled_toolsets=["file"],
        memory_tool_schemas=memory,
        context_engine_tool_schemas=context,
        tool_search_config=ToolSearchConfig.from_raw({"enabled": "off"}),
    )

    names = {tool["function"]["name"] for tool in surface.tool_defs}
    assert names == {"read_file"}
    assert surface.skipped == {
        "memory": [{"tool": "memory_recall", "reason": "toolset disabled"}],
        "context_engine": [
            {"tool": "context_expand", "reason": "toolset disabled"}
        ],
    }


def test_tool_search_runs_after_external_families_are_injected():
    from tools.registry import registry

    deferred_name = "surface_deferred_mcp_tool"
    registry.register(
        name=deferred_name,
        handler=lambda args, **kwargs: "{}",
        schema=_schema(deferred_name),
        toolset="mcp-surface-test",
    )
    try:
        surface = assemble_full_tool_surface(
            [_tool(deferred_name)],
            enabled_toolsets=["mcp-surface-test", "memory", "context_engine"],
            memory_tool_schemas=[_schema("surface_memory_recall")],
            context_engine_tool_schemas=[_schema("surface_context_expand")],
            tool_search_config=ToolSearchConfig.from_raw({"enabled": "on"}),
        )
    finally:
        registry.deregister(deferred_name)

    names = {tool["function"]["name"] for tool in surface.tool_defs}
    assert deferred_name not in names
    assert {"tool_search", "tool_describe", "tool_call"}.issubset(names)
    assert {"surface_memory_recall", "surface_context_expand"}.issubset(names)
    assert surface.tool_search_activated is True
    assert surface.deferred_names == [deferred_name]


def test_disabled_toolsets_override_enabled_external_families():
    surface = assemble_full_tool_surface(
        [_tool("read_file")],
        enabled_toolsets=["file", "memory", "context_engine"],
        disabled_toolsets=["memory", "context_engine"],
        memory_tool_schemas=[_schema("disabled_memory")],
        context_engine_tool_schemas=[_schema("disabled_context")],
        tool_search_config=ToolSearchConfig.from_raw({"enabled": "off"}),
    )

    names = {tool["function"]["name"] for tool in surface.tool_defs}
    assert names == {"read_file"}
    assert surface.skipped == {
        "memory": [{"tool": "disabled_memory", "reason": "toolset disabled"}],
        "context_engine": [
            {"tool": "disabled_context", "reason": "toolset disabled"}
        ],
    }


def test_schema_sanitization_failure_is_fail_soft(monkeypatch):
    def _raise(_schemas):
        raise TypeError("non-copyable schema")

    monkeypatch.setattr("tools.schema_sanitizer.sanitize_tool_schemas", _raise)

    surface = assemble_full_tool_surface(
        [_tool("read_file")],
        apply_tool_search=False,
    )

    assert [tool["function"]["name"] for tool in surface.tool_defs] == [
        "read_file"
    ]
