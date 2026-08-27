"""Regression tests for A2A client-tool schema registration."""

from __future__ import annotations

import json

from plugins.platforms.a2a import tools as a2a_tools
from tools import tool_search
from tools.registry import ToolRegistry


def test_a2a_call_schema_round_trips_through_tool_describe(monkeypatch):
    registry = ToolRegistry()

    class _Context:
        def register_tool(self, name, toolset, schema, handler, **kwargs):
            registry.register(
                name=name,
                toolset=toolset,
                schema=schema,
                handler=handler,
                **kwargs,
            )

    a2a_tools.register_tools(_Context())
    definitions = registry.get_definitions({"a2a_call"})
    monkeypatch.setattr(
        tool_search,
        "is_deferrable_tool_name",
        lambda name: name == "a2a_call",
    )

    described = json.loads(
        tool_search.dispatch_tool_describe(
            {"names": ["a2a_call"]},
            current_tool_defs=definitions,
        )
    )["tools"]["a2a_call"]

    assert described["description"]
    assert described["parameters"]["required"] == ["agent", "message"]
    assert set(described["parameters"]["properties"]) == {
        "agent",
        "message",
        "context_id",
    }
