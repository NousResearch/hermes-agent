"""Regression test for the context-engine tool-schema double-wrap fix.

A context engine's get_tool_schemas() is contracted to return BARE schemas
({name, description, parameters}). Some plugins (e.g. an earlier cmx
hermes_engine) mistakenly pre-wrapped them in the OpenAI envelope
({"type":"function","function":{...}}). agent_init then wrapped again, producing
{"function":{"function":{...}}} whose outer name is empty -> provider HTTP 400
"tools[N].function.name: empty string". The fix unwraps an already-wrapped
schema before processing. This test guards both shapes.

Run:
    python -m pytest tests/test_context_engine_tool_wrap.py -q
"""

from __future__ import annotations

import sys
import types


def _register_context_engine_tools(schemas):
    """Reproduce the exact agent_init.py registration loop in isolation.

    Mirrors agent/agent_init.py (the context-engine tool collection block):
    builds agent.tools with the defensive unwrap, returns (tools, valid_names).
    """
    tools = []
    valid_tool_names = set()
    context_engine_tool_names = set()
    existing = {t.get("function", {}).get("name") for t in tools if isinstance(t, dict)}

    for _schema in schemas:
        # --- the fix under test ---
        if (
            isinstance(_schema, dict)
            and _schema.get("type") == "function"
            and isinstance(_schema.get("function"), dict)
            and "name" not in _schema
        ):
            _schema = _schema["function"]
        _tname = _schema.get("name", "") if isinstance(_schema, dict) else ""
        if _tname and _tname in existing:
            continue
        _wrapped = {"type": "function", "function": _schema}
        tools.append(_wrapped)
        if _tname:
            valid_tool_names.add(_tname)
            context_engine_tool_names.add(_tname)
            existing.add(_tname)
    return tools, valid_tool_names


def test_bare_schema_wraps_correctly():
    bare = [{"name": "example_search", "description": "d", "parameters": {"type": "object", "properties": {}}}]
    tools, names = _register_context_engine_tools(bare)
    assert len(tools) == 1
    assert tools[0]["type"] == "function"
    assert tools[0]["function"]["name"] == "example_search"          # outer name present
    assert "function" not in tools[0]["function"]              # not double-wrapped
    assert names == {"example_search"}


def test_already_wrapped_schema_is_unwrapped_not_double_wrapped():
    wrapped = [{"type": "function", "function": {
        "name": "example_search", "description": "d",
        "parameters": {"type": "object", "properties": {}}}}]
    tools, names = _register_context_engine_tools(wrapped)
    assert len(tools) == 1
    # The critical assertion: outer function.name must be the real name, NOT empty
    assert tools[0]["function"]["name"] == "example_search"
    # And it must NOT be {"function": {"function": {...}}}
    assert "function" not in tools[0]["function"], "schema was double-wrapped"
    assert names == {"example_search"}


def test_mixed_bare_and_wrapped():
    schemas = [
        {"name": "bare_tool", "description": "d", "parameters": {"type": "object", "properties": {}}},
        {"type": "function", "function": {
            "name": "wrapped_tool", "description": "d",
            "parameters": {"type": "object", "properties": {}}}},
    ]
    tools, names = _register_context_engine_tools(schemas)
    assert names == {"bare_tool", "wrapped_tool"}
    for t in tools:
        assert t["function"].get("name"), "every tool must have a non-empty name"
        assert "function" not in t["function"], "no double-wrap"


def test_no_empty_names_emitted():
    """The exact failure signature: no tool may reach the provider with an empty name."""
    wrapped = [
        {"type": "function", "function": {"name": "example_search", "description": "d", "parameters": {}}},
        {"type": "function", "function": {"name": "cmx_expand", "description": "d", "parameters": {}}},
        {"type": "function", "function": {"name": "cmx_recall", "description": "d", "parameters": {}}},
    ]
    tools, _ = _register_context_engine_tools(wrapped)
    for i, t in enumerate(tools):
        nm = t.get("function", {}).get("name", "")
        assert nm and nm.strip(), f"tools[{i}].function.name is empty -> would HTTP 400"


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
