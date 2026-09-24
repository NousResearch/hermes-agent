"""Regression coverage for deferred schemas on the Chat Completions wire."""

from types import SimpleNamespace


def _tool(name: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"{name} description",
            "parameters": {"type": "object", "properties": {}},
        },
    }


def test_described_deferred_schema_is_projected_only_on_chat_completions():
    from agent.turn_request_assembly import (
        project_described_tool_schemas,
        record_described_deferred_tool_schemas,
    )

    agent = SimpleNamespace(api_mode="chat_completions", valid_tool_names={"tool_describe"})
    described = {
        "tools": {
            "session_search": {
                "description": "Search prior session messages.",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
            }
        }
    }

    record_described_deferred_tool_schemas(agent, described)
    projected = project_described_tool_schemas(agent, [_tool("tool_describe")])

    assert [item["function"]["name"] for item in projected] == ["tool_describe", "session_search"]
    assert "session_search" in agent.valid_tool_names


def test_undisclosed_deferred_schemas_remain_absent_and_other_wires_are_unchanged():
    from agent.turn_request_assembly import (
        project_described_tool_schemas,
        record_described_deferred_tool_schemas,
    )

    agent = SimpleNamespace(api_mode="codex_responses", valid_tool_names={"tool_describe"})
    record_described_deferred_tool_schemas(agent, {
        "tools": {"session_search": {"description": "Search.", "parameters": {"type": "object"}}}
    })

    projected = project_described_tool_schemas(agent, [_tool("tool_describe")])

    assert [item["function"]["name"] for item in projected] == ["tool_describe"]
    assert "session_search" not in agent.valid_tool_names
