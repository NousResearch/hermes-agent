from __future__ import annotations

import logging
from types import SimpleNamespace

from agent.conversation_loop import (
    _log_request_budget_telemetry,
    _request_budget_snapshot,
)


def test_request_budget_snapshot_breaks_out_system_and_tool_schema():
    messages = [
        {"role": "system", "content": "system prompt" * 10},
        {"role": "user", "content": "hello"},
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "mcp_big_tool",
                "description": "x" * 400,
                "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
            },
        }
    ]

    budget = _request_budget_snapshot(messages, tools)

    assert budget["message_count"] == 2
    assert budget["tool_count"] == 1
    assert budget["system_prompt_chars"] >= len("system prompt")
    assert budget["system_prompt_tokens"] > 0
    assert budget["conversation_tokens"] > 0
    assert budget["tool_schema_chars"] > 400
    assert budget["tool_schema_tokens"] > 0
    assert budget["total_rough_tokens"] >= budget["message_tokens"]


def test_request_budget_snapshot_includes_top_tool_schema_contributors():
    messages = [{"role": "user", "content": "hello"}]
    tools = [
        {"type": "function", "function": {"name": "tiny_tool", "description": "x", "parameters": {"type": "object", "properties": {}}}},
        {"type": "function", "function": {"name": "huge_tool", "description": "y" * 800, "parameters": {"type": "object", "properties": {"q": {"type": "string", "description": "z" * 400}}}}},
        {"type": "function", "function": {"name": "medium_tool", "description": "m" * 200, "parameters": {"type": "object", "properties": {}}}},
    ]

    budget = _request_budget_snapshot(messages, tools)

    top = budget["top_tool_schema_tokens"]
    assert [item["name"] for item in top[:3]] == ["huge_tool", "medium_tool", "tiny_tool"]
    assert top[0]["schema_chars"] > top[1]["schema_chars"] > top[2]["schema_chars"]
    assert top[0]["schema_tokens"] > 0


def test_request_budget_telemetry_logs_and_stores_last_snapshot(caplog):
    agent = SimpleNamespace()
    budget = {
        "total_rough_tokens": 123,
        "message_tokens": 45,
        "system_prompt_tokens": 20,
        "tool_schema_tokens": 58,
        "message_chars": 180,
        "system_prompt_chars": 80,
        "tool_schema_chars": 232,
        "tool_count": 7,
        "top_tool_schema_tokens": [
            {"name": "huge_tool", "schema_chars": 160, "schema_tokens": 40},
            {"name": "medium_tool", "schema_chars": 72, "schema_tokens": 18},
        ],
    }

    with caplog.at_level(logging.INFO, logger="agent.conversation_loop"):
        _log_request_budget_telemetry(agent, 3, budget)

    assert agent.session_last_request_budget == budget
    assert "Request budget #3: total~123 tokens" in caplog.text
    assert "tools~58" in caplog.text
    assert "tool_count=7" in caplog.text
    assert "top_tools=huge_tool~40, medium_tool~18" in caplog.text
