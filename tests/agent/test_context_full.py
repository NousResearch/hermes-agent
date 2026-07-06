"""Tests for raw live session context assembly."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.context_breakdown import compute_session_context_full


SOURCE_LABEL = (
    "Reconstructed base context (cached prefix + history; excludes per-turn "
    "ephemeral injections)"
)


class _MemoryStore:
    def format_for_system_prompt(self, namespace: str) -> str:
        if namespace == "memory":
            return "<memory>Remember pizza preference.</memory>"
        if namespace == "user":
            return "<user_profile>Alfredo likes direct answers.</user_profile>"
        return ""


def _make_agent(*, cached_system_prompt: str | None = "CACHED SYS BYTES"):
    return SimpleNamespace(
        model="openai/gpt-5.4",
        tools=[
            {"type": "function", "function": {"name": "terminal", "description": "run"}},
            {"type": "function", "function": {"name": "mcp_demo_tool", "description": "mcp"}},
            {"type": "function", "function": {"name": "delegate_task", "description": "spawn"}},
        ],
        _cached_system_prompt=cached_system_prompt,
        ephemeral_system_prompt=None,
        _memory_store=_MemoryStore(),
        _memory_enabled=True,
        _user_profile_enabled=True,
        context_compressor=SimpleNamespace(context_length=200_000, last_prompt_tokens=0),
    )


def _history():
    return [
        {"role": "user", "content": "<script>alert(1)</script> SECRET_TOKEN=abc123"},
        {
            "role": "assistant",
            "content": "I will call a tool.",
            "reasoning": "private chain summary",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "terminal", "arguments": "{\"command\": \"pwd\"}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "name": "terminal",
            "content": "tool output SECRET_TOKEN=abc123",
        },
    ]


def test_full_context_uses_cached_system_prompt_without_rebuilding_ordered_system_message():
    agent = _make_agent(cached_system_prompt="CACHED SYS BYTES")
    history = _history()
    build_parts = MagicMock(
        return_value={
            "stable": "MOCK REBUILT SYSTEM",
            "context": "MOCK RULES",
            "volatile": "MOCK VOLATILE",
        }
    )

    with patch("agent.system_prompt.build_system_prompt_parts", build_parts):
        data = compute_session_context_full(agent, history)

    build_parts.assert_not_called()
    assert data["available"] is True
    assert data["state"] == "ready"
    assert data["source"] == "reconstructed_base"
    assert data["source_label"] == SOURCE_LABEL
    assert data["raw_unredacted"] is True
    assert data["exact_capture_available"] is False
    assert data["model"] == "openai/gpt-5.4"
    assert data["context_max"] == 200_000
    assert data["context_used"] > 0

    assert [m["role"] for m in data["messages"]] == ["system", "user", "assistant", "tool"]
    assert data["messages"][0]["content_text"] == "CACHED SYS BYTES"
    assert data["messages"][0]["raw"] == {"role": "system", "content": "CACHED SYS BYTES"}

    ids = {item["id"] for item in data["slices"]}
    assert {
        "system_prompt",
        "rules",
        "skills",
        "memory",
        "tool_definitions",
        "mcp",
        "subagent_definitions",
        "conversation",
    } <= ids
    by_id = {item["id"]: item for item in data["slices"]}
    assert by_id["system_prompt"]["source_accuracy"] == "cached_exact"
    assert "Remember pizza preference" in by_id["memory"]["content_text"]
    assert "Alfredo likes direct answers" in by_id["memory"]["content_text"]
    assert all(item["source_accuracy"] in {"cached_exact", "reconstructed_current"} for item in data["slices"])
    assert all(item["truncated"] is False for item in data["slices"])

    user_msg = data["messages"][1]
    assert "<script>alert(1)</script>" in user_msg["content_text"]
    assert "SECRET_TOKEN=abc123" in user_msg["content_text"]
    assert "&lt;script" not in user_msg["content_text"]

    assistant_msg = data["messages"][2]
    assert assistant_msg["raw"]["tool_calls"] == history[1]["tool_calls"]
    assert assistant_msg["raw"]["reasoning"] == "private chain summary"
    assert assistant_msg["raw"] is not history[1]

    tool_msg = data["messages"][3]
    assert tool_msg["raw"]["tool_call_id"] == "call_1"
    assert tool_msg["raw"]["name"] == "terminal"


def test_full_context_appends_ephemeral_prompt_to_cached_ordered_system_message():
    agent = _make_agent(cached_system_prompt="CACHED SYS BYTES")
    agent.ephemeral_system_prompt = "EPHEMERAL SYS BYTES"

    data = compute_session_context_full(agent, [])

    assert data["messages"][0]["content_text"] == "CACHED SYS BYTES\n\nEPHEMERAL SYS BYTES"
    assert data["messages"][0]["raw"] == {
        "role": "system",
        "content": "CACHED SYS BYTES\n\nEPHEMERAL SYS BYTES",
    }


def test_full_context_cold_agent_reconstructs_current_system_prompt():
    agent = _make_agent(cached_system_prompt=None)
    parts = {
        "stable": "stable base\n<available_skills>skill list</available_skills>",
        "context": "AGENTS.md rules",
        "volatile": "Current time: now",
    }

    with patch("agent.system_prompt.build_system_prompt_parts", return_value=parts) as build_parts:
        data = compute_session_context_full(agent, [])

    build_parts.assert_called_once_with(agent)
    assert data["messages"][0]["role"] == "system"
    assert "stable base" in data["messages"][0]["content_text"]
    assert "AGENTS.md rules" in data["messages"][0]["content_text"]
    assert "Current time: now" in data["messages"][0]["content_text"]
    assert data["messages"][0]["raw"]["content"] == data["messages"][0]["content_text"]
    assert {item["source_accuracy"] for item in data["slices"]} <= {
        "cached_exact",
        "reconstructed_current",
    }


def test_full_context_is_copy_only_and_does_not_mutate_agent_or_history():
    """The inspector must never mutate the live agent or conversation history.

    Prompt caching is sacred: reading context for display cannot perturb
    ``_cached_system_prompt``, ``agent.tools``, or the passed history object
    graph. Also proves the returned ``raw`` messages are deep copies (mutating
    them must not touch the source history).
    """
    import copy

    agent = _make_agent(cached_system_prompt="CACHED SYS BYTES")
    history = _history()

    cached_before = agent._cached_system_prompt
    tools_before = copy.deepcopy(agent.tools)
    history_before = copy.deepcopy(history)

    data = compute_session_context_full(agent, history)

    # Live agent state untouched.
    assert agent._cached_system_prompt == cached_before
    assert agent.tools == tools_before
    # Source history object graph untouched (identity of list + equality of contents).
    assert history == history_before

    # Mutating the returned raw payload must not bleed into the source history.
    data["messages"][1]["raw"]["content"] = "TAMPERED"
    if isinstance(data["messages"][1]["raw"].get("content"), str):
        assert history[0]["content"] != "TAMPERED"
    assert history == history_before
