"""Behavioral regressions for the final model-visible tool-result budget."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.agent_runtime_helpers import sanitize_api_messages
from tests.run_agent.test_tool_call_incremental_persistence import (
    _make_agent,
    _mock_tool_call,
)


def _paired_messages(content, *, metadata=None):
    tool = {
        "role": "tool",
        "tool_call_id": "call_budget",
        "name": "synthetic",
        "content": content,
    }
    if metadata is not None:
        tool["_tool_result_budget"] = metadata
    return [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_budget",
                    "type": "function",
                    "function": {"name": "synthetic", "arguments": "{}"},
                }
            ],
        },
        tool,
    ]


def test_pre_model_boundary_caps_legacy_tool_messages_without_wire_metadata(caplog):
    caplog.set_level("INFO", logger="tools.tool_result_storage")
    sanitized = sanitize_api_messages(_paired_messages("[]" * 8_000))

    assert len(sanitized[-1]["content"].encode("utf-8")) <= 10_000
    assert "_tool_result_budget" not in sanitized[-1]
    assert "initial=16000 limit=10000" in caplog.text
    assert "[][][][][][][][][][]" not in caplog.text


def test_pre_model_boundary_consumes_valid_call_local_override():
    sanitized = sanitize_api_messages(
        _paired_messages(
            "x" * 20_000,
            metadata={
                "schema_version": 1,
                "limit_tokens": 24_000,
                "override_requested": True,
            },
        )
    )

    assert len(sanitized[-1]["content"].encode("utf-8")) == 20_000
    assert "_tool_result_budget" not in sanitized[-1]


def test_copilot_acp_prompt_formatter_bounds_tool_result_text():
    from agent.copilot_acp_client import _format_messages_as_prompt

    prompt = _format_messages_as_prompt(
        [
            {
                "role": "tool",
                "name": "legacy_acp_result",
                "tool_call_id": "call_acp_budget",
                "content": "界" * 8_000,
            }
        ]
    )

    rendered = prompt.split("Tool:\n", 1)[1].split(
        "\n\nContinue the conversation", 1
    )[0]
    assert len(rendered.encode("utf-8")) <= 10_000


def test_sequential_override_is_removed_and_does_not_leak_to_next_call():
    agent = _make_agent()
    agent._flush_messages_to_session_db = MagicMock()
    assistant = SimpleNamespace(
        content="",
        tool_calls=[
            _mock_tool_call(
                arguments=json.dumps(
                    {"query": "first", "result_token_limit": 24_000}
                ),
                call_id="call_override",
            ),
            _mock_tool_call(
                arguments=json.dumps({"query": "second"}),
                call_id="call_default",
            ),
        ],
    )
    seen_args = []

    def dispatch(_name, args, _task, **_kwargs):
        seen_args.append(dict(args))
        return "x" * 20_000

    messages = []
    with patch("run_agent.handle_function_call", side_effect=dispatch):
        agent._execute_tool_calls_sequential(assistant, messages, "task-budget")

    assert seen_args == [{"query": "first"}, {"query": "second"}]
    assert 10_000 < len(messages[0]["content"].encode("utf-8")) <= 24_000
    assert messages[0]["_tool_result_budget"]["override_requested"] is True
    assert len(messages[1]["content"].encode("utf-8")) <= 10_000
    assert messages[1]["_tool_result_budget"]["limit_tokens"] == 10_000


def test_concurrent_override_is_call_local_and_removed_before_handler():
    agent = _make_agent()
    agent._flush_messages_to_session_db = MagicMock()
    assistant = SimpleNamespace(
        content="",
        tool_calls=[
            _mock_tool_call(
                arguments=json.dumps(
                    {"query": "first", "result_token_limit": 24_000}
                ),
                call_id="call_override",
            ),
            _mock_tool_call(
                arguments=json.dumps({"query": "second"}),
                call_id="call_default",
            ),
        ],
    )
    seen_args = {}

    def invoke(_name, args, _task, call_id, **_kwargs):
        seen_args[call_id] = dict(args)
        return "x" * 20_000

    messages = []
    with patch.object(agent, "_invoke_tool", side_effect=invoke):
        agent._execute_tool_calls_concurrent(assistant, messages, "task-budget")

    assert seen_args == {
        "call_override": {"query": "first"},
        "call_default": {"query": "second"},
    }
    assert 10_000 < len(messages[0]["content"].encode("utf-8")) <= 24_000
    assert len(messages[1]["content"].encode("utf-8")) <= 10_000


@pytest.mark.parametrize(
    "invalid",
    [True, "12000", 12.5, 0, -1, 32_001],
)
def test_invalid_override_is_rejected_before_sequential_handler(invalid):
    agent = _make_agent()
    agent._flush_messages_to_session_db = MagicMock()
    assistant = SimpleNamespace(
        content="",
        tool_calls=[
            _mock_tool_call(
                arguments=json.dumps(
                    {"query": "never-run", "result_token_limit": invalid}
                ),
                call_id="call_invalid",
            )
        ],
    )
    messages = []

    with patch("run_agent.handle_function_call") as dispatch:
        agent._execute_tool_calls_sequential(assistant, messages, "task-budget")

    dispatch.assert_not_called()
    assert "result_token_limit" in messages[0]["content"]
    assert "error" in messages[0]["content"]


def test_agent_level_session_search_uses_same_final_budget():
    agent = _make_agent()
    agent._flush_messages_to_session_db = MagicMock()
    agent._get_session_db_for_recall = MagicMock(return_value=object())
    assistant = SimpleNamespace(
        content="",
        tool_calls=[
            _mock_tool_call(
                name="session_search",
                arguments=json.dumps({"query": "large history"}),
                call_id="call_session_search",
            )
        ],
    )
    messages = []

    with patch(
        "tools.session_search_tool.session_search",
        return_value="[]" * 8_000,
    ):
        agent._execute_tool_calls_sequential(assistant, messages, "task-budget")

    assert len(messages[0]["content"].encode("utf-8")) <= 10_000
    assert messages[0]["_tool_result_budget"]["truncated"] is True


def test_tool_call_bridge_propagates_inner_override_and_strips_it():
    agent = _make_agent()
    agent._flush_messages_to_session_db = MagicMock()
    assistant = SimpleNamespace(
        content="",
        tool_calls=[
            _mock_tool_call(
                name="tool_call",
                arguments=json.dumps(
                    {
                        "name": "mcp_budget_probe",
                        "arguments": {
                            "payload": "safe",
                            "result_token_limit": 24_000,
                        },
                    }
                ),
                call_id="call_bridge",
            )
        ],
    )
    seen = {}

    def dispatch(name, args, _task, **_kwargs):
        seen["name"] = name
        seen["args"] = dict(args)
        return "x" * 20_000

    messages = []
    with (
        patch(
            "tools.tool_search.resolve_underlying_call",
            return_value=(
                "mcp_budget_probe",
                {"payload": "safe", "result_token_limit": 24_000},
                None,
            ),
        ),
        patch(
            "agent.tool_executor._tool_search_scoped_names",
            return_value={"mcp_budget_probe"},
        ),
        patch("run_agent.handle_function_call", side_effect=dispatch),
    ):
        agent._execute_tool_calls_sequential(assistant, messages, "task-budget")

    assert seen == {"name": "mcp_budget_probe", "args": {"payload": "safe"}}
    assert 10_000 < len(messages[0]["content"].encode("utf-8")) <= 24_000
    assert messages[0]["_tool_result_budget"]["override_requested"] is True
