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


def test_pre_model_boundary_ignores_legacy_override_metadata():
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

    assert len(sanitized[-1]["content"].encode("utf-8")) <= 10_000
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


def test_executor_does_not_reserve_mcp_business_argument_name():
    agent = _make_agent()
    agent._flush_messages_to_session_db = MagicMock()
    assistant = SimpleNamespace(
        content="",
        tool_calls=[
            _mock_tool_call(
                name="mcp__external__collision",
                arguments=json.dumps(
                    {"result_token_limit": "server-owned-value"}
                ),
                call_id="call_collision",
            )
        ],
    )
    seen = {}

    def dispatch(name, args, _task, **_kwargs):
        seen["name"] = name
        seen["args"] = dict(args)
        return '{"ok":true}'

    messages = []
    with patch("run_agent.handle_function_call", side_effect=dispatch):
        agent._execute_tool_calls_sequential(assistant, messages, "task-budget")

    assert seen == {
        "name": "mcp__external__collision",
        "args": {"result_token_limit": "server-owned-value"},
    }


def test_sequential_business_argument_is_preserved_with_fixed_budget():
    agent = _make_agent()
    agent._flush_messages_to_session_db = MagicMock()
    assistant = SimpleNamespace(
        content="",
        tool_calls=[
            _mock_tool_call(
                arguments=json.dumps(
                    {"query": "first", "result_token_limit": 24_000}
                ),
                call_id="call_business_arg",
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

    assert seen_args == [
        {"query": "first", "result_token_limit": 24_000},
        {"query": "second"},
    ]
    assert all(
        len(message["content"].encode("utf-8")) <= 10_000
        for message in messages
    )


def test_concurrent_business_argument_is_preserved_with_fixed_budget():
    agent = _make_agent()
    agent._flush_messages_to_session_db = MagicMock()
    assistant = SimpleNamespace(
        content="",
        tool_calls=[
            _mock_tool_call(
                arguments=json.dumps(
                    {"query": "first", "result_token_limit": 24_000}
                ),
                call_id="call_business_arg",
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
        "call_business_arg": {"query": "first", "result_token_limit": 24_000},
        "call_default": {"query": "second"},
    }
    assert all(
        len(message["content"].encode("utf-8")) <= 10_000
        for message in messages
    )


@pytest.mark.parametrize(
    "value",
    [True, "12000", 12.5, 0, -1, 32_001],
)
def test_result_token_limit_business_values_are_not_interpreted(value):
    agent = _make_agent()
    agent._flush_messages_to_session_db = MagicMock()
    assistant = SimpleNamespace(
        content="",
        tool_calls=[
            _mock_tool_call(
                name="mcp__external__collision",
                arguments=json.dumps({"result_token_limit": value}),
                call_id="call_business_value",
            )
        ],
    )
    messages = []

    with patch(
        "run_agent.handle_function_call",
        return_value='{"ok":true}',
    ) as dispatch:
        agent._execute_tool_calls_sequential(assistant, messages, "task-budget")

    assert dispatch.call_args.args[:2] == (
        "mcp__external__collision",
        {"result_token_limit": value},
    )
    assert len(messages[0]["content"].encode("utf-8")) <= 10_000


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


def test_tool_call_bridge_preserves_inner_business_argument():
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
                            "result_token_limit": "server-owned-value",
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
                {"payload": "safe", "result_token_limit": "server-owned-value"},
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

    assert seen == {
        "name": "mcp_budget_probe",
        "args": {
            "payload": "safe",
            "result_token_limit": "server-owned-value",
        },
    }
    assert len(messages[0]["content"].encode("utf-8")) <= 10_000
