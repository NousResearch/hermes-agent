from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from agent.turn_events import record_turn_tool_events
from agent.tool_dispatch_helpers import _maybe_wrap_untrusted
from gateway.becky_loops import should_auto_close_becky_loop


def _tool_call(call_id: str, name: str, arguments: dict) -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


def test_records_one_successful_direct_tool_without_raw_result_content() -> None:
    agent = SimpleNamespace(_turn_tool_events=[])
    assistant = SimpleNamespace(
        tool_calls=[_tool_call("call-1", "mcp_todoist_add_tasks", {"tasks": []})]
    )
    messages = [
        {"role": "assistant", "tool_calls": [{"id": "call-1"}]},
        {
            "role": "tool",
            "name": "mcp_todoist_add_tasks",
            "tool_call_id": "call-1",
            "content": '{"success": true, "tasks": [{"id": "1"}]}'
        },
    ]

    record_turn_tool_events(agent, assistant, messages, 0)

    assert agent._turn_tool_events == [
        {
            "name": "mcp_todoist_add_tasks",
            "requested_name": "mcp_todoist_add_tasks",
            "success": True,
            "arguments": {},
        }
    ]
    assert "content" not in agent._turn_tool_events[0]


def test_records_success_from_the_wrapped_mcp_tool_result() -> None:
    agent = SimpleNamespace(_turn_tool_events=[])
    assistant = SimpleNamespace(
        tool_calls=[_tool_call("call-1", "mcp_todoist_add_tasks", {})]
    )
    raw_content = '{"success": true, "tasks": [{"id": "1"}]}'
    messages = [
        {
            "role": "tool",
            "name": "mcp_todoist_add_tasks",
            "tool_call_id": "call-1",
            "content": _maybe_wrap_untrusted(
                "mcp_todoist_add_tasks", raw_content
            ),
        }
    ]

    record_turn_tool_events(agent, assistant, messages, 0)

    assert agent._turn_tool_events[0]["success"] is True


def test_records_terminal_exit_code_and_marks_failed_command() -> None:
    agent = SimpleNamespace(_turn_tool_events=[])
    assistant = SimpleNamespace(
        tool_calls=[
            _tool_call(
                "call-1",
                "terminal",
                {"command": "python google_api.py calendar create"},
            )
        ]
    )
    messages = [
        {
            "role": "tool",
            "name": "terminal",
            "tool_call_id": "call-1",
            "content": json.dumps({"exit_code": 1, "error": "failed"}),
        }
    ]

    record_turn_tool_events(agent, assistant, messages, 0)

    assert agent._turn_tool_events == [
        {
            "name": "terminal",
            "requested_name": "terminal",
            "success": False,
            "arguments": {
                "command": "python google_api.py calendar create",
                "background": False,
            },
            "exit_code": 1,
        }
    ]


def test_marks_structured_false_success_as_failed() -> None:
    agent = SimpleNamespace(_turn_tool_events=[])
    assistant = SimpleNamespace(
        tool_calls=[_tool_call("call-1", "mcp_todoist_add_tasks", {})]
    )
    messages = [
        {
            "role": "tool",
            "name": "mcp_todoist_add_tasks",
            "tool_call_id": "call-1",
            "content": '{"success": false}',
        }
    ]

    record_turn_tool_events(agent, assistant, messages, 0)

    assert agent._turn_tool_events[0]["success"] is False


def test_marks_tool_call_unwrap_as_tool_search() -> None:
    agent = SimpleNamespace(_turn_tool_events=[])
    assistant = SimpleNamespace(
        tool_calls=[
            _tool_call(
                "call-1",
                "tool_call",
                {
                    "name": "mcp_todoist_add_tasks",
                    "arguments": {},
                },
            )
        ]
    )
    messages = [
        {
            "role": "tool",
            "name": "mcp_todoist_add_tasks",
            "tool_call_id": "call-1",
            "content": '{"success": true}',
        }
    ]

    record_turn_tool_events(agent, assistant, messages, 0)

    assert agent._turn_tool_events[0]["name"] == "mcp_todoist_add_tasks"
    assert agent._turn_tool_events[0]["via_tool_search"] is True


def test_mismatched_tool_result_id_is_not_correlated_as_success() -> None:
    agent = SimpleNamespace(_turn_tool_events=[])
    assistant = SimpleNamespace(
        tool_calls=[_tool_call("call-1", "mcp_todoist_add_tasks", {})]
    )
    messages = [
        {
            "role": "tool",
            "name": "mcp_todoist_add_tasks",
            "tool_call_id": "other-call",
            "content": '{"success": true, "tasks": [{"id": "1"}]}',
        }
    ]

    record_turn_tool_events(agent, assistant, messages, 0)

    assert agent._turn_tool_events[0]["success"] is False


def test_extra_tool_result_prevents_single_action_evidence() -> None:
    agent = SimpleNamespace(_turn_tool_events=[])
    assistant = SimpleNamespace(
        tool_calls=[_tool_call("call-1", "mcp_todoist_add_tasks", {})]
    )
    messages = [
        {
            "role": "tool",
            "name": "mcp_todoist_add_tasks",
            "tool_call_id": "call-1",
            "content": '{"success": true, "tasks": [{"id": "1"}]}',
        },
        {
            "role": "tool",
            "name": "mcp_todoist_add_tasks",
            "tool_call_id": "extra-call",
            "content": '{"success": true, "tasks": [{"id": "2"}]}',
        },
    ]

    record_turn_tool_events(agent, assistant, messages, 0)

    result = {
        "completed": True,
        "final_response": "Done.",
        "turn_exit_reason": "text_response(finish_reason=stop)",
        "turn_tool_events": agent._turn_tool_events,
    }
    assert should_auto_close_becky_loop(result) is False


def test_missing_tool_call_id_does_not_use_positional_fallback() -> None:
    agent = SimpleNamespace(_turn_tool_events=[])
    assistant = SimpleNamespace(
        tool_calls=[SimpleNamespace(
            id="",
            function=SimpleNamespace(
                name="mcp_todoist_add_tasks", arguments="{}"
            ),
        )]
    )
    messages = [
        {
            "role": "tool",
            "name": "mcp_todoist_add_tasks",
            "tool_call_id": "",
            "content": '{"success": true, "tasks": [{"id": "1"}]}',
        }
    ]

    record_turn_tool_events(agent, assistant, messages, 0)

    assert agent._turn_tool_events[0]["success"] is False


@pytest.mark.parametrize(
    "content",
    [
        "",
        "done",
        "success",
        "provider returned a response",
        "{}",
        '{"status": "pending", "message": "success"}',
        '{"status": "pending", "message": "Task added"}',
        {"status": "pending"},
        {},
        "Task added but an error occurred",
    ],
)
def test_unknown_or_pending_tool_result_is_ambiguous(content: object) -> None:
    agent = SimpleNamespace(_turn_tool_events=[])
    assistant = SimpleNamespace(
        tool_calls=[_tool_call("call-1", "mcp_todoist_add_tasks", {})]
    )
    messages = [
        {
            "role": "tool",
            "name": "mcp_todoist_add_tasks",
            "tool_call_id": "call-1",
            "content": content,
        }
    ]

    record_turn_tool_events(agent, assistant, messages, 0)

    assert agent._turn_tool_events[0]["success"] is False


@pytest.mark.parametrize(
    "content",
    [
        '{"success": true, "tasks": [{"id": "1"}]}',
        "Task added successfully",
    ],
)
def test_explicit_success_receipt_is_recorded_as_success(content: str) -> None:
    agent = SimpleNamespace(_turn_tool_events=[])
    assistant = SimpleNamespace(
        tool_calls=[_tool_call("call-1", "mcp_todoist_add_tasks", {})]
    )
    messages = [
        {
            "role": "tool",
            "name": "mcp_todoist_add_tasks",
            "tool_call_id": "call-1",
            "content": content,
        }
    ]

    record_turn_tool_events(agent, assistant, messages, 0)

    assert agent._turn_tool_events[0]["success"] is True
