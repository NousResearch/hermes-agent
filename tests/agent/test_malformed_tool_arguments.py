"""Malformed model tool arguments are rejected at the dispatch boundary."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent


def _make_agent() -> AIAgent:
    tool_defs = [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "search",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    with (
        patch("model_tools.get_tool_definitions", return_value=tool_defs),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("hermes_cli.config.load_config", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.client = MagicMock()
    agent._flush_messages_to_session_db = MagicMock()
    return agent


def _tool_call(call_id: str, arguments: str):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name="web_search", arguments=arguments),
    )


@pytest.mark.parametrize("dispatch_mode", ["sequential", "concurrent"])
@pytest.mark.parametrize(
    "bad_arguments",
    [
        pytest.param("not-json", id="malformed-json"),
        pytest.param('"scalar"', id="scalar"),
        pytest.param("[]", id="list"),
        pytest.param("", id="empty"),
        pytest.param('{"query": "cut off', id="truncated"),
    ],
)
def test_malformed_arguments_are_rejected_without_blocking_valid_sibling(
    dispatch_mode: str,
    bad_arguments: str,
):
    agent = _make_agent()
    assistant_message = SimpleNamespace(
        content="",
        tool_calls=[
            _tool_call("call-bad", bad_arguments),
            _tool_call("call-good", '{"query": "valid"}'),
        ],
    )
    messages = []
    executed = []

    def fake_dispatch(name, args, task_id, *positional, **kwargs):
        call_id = kwargs.get("tool_call_id") or (positional[0] if positional else None)
        executed.append((name, args, call_id))
        return json.dumps({"ok": args["query"]})

    with (
        patch("model_tools.handle_function_call", side_effect=fake_dispatch),
        patch.object(agent, "_invoke_tool", side_effect=fake_dispatch),
        patch(
            "agent.tool_executor.maybe_persist_tool_result",
            side_effect=lambda **kwargs: kwargs["content"],
        ),
    ):
        execute = getattr(agent, f"_execute_tool_calls_{dispatch_mode}")
        execute(assistant_message, messages, "task-1")

    assert executed == [("web_search", {"query": "valid"}, "call-good")]
    assert [message["tool_call_id"] for message in messages] == ["call-bad", "call-good"]
    assert len([message for message in messages if message["tool_call_id"] == "call-bad"]) == 1

    assert '"error": "Invalid tool arguments"' in messages[0]["content"]
    assert "JSON object" in messages[0]["content"]
    assert json.loads(messages[1]["content"]) == {"ok": "valid"}


def _wrapped(layers: int, innermost: dict) -> dict:
    payload = innermost
    for _ in range(layers):
        payload = {"arguments": payload}
    return payload


def test_nested_arguments_wrappers_are_unwrapped():
    from agent.tool_executor import _parse_tool_arguments

    args, parse_error = _parse_tool_arguments(json.dumps(_wrapped(1, {"query": "x"})), "web_search")
    assert parse_error is None
    assert args == {"query": "x"}

    args, parse_error = _parse_tool_arguments(
        json.dumps(_wrapped(11, {"query": "deep"})), "web_search"
    )
    assert parse_error is None
    assert args == {"query": "deep"}


def test_plain_arguments_object_is_untouched():
    from agent.tool_executor import _parse_tool_arguments

    args, parse_error = _parse_tool_arguments('{"query": "x", "limit": 2}', "web_search")
    assert parse_error is None
    assert args == {"query": "x", "limit": 2}


def test_sole_key_arguments_with_non_dict_value_is_preserved():
    from agent.tool_executor import _parse_tool_arguments

    args, parse_error = _parse_tool_arguments('{"arguments": "not-an-object"}', "web_search")
    assert parse_error is None
    assert args == {"arguments": "not-an-object"}


def test_multi_key_dict_with_arguments_key_is_preserved():
    from agent.tool_executor import _parse_tool_arguments

    raw = '{"arguments": {"query": "inner"}, "query": "outer"}'
    args, parse_error = _parse_tool_arguments(raw, "web_search")
    assert parse_error is None
    assert args == {"arguments": {"query": "inner"}, "query": "outer"}


def test_unwrapping_is_bounded():
    from agent.tool_executor import _UNWRAP_ARGUMENTS_MAX_LAYERS, _parse_tool_arguments

    args, parse_error = _parse_tool_arguments(
        json.dumps(_wrapped(_UNWRAP_ARGUMENTS_MAX_LAYERS + 4, {"query": "deep"})), "web_search"
    )
    assert parse_error is None
    assert args == _wrapped(4, {"query": "deep"})


def test_exempt_tool_arguments_wrapper_is_preserved():
    from agent.tool_executor import _parse_tool_arguments

    args, parse_error = _parse_tool_arguments(
        json.dumps(_wrapped(2, {"arguments": "payload"})), "tool_call"
    )
    assert parse_error is None
    assert args == _wrapped(2, {"arguments": "payload"})


@pytest.mark.parametrize("dispatch_mode", ["sequential", "concurrent"])
def test_nested_arguments_wrapper_is_unwrapped_before_dispatch(dispatch_mode: str):
    agent = _make_agent()
    assistant_message = SimpleNamespace(
        content="",
        tool_calls=[
            _tool_call("call-wrapped", json.dumps(_wrapped(3, {"query": "peeled"}))),
        ],
    )
    messages = []
    executed = []

    def fake_dispatch(name, args, task_id, *positional, **kwargs):
        call_id = kwargs.get("tool_call_id") or (positional[0] if positional else None)
        executed.append((name, args, call_id))
        return json.dumps({"ok": args["query"]})

    with (
        patch("model_tools.handle_function_call", side_effect=fake_dispatch),
        patch.object(agent, "_invoke_tool", side_effect=fake_dispatch),
        patch(
            "agent.tool_executor.maybe_persist_tool_result",
            side_effect=lambda **kwargs: kwargs["content"],
        ),
    ):
        execute = getattr(agent, f"_execute_tool_calls_{dispatch_mode}")
        execute(assistant_message, messages, "task-1")

    assert executed == [("web_search", {"query": "peeled"}, "call-wrapped")]
    assert json.loads(messages[0]["content"]) == {"ok": "peeled"}
