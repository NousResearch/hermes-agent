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
        patch("run_agent.get_tool_definitions", return_value=tool_defs),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("hermes_cli.config.load_config", return_value={}),
        patch("run_agent.OpenAI"),
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


def _named_tool_call(call_id: str, name: str, arguments: str):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
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
        patch("run_agent.handle_function_call", side_effect=fake_dispatch),
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


# ---------------------------------------------------------------------------
# Issue #83937: empty-string arguments for tools with no required params must
# be accepted as {} at the execution boundary, not rejected as invalid JSON.
# ---------------------------------------------------------------------------


def _register_test_tool(name: str, required=()):
    from tools.registry import registry

    def _handler(args, task_id=None, **kw):
        return json.dumps({"ok": True, "args": args})

    params = {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "q"}},
    }
    if required:
        params["required"] = list(required)
    registry.register(
        name=name,
        toolset="test-empty-83937",
        handler=_handler,
        schema={"type": "function",
                "function": {"name": name, "description": f"desc {name}",
                             "parameters": params}},
    )


@pytest.mark.parametrize("empty", ["", "   "])
def test_parse_empty_arguments_accepted_for_tool_without_required(empty):
    """Empty/whitespace-only arguments parse to {} for tools whose schema
    declares no required parameters (issue #83937)."""
    from agent.tool_executor import _parse_tool_arguments

    _register_test_tool("parse_empty_ok_83937")
    args, err = _parse_tool_arguments(empty, "parse_empty_ok_83937")
    assert err is None
    assert args == {}


def test_parse_empty_arguments_still_rejected_for_tool_with_required():
    """Tools with required params keep the strict rejection (issue #83937)."""
    from agent.tool_executor import _parse_tool_arguments

    _register_test_tool("parse_empty_req_83937", required=("query",))
    args, err = _parse_tool_arguments("", "parse_empty_req_83937")
    assert err is not None
    assert "Invalid tool arguments" in err


def test_parse_empty_object_arguments_unchanged():
    """Literal '{}' parses to {} as before — normalization is a no-op."""
    from agent.tool_executor import _parse_tool_arguments

    _register_test_tool("parse_empty_obj_83937")
    args, err = _parse_tool_arguments("{}", "parse_empty_obj_83937")
    assert err is None
    assert args == {}


@pytest.mark.parametrize("dispatch_mode", ["sequential", "concurrent"])
def test_empty_arguments_execute_for_tool_without_required(dispatch_mode):
    """End-to-end: an empty-string tool_call for a no-required-param tool
    executes with {} instead of looping on a JSON parse error."""
    agent = _make_agent()
    _register_test_tool("empty_exec_ok_83937")
    assistant_message = SimpleNamespace(
        content="",
        tool_calls=[
            _named_tool_call("call-empty", "empty_exec_ok_83937", ""),
            _named_tool_call("call-obj", "empty_exec_ok_83937", "{}"),
        ],
    )
    messages = []
    executed = []

    def fake_dispatch(name, args, task_id, *positional, **kwargs):
        executed.append((name, dict(args)))
        return json.dumps({"ok": True})

    with (
        patch("run_agent.handle_function_call", side_effect=fake_dispatch),
        patch.object(agent, "_invoke_tool", side_effect=fake_dispatch),
        patch(
            "agent.tool_executor.maybe_persist_tool_result",
            side_effect=lambda **kwargs: kwargs["content"],
        ),
    ):
        execute = getattr(agent, f"_execute_tool_calls_{dispatch_mode}")
        execute(assistant_message, messages, "task-1")

    assert executed == [("empty_exec_ok_83937", {}), ("empty_exec_ok_83937", {})]
    assert all(message["role"] == "tool" for message in messages)
    assert len(messages) == 2
