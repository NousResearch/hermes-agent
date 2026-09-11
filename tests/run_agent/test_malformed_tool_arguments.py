"""Malformed model tool arguments are rejected at the dispatch boundary."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
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


def _tool_call(call_id: str, arguments: Any, name: str = "web_search"):
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
        pytest.param('{"query":' + "[" * 2_000 + "0" + "]" * 2_000 + "}", id="too-deep"),
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


@pytest.mark.parametrize("dispatch_mode", ["sequential", "concurrent"])
@pytest.mark.parametrize(
    "argument_shape",
    ["direct", "decoded-direct", "deferred-legacy", "deferred-batch", "decoded-deferred-batch"],
)
def test_non_replayable_history_arguments_are_blocked_while_fresh_arguments_run(
    dispatch_mode: str, argument_shape: str,
):
    from agent.historical_tool_arguments import omit_historical_tool_arguments

    agent = _make_agent()
    fresh_query = "fresh-" + "x" * 800
    omitted = omit_historical_tool_arguments("z" * 900)
    blocked_call = _tool_call("call-history", omitted)
    if argument_shape == "decoded-direct":
        blocked_call = _tool_call("call-history", json.loads(omitted))
    elif argument_shape == "deferred-legacy":
        blocked_call = _tool_call(
            "call-history",
            json.dumps({"name": "web_search", "arguments": omitted}),
            name="tool_call",
        )
    elif argument_shape == "deferred-batch":
        blocked_call = _tool_call(
            "call-history",
            json.dumps({"calls": [{"name": "web_search", "arguments": omitted}]}),
            name="tool_call",
        )
    elif argument_shape == "decoded-deferred-batch":
        blocked_call = _tool_call(
            "call-history",
            {"calls": [{"name": "web_search", "arguments": json.loads(omitted)}]},
            name="tool_call",
        )
    assistant_message = SimpleNamespace(
        content="",
        tool_calls=[
            blocked_call,
            _tool_call("call-fresh", json.dumps({"query": fresh_query})),
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

    assert executed == [("web_search", {"query": fresh_query}, "call-fresh")]
    assert [message["tool_call_id"] for message in messages] == ["call-history", "call-fresh"]
    assert "non_replayable_history_arguments" in messages[0]["content"]
    assert fresh_query in messages[1]["content"]


def test_non_replayable_arguments_never_reach_a_real_registry_handler(tmp_path):
    from agent.historical_tool_arguments import omit_historical_tool_arguments
    from tools.registry import registry

    tool_name = "historical_argument_write_probe"
    schema = {
        "name": tool_name,
        "description": "Synthetic write probe",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    }
    executed_paths = []

    def write_probe(args, **_kwargs):
        executed_paths.append(args["path"])
        Path(args["path"]).write_text(args["content"], encoding="utf-8")
        return json.dumps({"ok": True})

    registry.register(name=tool_name, toolset="test", schema=schema, handler=write_probe)
    with (
        patch("model_tools.get_tool_definitions", return_value=[{"type": "function", "function": schema}]),
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
    agent.client = MagicMock()  # type: ignore[assignment]
    agent._flush_messages_to_session_db = MagicMock()
    fresh_path = tmp_path / "fresh.txt"
    fresh_content = "fresh-" + "x" * 800
    assistant_message = SimpleNamespace(
        content="",
        tool_calls=[
            _tool_call("call-history", omit_historical_tool_arguments("z" * 900), tool_name),
            _tool_call(
                "call-fresh",
                json.dumps({"path": str(fresh_path), "content": fresh_content}),
                tool_name,
            ),
        ],
    )
    messages = []

    with patch(
        "agent.tool_executor.maybe_persist_tool_result",
        side_effect=lambda **kwargs: kwargs["content"],
    ):
        agent._execute_tool_calls_sequential(assistant_message, messages, "task-1")

    assert executed_paths == [str(fresh_path)]
    assert fresh_path.read_text(encoding="utf-8") == fresh_content
    assert "non_replayable_history_arguments" in messages[0]["content"]


@pytest.mark.parametrize("dispatch_mode", ["sequential", "concurrent"])
def test_fresh_wide_decoded_arguments_are_not_mistaken_for_history(dispatch_mode: str):
    agent = _make_agent()
    fresh_args = {"query": "fresh", "items": list(range(10_001))}
    assistant_message = SimpleNamespace(
        content="",
        tool_calls=[_tool_call("call-fresh-wide", fresh_args)],
    )
    messages = []
    executed = []

    def fake_dispatch(name, args, task_id, *positional, **kwargs):
        call_id = kwargs.get("tool_call_id") or (positional[0] if positional else None)
        executed.append((name, args, call_id))
        return json.dumps({"ok": True})

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

    assert executed == [("web_search", fresh_args, "call-fresh-wide")]


@pytest.mark.parametrize("dispatch_mode", ["sequential", "concurrent"])
def test_deep_deferred_arguments_fail_without_blocking_a_valid_sibling(dispatch_mode: str):
    agent = _make_agent()
    too_deep = '{"query":' + "[" * 2_000 + "0" + "]" * 2_000 + "}"
    assistant_message = SimpleNamespace(
        content="",
        tool_calls=[
            _tool_call(
                "call-deep",
                json.dumps({"name": "web_search", "arguments": too_deep}),
                "tool_call",
            ),
            _tool_call("call-good", json.dumps({"query": "valid"})),
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
    assert [message["tool_call_id"] for message in messages] == ["call-deep", "call-good"]


@pytest.mark.parametrize("dispatch_mode", ["sequential", "concurrent"])
def test_decoded_connector_batch_reaches_per_entry_dispatch(dispatch_mode: str):
    from agent.historical_tool_arguments import omit_historical_tool_arguments

    agent = _make_agent()
    calls = [
        {
            "name": "connectors__gmail__SEND_EMAIL",
            "arguments": json.loads(omit_historical_tool_arguments("x" * 900)),
        },
        {
            "name": "connectors__slack__POST_MESSAGE",
            "arguments": {"body": "fresh"},
        },
    ]
    assistant_message = SimpleNamespace(
        content="",
        tool_calls=[_tool_call("call-connectors", {"calls": calls}, "tool_call")],
    )
    messages = []
    executed = []

    def fake_dispatch(name, args, task_id, *positional, **kwargs):
        call_id = kwargs.get("tool_call_id") or (positional[0] if positional else None)
        executed.append((name, args, call_id))
        return json.dumps({"ok": True})

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

    assert executed == [("tool_call", {"calls": calls}, "call-connectors")]
