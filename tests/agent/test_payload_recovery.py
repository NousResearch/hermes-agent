"""Recovery contracts for invalid and incomplete tool-call payloads."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent


def _tool_defs(*names: str) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"{name} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for name in names
    ]


def _tool_call(name: str, arguments: str, call_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _response(*tool_calls: SimpleNamespace, content: str = "", finish_reason: str = "tool_calls") -> SimpleNamespace:
    message = SimpleNamespace(content=content, tool_calls=list(tool_calls))
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish_reason)],
        model="test/model",
        usage=None,
    )


def _agent(*tool_names: str) -> AIAgent:
    with (
        patch("model_tools.get_tool_definitions", return_value=_tool_defs(*tool_names)),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("hermes_cli.config.load_config", return_value={}),
        patch("hermes_cli.config.load_config_readonly", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            max_iterations=8,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    agent._persist_session = lambda *args, **kwargs: None
    agent._flush_messages_to_session_db = lambda *args, **kwargs: True
    return agent


@pytest.mark.parametrize(
    ("bad_name", "bad_arguments"),
    [
        ("write_file", '{"path":"out.txt","content":"cut'),
        ("execute_code", json.dumps({"code": 'payload = """cut'})),
        ("execute_code", json.dumps({"code": "print('cut')\u0000"})),
        ("execute_code", json.dumps({"code": "print(" + chr(0xD800) + ")"})),
        (
            "tool_call",
            json.dumps({"name": "execute_code", "arguments": {"code": "if True:\n"}}),
        ),
        (
            "tool_call",
            json.dumps({
                "name": "execute_code",
                "arguments": json.dumps({"code": "if True:\n"}),
            }),
        ),
    ],
)
def test_incomplete_payload_recovers_without_partial_batch_side_effects(
    bad_name, bad_arguments, monkeypatch
):
    agent = _agent("execute_code", "tool_call", "write_file")
    bad_batch = _response(
        _tool_call(bad_name, bad_arguments, "bad"),
        _tool_call("write_file", json.dumps({"path": "must-not-run", "content": "x"}), "sibling"),
    )
    recovered = _response(
        _tool_call("write_file", json.dumps({"path": "complete", "content": "好" * 20_000}), "good")
    )
    final = _response(content="done", finish_reason="stop")
    responses = iter((bad_batch, recovered, final))
    requests = []
    dispatched = []

    def _api_call(api_kwargs):
        requests.append(api_kwargs)
        return next(responses)

    agent._interruptible_api_call = _api_call

    def _dispatch(name, args, *positional, **kwargs):
        dispatched.append((name, args))
        return json.dumps({"success": True})

    monkeypatch.setattr("model_tools.handle_function_call", _dispatch)
    with patch("tools.terminal_tool._get_env_config", return_value={"env_type": "local"}), patch(
        "tools.code_execution_tool._get_execution_mode", return_value="strict"
    ), patch(
        "tools.tool_search.load_config_readonly",
        return_value=SimpleNamespace(effective_defer_tools=frozenset({"execute_code"})),
    ), patch("tools.tool_search.is_deferrable_tool_name", return_value=True), patch(
        "tools.tool_search.validate_deferred_call_args", return_value=None
    ), patch("agent.tool_executor._tool_search_scoped_names", return_value={"execute_code"}):
        result = agent.run_conversation("write it")

    assert result["completed"] is True
    assert result["final_response"] == "done"
    assert [(name, args["path"]) for name, args in dispatched] == [("write_file", "complete")]
    assert len(requests) == 3
    assert [m["content"] for m in result["messages"] if m["role"] == "user"] == ["write it"]


def test_repeated_incomplete_payloads_stop_after_three_requests(monkeypatch):
    agent = _agent("execute_code", "write_file")
    requests = []
    dispatched = []

    def _api_call(api_kwargs):
        requests.append(api_kwargs)
        return _response(
            _tool_call("execute_code", json.dumps({"code": "def unfinished("}), f"bad-{len(requests)}"),
            _tool_call("write_file", json.dumps({"path": "must-not-run", "content": "x"}), f"side-{len(requests)}"),
        )

    agent._interruptible_api_call = _api_call
    monkeypatch.setattr(
        "model_tools.handle_function_call",
        lambda name, args, *positional, **kwargs: dispatched.append((name, args)) or json.dumps({"success": True}),
    )
    with patch("tools.terminal_tool._get_env_config", return_value={"env_type": "local"}), patch(
        "tools.code_execution_tool._get_execution_mode", return_value="strict"
    ):
        result = agent.run_conversation("run it")

    assert result["completed"] is False
    assert result["partial"] is True
    assert "invalid or incomplete tool-call payloads" in result["error"]
    assert len(requests) == 3
    assert dispatched == []


def test_python_preflight_ignores_unknown_call_and_dispatches_valid_sibling(monkeypatch):
    agent = _agent("tool_call", "write_file")
    responses = iter(
        (
            _response(
                _tool_call("execute_code", json.dumps({"code": "if True:"}), "unknown"),
                _tool_call("write_file", json.dumps({"path": "valid", "content": "x"}), "valid"),
            ),
            _response(content="done", finish_reason="stop"),
        )
    )
    dispatched = []
    agent._interruptible_api_call = lambda api_kwargs: next(responses)
    monkeypatch.setattr(
        "model_tools.handle_function_call",
        lambda name, args, *positional, **kwargs: dispatched.append((name, args))
        or json.dumps({"success": True}),
    )

    with patch("tools.terminal_tool._get_env_config", return_value={"env_type": "local"}), patch(
        "tools.code_execution_tool._get_execution_mode", return_value="strict"
    ):
        result = agent.run_conversation("write it")

    assert result["completed"] is True
    assert [(name, args["path"]) for name, args in dispatched] == [("write_file", "valid")]
    unknown_result = next(
        message for message in result["messages"]
        if message["role"] == "tool" and message["tool_call_id"] == "unknown"
    )
    assert "does not exist" in unknown_result["content"]


def test_python_preflight_defers_to_remote_interpreter(monkeypatch):
    agent = _agent("execute_code")
    responses = iter(
        (
            _response(_tool_call("execute_code", json.dumps({"code": "def remote_only("}), "remote")),
            _response(content="remote handled it", finish_reason="stop"),
        )
    )
    dispatched = []
    agent._interruptible_api_call = lambda api_kwargs: next(responses)
    monkeypatch.setattr(
        "model_tools.handle_function_call",
        lambda name, args, *positional, **kwargs: dispatched.append((name, args))
        or json.dumps({"error": "remote compiler result"}),
    )

    with patch("tools.terminal_tool._get_env_config", return_value={"env_type": "ssh"}):
        result = agent.run_conversation("run remotely")

    assert result["completed"] is True
    assert [name for name, _ in dispatched] == ["execute_code"]


def test_python_preflight_does_not_block_out_of_scope_bridge_sibling(monkeypatch):
    agent = _agent("tool_call", "write_file")
    responses = iter(
        (
            _response(
                _tool_call(
                    "tool_call",
                    json.dumps({"name": "execute_code", "arguments": {"code": "if True:"}}),
                    "scoped-out",
                ),
                _tool_call("write_file", json.dumps({"path": "valid", "content": "x"}), "valid"),
            ),
            _response(content="done", finish_reason="stop"),
        )
    )
    dispatched = []
    agent._interruptible_api_call = lambda api_kwargs: next(responses)
    monkeypatch.setattr("agent.tool_executor._tool_search_scoped_names", lambda agent: frozenset())
    monkeypatch.setattr(
        "model_tools.handle_function_call",
        lambda name, args, *positional, **kwargs: dispatched.append((name, args))
        or json.dumps({"success": True}),
    )

    with patch("tools.terminal_tool._get_env_config", return_value={"env_type": "local"}), patch(
        "tools.code_execution_tool._get_execution_mode", return_value="strict"
    ), patch(
        "tools.tool_search.load_config_readonly",
        return_value=SimpleNamespace(effective_defer_tools=frozenset({"execute_code"})),
    ), patch(
        "tools.tool_search.is_deferrable_tool_name", return_value=True
    ):
        result = agent.run_conversation("write it")

    assert result["completed"] is True
    assert [name for name, _ in dispatched] == ["write_file"]
