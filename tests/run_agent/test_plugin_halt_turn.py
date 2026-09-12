"""Runtime tests for the plugin ``halt_turn`` directive (#107665)."""

import json
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from run_agent import AIAgent


def _make_tool_defs(*names: str) -> list[dict]:
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


def _mock_tool_call(name="web_search", arguments="{}", call_id=None):
    return SimpleNamespace(
        id=call_id or f"call_{uuid.uuid4().hex[:8]}",
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _mock_response(content="Hello", finish_reason="stop", tool_calls=None):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=msg, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


def _make_agent(
    *tool_names: str,
    max_iterations: int = 10,
    config: dict | None = None,
    platform: str | None = None,
) -> AIAgent:
    with (
        patch("model_tools.get_tool_definitions", return_value=_make_tool_defs(*tool_names)),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("hermes_cli.config.load_config", return_value=config or {}),
        patch("hermes_cli.config.load_config_readonly", return_value=config or {}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            max_iterations=max_iterations,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            platform=platform or "cli",
        )
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    agent._disable_streaming = True
    return agent


def _tool_only_responses(n: int = 9, tool_name: str = "web_search"):
    return [
        _mock_response(
            content="",
            finish_reason="tool_calls",
            tool_calls=[_mock_tool_call(tool_name, "{}", f"c{i}")],
        )
        for i in range(1, n + 1)
    ]


def _invoke_for_hook(target_hook: str, payload):
    def _invoke(hook_name, **_kwargs):
        if hook_name == target_hook:
            return [payload]
        return []

    return _invoke


def _run_conversation(agent: AIAgent, **extra_patches):
    patches = [
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ]
    for target, kwargs in extra_patches.items():
        patches.append(patch(target, **kwargs))
    from contextlib import ExitStack

    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        return agent.run_conversation("search once")


def test_post_tool_call_halt_ends_turn_and_streams_response():
    agent = _make_agent("web_search", max_iterations=10)
    agent.client.chat.completions.create.side_effect = _tool_only_responses()
    deltas: list = []
    agent.stream_delta_callback = lambda d: deltas.append(d)

    result = _run_conversation(
        agent,
        **{
            "hermes_cli.lifecycle.invoke_hook": {
                "side_effect": _invoke_for_hook(
                    "post_tool_call",
                    {"action": "halt_turn", "response": "stopped by plugin"},
                )
            },
            "hermes_cli.lifecycle.has_hook": {"return_value": True},
            "model_tools.handle_function_call": {"return_value": json.dumps({"ok": True})},
        },
    )

    assert result["turn_exit_reason"] == "plugin_halt"
    assert result["final_response"] == "stopped by plugin"
    text_deltas = [d for d in deltas if isinstance(d, str)]
    assert "stopped by plugin" in text_deltas, (
        f"halt message was never streamed; callback only saw {deltas!r}"
    )
    assert agent.client.chat.completions.create.call_count == 1


def test_pre_tool_call_halt_skips_tool_and_ends_turn():
    agent = _make_agent("web_search", max_iterations=10)
    agent.client.chat.completions.create.side_effect = _tool_only_responses()
    deltas: list = []
    agent.stream_delta_callback = lambda d: deltas.append(d)

    with patch("model_tools.handle_function_call", return_value="SHOULD_NOT_RUN") as mock_hfc:
        result = _run_conversation(
            agent,
            **{
                "hermes_cli.lifecycle.invoke_hook": {
                    "side_effect": _invoke_for_hook(
                        "pre_tool_call",
                        {"action": "halt_turn", "response": "halt now"},
                    )
                },
            },
        )

    mock_hfc.assert_not_called()
    assert result["turn_exit_reason"] == "plugin_halt"
    assert result["final_response"] == "halt now"
    text_deltas = [d for d in deltas if isinstance(d, str)]
    assert "halt now" in text_deltas, (
        f"halt message was never streamed; callback only saw {deltas!r}"
    )
    assert agent.client.chat.completions.create.call_count == 1


def test_fail_open_empty_halt_response_continues_turn():
    agent = _make_agent("web_search", max_iterations=10)
    agent.client.chat.completions.create.side_effect = [
        *_tool_only_responses(1),
        _mock_response(content="done", finish_reason="stop", tool_calls=None),
    ]

    result = _run_conversation(
        agent,
        **{
            "hermes_cli.lifecycle.invoke_hook": {
                "side_effect": _invoke_for_hook(
                    "post_tool_call",
                    {"action": "halt_turn", "response": ""},
                )
            },
            "hermes_cli.lifecycle.has_hook": {"return_value": True},
            "model_tools.handle_function_call": {"return_value": json.dumps({"ok": True})},
        },
    )

    assert result["turn_exit_reason"] != "plugin_halt"
    assert result["turn_exit_reason"].startswith("text_response")
    assert result["final_response"] == "done"


def test_fail_open_block_is_not_plugin_halt():
    agent = _make_agent("web_search", max_iterations=10)
    agent.client.chat.completions.create.side_effect = [
        *_tool_only_responses(1),
        _mock_response(content="done", finish_reason="stop", tool_calls=None),
    ]

    with patch("model_tools.handle_function_call", return_value="SHOULD_NOT_RUN") as mock_hfc:
        result = _run_conversation(
            agent,
            **{
                "hermes_cli.lifecycle.invoke_hook": {
                    "side_effect": _invoke_for_hook(
                        "pre_tool_call",
                        {"action": "block", "message": "nope"},
                    )
                },
            },
        )

    mock_hfc.assert_not_called()
    assert result["turn_exit_reason"] != "plugin_halt"
    assert result["turn_exit_reason"].startswith("text_response")
    tool_contents = [m["content"] for m in result["messages"] if m.get("role") == "tool"]
    assert any("nope" in content for content in tool_contents)


def test_post_tool_call_string_return_does_not_replace_tool_result(monkeypatch):
    """Thin wrapper: observational ``post_tool_call`` must not rewrite the tool result."""
    from tests.test_transform_tool_result_hook import test_post_tool_call_remains_observational

    test_post_tool_call_remains_observational(monkeypatch)
