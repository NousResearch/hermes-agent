"""``agent.tool_choice: required`` — wire-level tool enforcement with a respond exit.

With ``required`` every chat_completions request that carries tools sends
``tool_choice="required"`` plus a ``respond`` tool, and a respond-only reply comes back out of
the turn as an ordinary text answer (final_response, history), so delivery is unchanged.
Default ``auto`` changes nothing.
"""

import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.chat_completion_helpers import build_api_kwargs
from agent.tool_choice import RESPOND_TOOL_NAME, unwrap_respond_call, with_respond_tool
from agent.transports.types import NormalizedResponse, build_tool_call
from run_agent import AIAgent

_MSGS = [{"role": "user", "content": "hi"}]


@pytest.fixture(autouse=True)
def _disable_background_titles(monkeypatch):
    monkeypatch.setattr("agent.title_generator.maybe_auto_title", lambda *args, **kwargs: None)


def _tool_defs(*names):
    return [
        {"type": "function", "function": {
            "name": n, "description": f"{n} tool", "parameters": {"type": "object", "properties": {}}}}
        for n in names
    ]


def _agent(tool_choice="required"):
    hermes_home = Path(tempfile.mkdtemp(prefix="hermes-test-home-"))
    (hermes_home / "logs").mkdir(parents=True, exist_ok=True)
    with (
        patch("model_tools.get_tool_definitions", return_value=_tool_defs("terminal")),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
        patch("run_agent._hermes_home", hermes_home),
        patch("agent.model_metadata.fetch_model_metadata", return_value={}),
    ):
        agent = AIAgent(
            api_key="test-key", base_url="https://openrouter.ai/api/v1",
            quiet_mode=True, skip_context_files=True, skip_memory=True,
        )
    agent._tool_choice = tool_choice
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    return agent


def _call(name, arguments="{}", call_id="c1"):
    return SimpleNamespace(id=call_id, type="function", function=SimpleNamespace(name=name, arguments=arguments))


def _response(content="", tool_calls=None, finish_reason="tool_calls"):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason=finish_reason)],
                           model="test/model", usage=None)


def _tool_names(kwargs):
    return [t["function"]["name"] for t in kwargs.get("tools") or []]


# --- request side --------------------------------------------------------------------------

def test_required_sends_tool_choice_and_respond_tool():
    kwargs = build_api_kwargs(_agent(), _MSGS)
    assert kwargs["tool_choice"] == "required"
    assert _tool_names(kwargs) == ["terminal", RESPOND_TOOL_NAME]


def test_auto_default_changes_nothing():
    kwargs = build_api_kwargs(_agent("auto"), _MSGS)
    assert "tool_choice" not in kwargs
    assert RESPOND_TOOL_NAME not in _tool_names(kwargs)


def test_config_value_is_case_and_space_tolerant():
    assert build_api_kwargs(_agent(" Required "), _MSGS)["tool_choice"] == "required"


def test_no_tools_sends_nothing():
    agent = _agent()
    agent.tools = []
    kwargs = build_api_kwargs(agent, _MSGS)
    assert "tool_choice" not in kwargs
    assert not kwargs.get("tools")


def test_explicit_request_override_wins():
    agent = _agent()
    agent.request_overrides = {"tool_choice": "auto"}
    kwargs = build_api_kwargs(agent, _MSGS)
    assert kwargs["tool_choice"] == "auto"
    assert RESPOND_TOOL_NAME not in _tool_names(kwargs)


@pytest.mark.parametrize("platform, expected", [
    ("telegram", True), ("cli", False), (None, False), ("cron", False), ("discord", True),
])
def test_per_platform_mapping(platform, expected):
    from agent.tool_choice import tool_choice_required
    agent = _agent({"telegram": "required", "discord": "Required", "default": "auto"})
    agent.platform = platform
    assert tool_choice_required(agent) is expected


def test_per_platform_mapping_default_applies_to_unlisted_platforms():
    from agent.tool_choice import tool_choice_required
    agent = _agent({"cli": "auto", "default": "required"})
    agent.platform = "slack"
    assert tool_choice_required(agent) is True
    agent.platform = None  # no platform = cli
    assert tool_choice_required(agent) is False


def test_non_chat_completions_modes_are_untouched():
    agent = _agent()
    agent.api_mode = "anthropic_messages"
    from agent.tool_choice import tool_choice_required
    assert tool_choice_required(agent) is False


def test_respond_tool_is_appended_once():
    tools = with_respond_tool(_tool_defs("terminal"))
    assert with_respond_tool(tools) is tools


def test_config_default_is_auto_and_plumbed():
    from hermes_cli.config_defaults import DEFAULT_CONFIG
    assert DEFAULT_CONFIG["agent"]["tool_choice"] == "auto"
    from agent.agent_init import _apply_agent_section
    agent = SimpleNamespace(run_budget_seconds=0)
    with patch("agent.agent_init.normalize_budget_warning_ratio", return_value=0.7):
        try:
            _apply_agent_section(agent, {"agent": {"tool_choice": "required"}})
        except Exception:
            pass  # later sections need a full agent; the field is set before them
    assert agent._tool_choice == "required"


# --- response side -------------------------------------------------------------------------

def _normalized(content, *calls):
    return NormalizedResponse(content=content, tool_calls=list(calls) or None, finish_reason="tool_calls")


def test_respond_only_becomes_plain_text():
    msg = _normalized("", build_tool_call("c1", RESPOND_TOOL_NAME, {"message": "Installed gqrx."}))
    assert unwrap_respond_call(msg) is True
    assert (msg.content, msg.tool_calls, msg.finish_reason) == ("Installed gqrx.", None, "stop")


def test_respond_message_wins_over_narration_content():
    msg = _normalized("I will reply now.", build_tool_call("c1", RESPOND_TOOL_NAME, {"message": "Done."}))
    unwrap_respond_call(msg)
    assert msg.content == "Done."


def test_malformed_respond_arguments_fall_back_to_raw_text():
    msg = _normalized("", build_tool_call("c1", RESPOND_TOOL_NAME, "not json"))
    unwrap_respond_call(msg)
    assert msg.content == "not json"


def test_respond_mixed_with_real_tools_is_dropped():
    terminal = build_tool_call("c1", "terminal", {"command": "ls"})
    msg = _normalized("", terminal, build_tool_call("c2", RESPOND_TOOL_NAME, {"message": "early"}))
    assert unwrap_respond_call(msg) is False
    assert msg.tool_calls == [terminal]


def test_plain_tool_calls_untouched():
    terminal = build_tool_call("c1", "terminal", {"command": "ls"})
    msg = _normalized("", terminal)
    assert unwrap_respond_call(msg) is False
    assert msg.tool_calls == [terminal]


# --- full turn -----------------------------------------------------------------------------

def _run(agent, message="install it"):
    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        return agent.run_conversation(message)


def test_turn_ends_with_respond_message_as_final_response():
    agent = _agent()
    agent.client.chat.completions.create.return_value = _response(
        tool_calls=[_call(RESPOND_TOOL_NAME, '{"message": "All set."}')])
    result = _run(agent)
    assert result["final_response"] == "All set."
    assert agent.client.chat.completions.create.call_count == 1  # no dropped-tool-call retries
    sent = agent.client.chat.completions.create.call_args.kwargs
    assert sent["tool_choice"] == "required"
    assert RESPOND_TOOL_NAME in _tool_names(sent)
    last = result["messages"][-1]
    assert last["role"] == "assistant" and last["content"] == "All set." and not last.get("tool_calls")


def test_real_tools_run_before_the_respond_reply():
    agent = _agent()
    agent.client.chat.completions.create.side_effect = [
        _response(tool_calls=[_call("terminal", '{"command": "brew install gqrx"}', "c1"),
                              _call(RESPOND_TOOL_NAME, '{"message": "too early"}', "c2")]),
        _response(tool_calls=[_call(RESPOND_TOOL_NAME, '{"message": "Installed."}', "c3")]),
    ]
    executed = []

    def _fake_execute(assistant_message, messages, effective_task_id, api_call_count=0):
        executed.extend(tc.function.name for tc in assistant_message.tool_calls)
        from agent.tool_dispatch_helpers import make_tool_result_message
        messages.append(make_tool_result_message("terminal", "ok", "c1"))

    with patch.object(agent, "_execute_tool_calls", side_effect=_fake_execute):
        result = _run(agent)
    assert executed == ["terminal"]
    assert result["final_response"] == "Installed."
    assert agent.client.chat.completions.create.call_count == 2
