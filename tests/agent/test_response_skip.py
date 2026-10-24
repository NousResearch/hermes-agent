"""Explicit messaging silence must terminate successfully, not recover as an empty reply."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from run_agent import AIAgent
from hermes_state import SessionDB


def _response(*, tool=False, content="", args=None):
    calls = [SimpleNamespace(id="skip-1", type="function", function=SimpleNamespace(
        name="skip_response", arguments=json.dumps(
            {"reason": "The user asked me to stop replying."} if args is None else args),
    ))] if tool else None
    return SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content=content, tool_calls=calls, reasoning_content=None),
        finish_reason="tool_calls" if tool else "stop",
    )], usage=None)


@pytest.fixture
def make_agent(monkeypatch, tmp_path):
    monkeypatch.setattr("hermes_cli.plugins.discover_plugins", lambda: None)
    monkeypatch.setattr("model_tools.get_tool_definitions", lambda **kw: [])
    monkeypatch.setattr("model_tools.check_toolset_requirements", lambda **kw: {})
    monkeypatch.setattr("agent.process_bootstrap.OpenAI", MagicMock())

    databases = []

    def make(platform="discord"):
        db = SessionDB(tmp_path / f"session-{len(databases)}.db")
        databases.append(db)
        agent = AIAgent(api_key="test-key", base_url="https://example.invalid/v1",
                        platform=platform, chat_id="chat-1", quiet_mode=True,
                        skip_context_files=True, skip_memory=True, session_db=db)
        agent._cached_system_prompt = "You are helpful."
        agent._use_prompt_caching = False
        agent.compression_enabled = False
        agent.save_trajectories = False
        agent.client = MagicMock()
        agent._disable_streaming = True
        return agent
    yield make
    for db in databases:
        db.close()


def test_skip_response_is_a_terminal_success_and_does_not_poison_next_turn(make_agent):
    agent = make_agent()
    progress = []
    agent.tool_progress_callback = lambda *args, **kw: progress.append(args)
    agent.client.chat.completions.create.side_effect = [
        _response(tool=True), _response(content="I am answering your new request."),
    ]
    result = agent.run_conversation("Stop replying. I am talking to another bot.")

    assert result["completed"] is True
    assert result["failed"] is False
    assert result["interrupted"] is False
    assert result["turn_exit_reason"] == "skip_response"
    assert result["final_response"] == ""
    assert result["api_calls"] == 1
    assert not [e for e in progress if len(e) > 1 and e[1] == "skip_response"]
    receipt = result["messages"][-1]
    assert receipt["role"] == "assistant"
    assert receipt["display_kind"] == "hidden"
    assert receipt["display_metadata"]["response_disposition"] == "skip_response"
    assert receipt["display_metadata"]["reason"] == "The user asked me to stop replying."
    # The real SQLite transcript must retain the receipt for crash recovery.
    persisted = agent._session_db.get_messages_as_conversation(agent.session_id)
    assert persisted[-1]["display_metadata"] == receipt["display_metadata"]

    second = agent.run_conversation("Now answer me.", conversation_history=result["messages"])
    assert second["completed"] is True
    assert second["final_response"] == "I am answering your new request."
    assert second["turn_exit_reason"].startswith("text_response")


@pytest.mark.asyncio
async def test_real_agent_skip_survives_gateway_turn_runner(make_agent, monkeypatch, tmp_path):
    import sys
    import run_agent
    from gateway.config import Platform
    from gateway.session import SessionSource
    from tests.gateway.test_run_cleanup_progress import CleanupCaptureAdapter, _make_runner, _install_fakes

    agent = make_agent()
    agent.client.chat.completions.create.return_value = _response(tool=True)
    adapter = CleanupCaptureAdapter(platform=Platform.DISCORD)
    runner = _make_runner(adapter)
    gateway_run = _install_fakes(monkeypatch, lambda **kw: agent, cleanup_on=False)
    monkeypatch.setitem(sys.modules, "run_agent", run_agent)
    class ExistingAgent(AIAgent):
        def __new__(cls, **kwargs):
            return agent

    monkeypatch.setattr(run_agent, "AIAgent", ExistingAgent)
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    result = await runner._run_agent(
        message="Stop replying.", context_prompt="", history=[],
        source=SessionSource(platform=Platform.DISCORD, chat_id="chat-1"),
        session_id=agent.session_id, session_key="agent:main:discord:group:chat-1",
    )
    assert result["final_response"] == ""
    assert result["turn_exit_reason"] == "skip_response"
    assert result["completed"] is True
    assert not adapter.sent


@pytest.mark.parametrize("platform", ["cli", "local", "api_server", "cron", "subagent"])
def test_skip_tool_is_not_advertised_outside_messaging_sessions(make_agent, platform):
    from tools.registry import registry

    agent = make_agent(platform)
    agent.client.chat.completions.create.return_value = _response(content="An ordinary answer.")
    assert agent.run_conversation("Please answer.")["final_response"] == "An ordinary answer."
    assert "skip_response" not in agent.valid_tool_names
    assert "skip_response" not in registry._tools


@pytest.mark.parametrize("args", [{}, {"reason": " "}, {"reason": 42}, {"reason": "x" * 501}])
def test_invalid_skip_does_not_silence_an_ordinary_answer(make_agent, args):
    agent = make_agent()
    agent.client.chat.completions.create.side_effect = [
        _response(tool=True, args=args), _response(content="An ordinary answer."),
    ]
    result = agent.run_conversation("Please answer.")
    assert result["final_response"] == "An ordinary answer."
    assert result["completed"] is True
    assert not agent._response_skip_reason


def test_skip_is_scoped_to_its_agent_not_another_profile(make_agent, tmp_path):
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override

    token = set_hermes_home_override(tmp_path / "profile-a")
    try:
        first = make_agent()
        first.max_iterations = 1
        first.client.chat.completions.create.return_value = _response(tool=True)
        skipped = first.run_conversation("Stay silent.")
        assert skipped["completed"] is True  # terminal success even on the final allowed iteration
        assert skipped["turn_exit_reason"] == "skip_response"
    finally:
        reset_hermes_home_override(token)

    token = set_hermes_home_override(tmp_path / "profile-b")
    try:
        other = make_agent()
        other._chat_id = "different-chat"
        other.client.chat.completions.create.return_value = _response(content="A different profile's answer.")
        result = other.run_conversation("Answer me.")
        assert result["final_response"] == "A different profile's answer."
        assert not other._response_skip_reason
    finally:
        reset_hermes_home_override(token)


def test_skip_in_a_mixed_tool_batch_is_not_terminal(make_agent):
    agent = make_agent()
    response = _response(tool=True)
    response.choices[0].message.tool_calls.append(SimpleNamespace(
        id="skip-2", type="function", function=SimpleNamespace(
            name="skip_response", arguments=json.dumps({"reason": "A second skip in the same batch."}),
        ),
    ))
    agent.client.chat.completions.create.side_effect = [response, _response(content="An ordinary answer.")]
    result = agent.run_conversation("Please answer.")
    assert result["final_response"] == "An ordinary answer."
    assert not agent._response_skip_reason
