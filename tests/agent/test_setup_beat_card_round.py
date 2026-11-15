"""A setup_beat round in the turn loop. It never ends in the empty-response nudge: the line written with
the card is the turn's reply when the follow-up comes back empty (agent/turn_tool_round.py
``_LINE_IS_REPLY_TOOLS``). A card the user wrote past ends the turn on its result."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent import onboarding_record
from agent.agent_init import _CALLBACK_PARAMS
from agent.onboarding_flow import Facts
from agent.turn_empty_response import recover_empty_response
from agent.turn_tool_round import stage_tool_call_message
from hermes_state import SessionDB
from run_agent import AIAgent
from tests.agent.test_run_agent import _mock_response
from tools.setup_beat_tool import SETUP_BEAT_SCHEMA


def _agent():
    agent = AIAgent.__new__(AIAgent)
    agent.provider, agent.model, agent.session_id, agent.tools = "nous", "m", "s1", []
    agent.log_prefix, agent.quiet_mode, agent.verbose_logging = "", True, False
    agent._mute_post_response, agent._base_url, agent.api_mode = False, "", "chat_completions"
    agent._current_streamed_assistant_text = ""
    agent._stream_callback = None
    for name in _CALLBACK_PARAMS:
        setattr(agent, name, None)
    return agent


def _call(name: str):
    return SimpleNamespace(id=f"call_{name}", type="function",
                           function=SimpleNamespace(name=name, arguments="{}"))


@pytest.mark.parametrize(("tool", "reused"), [("setup_beat", True), ("terminal", False)])
def test_an_empty_follow_up_after_a_card_round_keeps_the_cards_line(tool, reused):
    agent = _agent()
    messages = [{"role": "user", "content": "hi"}]
    staged, _ = stage_tool_call_message(
        agent, assistant_message=SimpleNamespace(content="Pick a colour whenever you're back.",
                                                 tool_calls=[_call(tool)], reasoning=None),
        finish_reason="tool_calls", messages=messages)
    messages += [staged, {"role": "tool", "tool_call_id": f"call_{tool}", "content": "{}"}]

    verdict = recover_empty_response(
        agent, SimpleNamespace(content="", tool_calls=None, reasoning=None), None, "stop", final_response="",
        messages=messages, api_messages=[], conversation_history=[], active_system_prompt="",
        api_call_count=2, turn_exit_reason=None, preflight_compression_blocked=False)

    if reused:
        assert verdict.action == "break" and verdict.final_response == "Pick a colour whenever you're back."
        assert verdict.turn_exit_reason == "fallback_prior_turn_content"
    else:
        assert verdict.final_response != "Pick a colour whenever you're back."
    assert agent._mute_post_response is False


def _loop_agent(db):
    with (patch("model_tools.get_tool_definitions", return_value=[{"type": "function", "function": SETUP_BEAT_SCHEMA}]),
          patch("model_tools.check_toolset_requirements", return_value={}),
          patch("agent.process_bootstrap.OpenAI")):
        agent = AIAgent(api_key="test-key-1234567890", base_url="https://openrouter.ai/api/v1", platform="desktop",
                        quiet_mode=True, skip_context_files=True, skip_memory=True)
    agent.client = MagicMock()
    agent._disable_streaming, agent._cached_system_prompt, agent._use_prompt_caching = True, "You are helpful.", False
    agent.tool_delay, agent.compression_enabled, agent.save_trajectories = 0, False, False
    db.create_session(agent.session_id, "desktop")
    onboarding_record.start(db, agent.session_id, Facts(os="macos"))
    agent._session_db = db
    return agent


def test_a_card_closed_by_a_message_ends_the_turn_without_another_model_call(tmp_path, monkeypatch):
    """The message runs as the user's own next turn, so the old turn must not go on to answer it."""
    monkeypatch.setattr("hermes_cli.onboarding_run.set_run", lambda *a, **k: None)
    db = SessionDB(db_path=tmp_path / "state.db")
    agent = _loop_agent(db)
    agent.setup_beat_callback = lambda payload: {"wrote_instead": True}
    calls = []

    def model_call(api_kwargs):
        calls.append(api_kwargs)
        if len(calls) > 1:
            return _mock_response(content="Elite puts a terminal beside the chat.", finish_reason="stop")
        card = SimpleNamespace(id="c1", type="function",
                               function=SimpleNamespace(name="setup_beat", arguments='{"beat": "layout"}'))
        return _mock_response(content="Which one suits you?", finish_reason="tool_calls", tool_calls=[card])

    agent._interruptible_api_call = model_call
    with (patch.object(agent, "_flush_messages_to_session_db"), patch.object(agent, "_persist_session"),
          patch.object(agent, "_save_trajectory"), patch.object(agent, "_cleanup_task_resources")):
        result = agent.run_conversation("[opening]")

    assert len(calls) == 1
    tool = next(m for m in result["messages"] if m["role"] == "tool")
    assert json.loads(tool["content"])["wrote_instead"] is True
    # The tool tail is closed, so the user's message that follows alternates roles.
    assert result["interrupted"] is True and result["messages"][-1]["role"] == "assistant"
