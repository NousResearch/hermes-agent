"""Regression for #156822: incidental turn routes must not change thinking.

Loopback requests prove parameter/prefix continuity, not provider cache hits.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent.anthropic_adapter import build_anthropic_kwargs
from agent.voice_turn_route import begin_voice_turn_route, end_voice_turn_route
from hermes_cli.config import load_config_readonly
from run_agent import AIAgent
from tests.fakes.fake_llm_provider import FakeLLMServer, Text, write_hermes_home


def _agent(url, reasoning):
    return AIAgent(
        model="fake-model", provider="custom", base_url=url, api_key="test-key",
        reasoning_config=reasoning, max_tokens=4096, quiet_mode=True, skip_context_files=True,
        skip_memory=True, enabled_toolsets=["memory"],
    )


@pytest.mark.parametrize("reasoning", [None, {"enabled": True, "effort": "high"}, {"enabled": False}])
@pytest.mark.parametrize("config", ["", 'auxiliary:\n  voice_chat:\n    reasoning_effort: ""\n'])
def test_alternating_voice_text_keeps_request_settings(tmp_path, monkeypatch, reasoning, config):
    with FakeLLMServer([Text("answer") for _ in range(5)]) as server:
        home = write_hermes_home(tmp_path / ".hermes", server.base_url, extra_config=config)
        monkeypatch.setenv("HERMES_HOME", str(home))
        agent = _agent(server.base_url, reasoning)
        history = None
        for voice in (False, True, False, True, False):
            agent._voice_turn_pending = voice
            result = agent.run_conversation("question", conversation_history=history)
            assert result["completed"]
            history = result["messages"]
        requests = server.main_requests()
        assert len(requests) == 5
        for key in ("model", "reasoning_effort", "temperature", "tools"):
            assert all(request.get(key) == requests[0].get(key) for request in requests), key
        systems = [[message for message in request["messages"] if message["role"] == "system"] for request in requests]
        assert all(system == systems[0] for system in systems)
        assert agent.reasoning_config == reasoning


@pytest.mark.parametrize("reasoning", [None, {"enabled": True, "effort": "high"}, {"enabled": False}])
def test_thinking_only_continuation_keeps_session_reasoning(tmp_path, monkeypatch, reasoning):
    script = [Text("", reasoning="unfinished reasoning", finish_reason="length"), Text("finished")]
    with FakeLLMServer(script) as server:
        home = write_hermes_home(tmp_path / ".hermes", server.base_url)
        monkeypatch.setenv("HERMES_HOME", str(home))
        agent = _agent(server.base_url, reasoning)
        result = agent.run_conversation("question")
        assert result["completed"]
        requests = server.main_requests()
        assert len(requests) == 2
        assert requests[1].get("reasoning_effort") == requests[0].get("reasoning_effort")
        assert requests[1]["max_tokens"] > requests[0]["max_tokens"]
        assert agent.reasoning_config == reasoning


@pytest.mark.parametrize("model", ["claude-sonnet-4-6", "claude-opus-5-5"])
def test_default_voice_keeps_anthropic_thinking_namespace(tmp_path, monkeypatch, model):
    home = write_hermes_home(tmp_path / ".hermes", "http://127.0.0.1:9/v1")
    monkeypatch.setenv("HERMES_HOME", str(home))
    # Exercise the real merged defaults and voice route, then the native wire builder.
    assert load_config_readonly()["auxiliary"]["voice_chat"]
    agent = SimpleNamespace(
        provider="anthropic", model=model, base_url="https://api.anthropic.com",
        api_mode="anthropic_messages", reasoning_config={"enabled": True, "effort": "high"},
        _voice_turn_pending=False, _cached_system_prompt="stable system",
    )

    def wire():
        return build_anthropic_kwargs(
            model=model, messages=[{"role": "user", "content": "question"}], tools=None,
            max_tokens=4096, reasoning_config=agent.reasoning_config, is_oauth=False,
        )

    text = wire()
    agent._voice_turn_pending = True
    begin_voice_turn_route(agent, [{"role": "user", "content": "question"}], "stable system")
    voice = wire()
    end_voice_turn_route(agent)
    next_text = wire()
    for key in ("thinking", "output_config", "max_tokens", "temperature"):
        assert text.get(key) == voice.get(key) == next_text.get(key), key
