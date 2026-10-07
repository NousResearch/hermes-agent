"""``auxiliary.voice_chat``: a voice turn runs on the voice model, the next turn on the main one.

Two real loopback providers; the agent talks HTTP to both, so the assertion is on what each
server actually received rather than on agent attributes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from run_agent import AIAgent
from tests.fakes.fake_llm_provider import FakeLLMServer, Text, write_hermes_home


def _agent(home: Path, main_url: str):
    return AIAgent(
        model="fake-model", provider="custom", base_url=main_url, api_key="sk-fake-e2e",
        quiet_mode=True, skip_context_files=True, skip_memory=True, enabled_toolsets=["memory"],
    )


@pytest.mark.parametrize("voice_window_ok", [True, False])
def test_voice_turn_routes_then_restores(tmp_path, monkeypatch, voice_window_ok):
    with FakeLLMServer([Text("main one"), Text("main two")]) as main, \
            FakeLLMServer([Text("voice reply")]) as voice:
        context = "128000" if voice_window_ok else "1000"
        home = write_hermes_home(tmp_path / ".hermes", main.base_url, extra_config=(
            "auxiliary:\n  voice_chat:\n    provider: custom\n"
            f"    base_url: {voice.base_url}\n    model: voice-model\n    api_key: sk-fake-voice\n"
            "custom_providers:\n  - name: voice\n"
            f"    base_url: {voice.base_url}\n    models:\n      voice-model:\n        context_length: {context}\n"
        ))
        monkeypatch.setenv("HERMES_HOME", str(home))
        agent = _agent(home, main.base_url)

        first = agent.run_conversation("typed question")
        agent._voice_turn_pending = True
        spoken = agent.run_conversation("spoken question", conversation_history=first["messages"])
        agent.run_conversation("typed again", conversation_history=spoken["messages"])

        main_models = [r["model"] for r in main.main_requests()]
        voice_models = [r["model"] for r in voice.main_requests()]
        if voice_window_ok:
            assert voice_models == ["voice-model"]
            # Unconfigured effort: the voice turn goes out with reasoning off, the main turns untouched.
            assert voice.main_requests()[0]["reasoning_effort"] == "none"
            assert {r["reasoning_effort"] for r in main.main_requests()} == {"medium"}
            assert main_models == ["fake-model", "fake-model"]
            assert spoken["model"] == "voice-model"
        else:  # too large for the voice model's window: the main model answers, nothing compacts
            assert voice_models == []
            assert main_models == ["fake-model"] * 3
        assert agent.model == "fake-model"
        assert agent.base_url.rstrip("/") == main.base_url.rstrip("/")
        assert agent._fallback_activated is False


@pytest.mark.parametrize("model,api_mode,expected", [
    ("gpt-6-astra", "codex_responses", "low"),        # Responses ladder has no "none"
    ("claude-opus-5-5", "anthropic_messages", "low"),  # mandatory thinking
    ("gpt-5.6-sol", "codex_responses", None),          # "none" is on its ladder: stays off
    ("claude-sonnet-4-6", "anthropic_messages", None),  # accepts thinking.type=disabled
])
def test_reasoning_off_falls_to_the_lowest_valid_level(model, api_mode, expected):
    from types import SimpleNamespace

    from agent.voice_turn_route import _voice_reasoning

    agent = SimpleNamespace(model=model, api_mode=api_mode, provider="custom",
                            base_url="https://api.openai.com/v1" if "gpt" in model else "https://api.anthropic.com")
    got = _voice_reasoning(agent, {"enabled": False})
    assert got == ({"enabled": True, "effort": expected} if expected else {"enabled": False})


def test_voice_usage_never_becomes_the_session_route(tmp_path):
    from hermes_state import SessionDB

    db = SessionDB(tmp_path / "state.db")
    db.create_session("s1", source="cli", model="main-model")
    db.update_token_counts("s1", input_tokens=10, output_tokens=5, model="voice-model",
                           billing_provider="voicep", api_call_count=1, task="voice_chat")
    row = db.get_session("s1")
    assert row["model"] == "main-model"
    assert row["input_tokens"] == 10
    assert db.auxiliary_usage_by_task("s1")["voice_chat"]["input_tokens"] == 10
    assert db.get_recent_session_model_route("s1") is None


# --- composed with adaptive reasoning: the voice turn must hand back the adaptive turn's own config ---

DEBUG = "Why does the gateway keep failing after I restart it? error: connection refused"


def _adaptive_agent(baseline, **policy):
    from types import SimpleNamespace

    from agent.adaptive_reasoning import init_adaptive_reasoning_state

    agent = SimpleNamespace(reasoning_config={"enabled": True, "effort": baseline}, provider="custom",
                            model="fake-model", api_mode="chat_completions", base_url="http://localhost/v1",
                            _cached_system_prompt="system", _voice_turn_pending=True, platform="cli")
    init_adaptive_reasoning_state(agent, {"enabled": True, "max_effort": "high", **policy})
    return agent


@pytest.fixture
def effort_only_voice(monkeypatch):
    monkeypatch.setattr("agent.auxiliary_task_config._get_auxiliary_task_config",
                        lambda task: {"provider": "auto", "reasoning_effort": "none"})


@pytest.mark.parametrize("baseline,message,policy,adjusted", [
    ("medium", DEBUG, {}, "high"),                        # escalation
    ("high", "thanks!", {"min_effort": "low"}, "low"),    # downshift
])
def test_voice_turn_inside_adaptive_turn_restores_baseline(effort_only_voice, baseline, message, policy, adjusted):
    from agent.adaptive_reasoning import adaptive_reasoning_turn
    from agent.voice_turn_route import begin_voice_turn_route, end_voice_turn_route

    agent = _adaptive_agent(baseline, **policy)
    with adaptive_reasoning_turn(agent, message):
        applied = agent.reasoning_config
        assert applied == {"enabled": True, "effort": adjusted}
        begin_voice_turn_route(agent, [], "system")
        assert agent.reasoning_config == {"enabled": False}
        end_voice_turn_route(agent)
        assert agent.reasoning_config is applied
    assert agent.reasoning_config == {"enabled": True, "effort": baseline}


def test_voice_turn_exception_exit_restores_adaptive_baseline(effort_only_voice):
    from agent.adaptive_reasoning import adaptive_reasoning_turn
    from agent.voice_turn_route import begin_voice_turn_route, end_voice_turn_route

    agent = _adaptive_agent("medium")
    with pytest.raises(RuntimeError), adaptive_reasoning_turn(agent, DEBUG):
        begin_voice_turn_route(agent, [], "system")
        try:
            raise RuntimeError("tool loop failed")
        finally:
            end_voice_turn_route(agent)
    assert agent.reasoning_config == {"enabled": True, "effort": "medium"}


def test_mid_turn_replacement_before_voice_survives(effort_only_voice):
    """A fallback that re-resolved effort before the voice turn bound is kept, not clobbered."""
    from agent.adaptive_reasoning import adaptive_reasoning_turn
    from agent.voice_turn_route import begin_voice_turn_route, end_voice_turn_route

    agent = _adaptive_agent("medium")
    with adaptive_reasoning_turn(agent, DEBUG):
        replaced = agent.reasoning_config = {"enabled": True, "effort": "xhigh"}
        begin_voice_turn_route(agent, [], "system")
        end_voice_turn_route(agent)
    assert agent.reasoning_config is replaced


def test_voice_restores_a_copy_when_the_saved_config_was_mutated(effort_only_voice):
    from agent.voice_turn_route import begin_voice_turn_route, end_voice_turn_route

    agent = _adaptive_agent("medium")
    saved = agent.reasoning_config
    begin_voice_turn_route(agent, [], "system")
    saved["effort"] = "low"
    end_voice_turn_route(agent)
    assert agent.reasoning_config == {"enabled": True, "effort": "medium"}


@pytest.mark.parametrize("routed", [False, True])
def test_real_voice_turn_after_adaptive_escalation_next_turn_sends_baseline(tmp_path, monkeypatch, routed):
    """Effort-only and explicit voice-model routes: the escalated level never leaks to the next typed turn."""
    with FakeLLMServer([Text("first answer"), Text("second answer")]) as main, \
            FakeLLMServer([Text("voice reply")]) as voice:
        extra = ("auxiliary:\n  voice_chat:\n    provider: custom\n"
                 f"    base_url: {voice.base_url}\n    model: voice-model\n    api_key: sk-fake-e2e\n"
                 "custom_providers:\n  - name: voice\n"
                 f"    base_url: {voice.base_url}\n    models:\n      voice-model:\n        context_length: 128000\n"
                 ) if routed else ""
        home = write_hermes_home(tmp_path / ".hermes", main.base_url, extra_config=extra)
        monkeypatch.setenv("HERMES_HOME", str(home))
        agent = AIAgent(model="fake-model", provider="custom", base_url=main.base_url, api_key="sk-fake-e2e",
                        quiet_mode=True, skip_context_files=True, skip_memory=True, enabled_toolsets=["memory"],
                        reasoning_config={"enabled": True, "effort": "medium"},
                        adaptive_reasoning={"enabled": True, "max_effort": "high"})
        agent._voice_turn_pending = True
        first = agent.run_conversation(DEBUG)
        assert agent.reasoning_config == {"enabled": True, "effort": "medium"}
        agent.run_conversation("An ambiguous follow-up", conversation_history=first["messages"])
        spoken = (voice if routed else main).main_requests()[0]
        assert spoken["reasoning_effort"] == "none"
        assert main.main_requests()[-1]["reasoning_effort"] == "medium"
        assert agent.model == "fake-model"
