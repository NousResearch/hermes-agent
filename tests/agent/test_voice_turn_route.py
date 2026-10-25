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


@pytest.mark.parametrize("separate_voice_model", [False, True])
def test_mid_voice_turn_reports_the_sessions_own_route(tmp_path, monkeypatch, separate_voice_model):
    """Desktop adopts session.info's effort/model into the composer that seeds new chats: a live
    voice turn's reasoning-off (and voice model) must never be reported or persisted as the chat's."""
    from agent.voice_turn_route import begin_voice_turn_route, end_voice_turn_route
    from tui_gateway.server import _runtime_model_config, _session_info

    with FakeLLMServer([]) as main, FakeLLMServer([]) as voice:
        extra = ("auxiliary:\n  voice_chat:\n    provider: custom\n"
                 f"    base_url: {voice.base_url}\n    model: voice-model\n    api_key: sk-fake-voice\n"
                 if separate_voice_model else "")
        home = write_hermes_home(tmp_path / ".hermes", main.base_url, extra_config=extra)
        monkeypatch.setenv("HERMES_HOME", str(home))
        agent = _agent(home, main.base_url)
        agent.reasoning_config = {"enabled": True, "effort": "low"}
        agent._voice_turn_pending = True
        begin_voice_turn_route(agent, [{"role": "user", "content": "hi"}], "system")
        try:
            assert agent.reasoning_config == {"enabled": False}  # the voice turn itself still runs off
            assert agent.model == ("voice-model" if separate_voice_model else "fake-model")
            info = _session_info(agent, {})
            persisted = _runtime_model_config(agent)
            assert (info["reasoning_effort"], info["model"]) == ("low", "fake-model")
            assert persisted["reasoning_config"] == {"enabled": True, "effort": "low"}
            assert persisted["model"] == "fake-model"
        finally:
            end_voice_turn_route(agent)
        assert _session_info(agent, {})["reasoning_effort"] == "low"
