"""``auxiliary.voice_chat``: a voice turn runs on the voice model, the next turn on the main one.

Two real loopback providers; the agent talks HTTP to both, so the assertion is on what each
server actually received rather than on agent attributes.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

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
        # Verdicts the main route earned: the voice route starts without them, main gets them back.
        agent._stream_options_unsupported = True
        agent._codex_reasoning_replay_enabled, agent._codex_reasoning_replay_rejected = False, True
        agent._voice_turn_pending = True
        spoken = agent.run_conversation("spoken question", conversation_history=first["messages"])
        agent.run_conversation("typed again", conversation_history=spoken["messages"])
        assert agent._stream_options_unsupported is True
        assert (agent._codex_reasoning_replay_enabled, agent._codex_reasoning_replay_rejected) == (False, True)
        assert "stream_options" not in main.main_requests()[-1]

        main_models = [r["model"] for r in main.main_requests()]
        voice_models = [r["model"] for r in voice.main_requests()]
        if voice_window_ok:
            assert voice_models == ["voice-model"]
            assert "stream_options" in voice.main_requests()[0]
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


def test_session_runtime_view_preserves_requested_provider_during_voice_turn():
    from agent.voice_turn_route import session_runtime_view

    main = SimpleNamespace(
        model="voice-model", provider="custom", requested_provider="custom:voice",
        base_url="https://voice.example/v1", api_mode="chat_completions",
        reasoning_config={"enabled": False}, _voice_route_state=None,
    )
    main._voice_route_state = {
        "snapshot": {
            "model": "main-model", "provider": "custom", "requested_provider": "custom:main",
            "base_url": "https://main.example/v1", "api_mode": "chat_completions",
        },
        "reasoning_config": {"enabled": True},
    }
    view = session_runtime_view(main)
    assert (view.provider, view.requested_provider) == ("custom", "custom:main")


def test_voice_usage_never_becomes_the_session_route(tmp_path):
    from persistence.conversation.usage import BillingRoute, TokenCounts, UsageDelta
    from persistence_assembly import open_persistence, profile_scope
    from tests.fixtures.storage_read import session_row
    from tests.fixtures.storage_seed import seed_session

    with open_persistence(profile_scope(home=tmp_path)) as db:
        seed_session(db, "s1", "cli", model="main-model")
        db.usage.queue_usage("s1", UsageDelta(tokens=TokenCounts(input_tokens=10, output_tokens=5), api_call_count=1,
                                              route=BillingRoute(model="voice-model", billing_provider="voicep"),
                                              task="voice_chat"))
        assert db.usage.flush_usage()
        row = session_row(db, "s1")
        assert row["model"] == "main-model"
        assert row["input_tokens"] == 10
        assert db.usage.auxiliary_usage_by_task("s1")["voice_chat"].tokens.input_tokens == 10
        assert db.usage.recent_route("s1") is None


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


# --- composed with adaptive reasoning: the voice turn must hand back the adaptive turn's own config ---

DEBUG = "Why does the gateway keep failing after I restart it?"  # lone code signal: high


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


# --- an adaptive turn's adjusted level is the turn's own: session surfaces report the baseline ---


@pytest.mark.parametrize("baseline,message,policy,adjusted", [
    ("medium", DEBUG, {}, "high"),                        # escalation
    ("high", "thanks!", {"min_effort": "low"}, "low"),    # downshift
])
def test_session_view_reports_an_adjusted_turns_baseline(effort_only_voice, baseline, message, policy, adjusted):
    """Alone and under a voice route bound inside it; a level something else installs comes through as is."""
    from agent.adaptive_reasoning import adaptive_reasoning_turn
    from agent.voice_turn_route import begin_voice_turn_route, end_voice_turn_route, session_runtime_view

    agent = _adaptive_agent(baseline, **policy)
    own = {"enabled": True, "effort": baseline}
    with adaptive_reasoning_turn(agent, message):
        assert agent.reasoning_config == {"enabled": True, "effort": adjusted}  # what the request sends
        assert session_runtime_view(agent).reasoning_config == own
        begin_voice_turn_route(agent, [], "system")
        assert session_runtime_view(agent).reasoning_config == own
        end_voice_turn_route(agent)
        assert session_runtime_view(agent).reasoning_config == own
        fallback = agent.reasoning_config = {"enabled": True, "effort": "xhigh"}
        assert session_runtime_view(agent).reasoning_config is fallback
    assert session_runtime_view(agent) is agent


def _session_surfaces(agent):
    """Every session-level reader of the agent's effort: reported, persisted (resume row) or config.get."""
    from gateway.session_config import _reasoning
    from gateway.session_local import _reasoning_fields
    from tui_gateway.server import _runtime_model_config, _session_info

    row = _runtime_model_config(agent)
    return {"session_local": _reasoning_fields(agent, None, agent.model)["reasoning_effort"],
            "session_info": _session_info(agent, {})["reasoning_effort"],
            "row": ((row["reasoning_config"] or {}).get("effort"), row["reasoning_user_override"]),
            "config_get": _reasoning(agent, None, {})["value"]}


@pytest.mark.parametrize("pick", [None, "low"])
def test_adjusted_turn_reports_and_persists_the_session_level(tmp_path, monkeypatch, pick):
    """Desktop adopts session.info's effort into its persisted composer and ships a non-default one on
    session.create, which pins the new chat and turns adaptation off. Read while the escalated request
    is in flight, every session surface reports the baseline; the request carries the adjusted level.
    A session-scoped ``/reasoning`` pick made mid-turn (real ``config.set``) is reported as is."""
    from unittest.mock import patch

    from tui_gateway import server

    seen: dict = {}
    emitted: list = []

    def responder(record):
        if pick:
            with patch.dict(server._sessions, {"s1": {"session_key": "k1", "agent": agent}}), \
                    patch.object(server, "_write_config_key") as write_key, \
                    patch.object(server, "_persist_live_session_runtime"), \
                    patch.object(server, "_emit", lambda *a: emitted.append(a)):
                server._methods["config.set"]("rid", {"key": "reasoning", "session_id": "s1", "value": pick})
            write_key.assert_not_called()
        seen.update(request=record["body"].get("reasoning_effort"), **_session_surfaces(agent))
        return Text("ok")

    with FakeLLMServer(responder) as main:
        home = write_hermes_home(tmp_path / ".hermes", main.base_url)
        monkeypatch.setenv("HERMES_HOME", str(home))
        agent = AIAgent(model="fake-model", provider="custom", base_url=main.base_url, api_key="sk-fake-e2e",
                        quiet_mode=True, skip_context_files=True, skip_memory=True, enabled_toolsets=["memory"],
                        reasoning_config={"enabled": True, "effort": "medium"},
                        adaptive_reasoning={"enabled": True, "max_effort": "high"})
        agent.run_conversation(DEBUG)
    level = pick or "medium"
    assert seen == {"request": "high", "session_local": level, "session_info": level,
                    "row": (level, pick is not None), "config_get": level}
    assert [(e[0], e[2]["reasoning_effort"]) for e in emitted] == ([("session.info", pick)] if pick else [])
    assert agent.reasoning_config == {"enabled": True, "effort": level}


def _routed_voice_home(tmp_path, main, voice, context="128000"):
    return write_hermes_home(tmp_path / ".hermes", main.base_url, extra_config=(
        "auxiliary:\n  voice_chat:\n    provider: custom\n"
        f"    base_url: {voice.base_url}\n    model: voice-model\n    api_key: sk-fake-e2e\n"
        "custom_providers:\n  - name: voice\n"
        f"    base_url: {voice.base_url}\n    models:\n      voice-model:\n        context_length: {context}\n"))


def test_routed_voice_turn_inside_an_adjusted_turn_reports_the_sessions_route(tmp_path, monkeypatch):
    """Both at once: the voice model runs the request and the adaptive level is installed. The session
    view still names the chat's own model (the voice snapshot) and the adaptive baseline."""
    from agent.adaptive_reasoning import adaptive_reasoning_turn
    from agent.voice_turn_route import begin_voice_turn_route, end_voice_turn_route, session_runtime_view

    with FakeLLMServer([]) as main, FakeLLMServer([]) as voice:
        monkeypatch.setenv("HERMES_HOME", str(_routed_voice_home(tmp_path, main, voice)))
        agent = AIAgent(model="fake-model", provider="custom", base_url=main.base_url, api_key="sk-fake-e2e",
                        quiet_mode=True, skip_context_files=True, skip_memory=True, enabled_toolsets=["memory"],
                        reasoning_config={"enabled": True, "effort": "medium"},
                        adaptive_reasoning={"enabled": True, "max_effort": "high"})
        with adaptive_reasoning_turn(agent, DEBUG):
            agent._voice_turn_pending = True
            begin_voice_turn_route(agent, [{"role": "user", "content": "hi"}], "system")
            try:
                assert agent.model == "voice-model"
                view = session_runtime_view(agent)
                assert (view.model, view.reasoning_config) == ("fake-model", {"enabled": True, "effort": "medium"})
            finally:
                end_voice_turn_route(agent)
        assert (agent.model, agent.reasoning_config) == ("fake-model", {"enabled": True, "effort": "medium"})


def test_refused_voice_route_keeps_what_the_main_route_learns_that_turn(tmp_path, monkeypatch):
    """A voice route refused at bind (too large for its window) reinstalls the main runtime at once and
    keeps only the turn's reasoning to hand back: a verdict the main route earns that turn survives its end."""
    from agent.voice_turn_route import begin_voice_turn_route, end_voice_turn_route

    with FakeLLMServer([]) as main, FakeLLMServer([]) as voice:
        monkeypatch.setenv("HERMES_HOME", str(_routed_voice_home(tmp_path, main, voice, context="1000")))
        agent = _agent(tmp_path, main.base_url)
        agent.reasoning_config = {"enabled": True, "effort": "low"}
        assert agent._codex_reasoning_replay_rejected is False
        agent._voice_turn_pending = True
        begin_voice_turn_route(agent, [{"role": "user", "content": "word " * 4000}], "system")
        assert agent.model == "fake-model"  # refused: the main model answers
        agent._codex_reasoning_replay_rejected = True  # learned on the main route during the turn
        end_voice_turn_route(agent)
        assert agent._codex_reasoning_replay_rejected is True
        assert (agent.model, agent.reasoning_config) == ("fake-model", {"enabled": True, "effort": "low"})
