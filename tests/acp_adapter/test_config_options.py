"""Typed ACP options reject unsupported mutations and retain session reasoning on resume."""
from types import SimpleNamespace

import acp
import pytest
from acp.schema import ModelInfo, SessionModelState

from acp_adapter.config_options import build_config_options, reasoning_efforts
from acp_adapter.server import HermesACPAgent
from acp_adapter.session import SessionManager
from hermes_state import SessionDB


@pytest.fixture
def configured_session(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {
        "model": {"default": "gpt-6.1-sol", "provider": "openai-codex"},
    })
    db = SessionDB(tmp_path / "state.db")
    manager = SessionManager(db=db)

    def construct(**kwargs):
        return SimpleNamespace(
            model=kwargs.get("model") or "gpt-6.1-sol", provider="openai-codex",
            base_url="https://chatgpt.com/backend-api/codex", api_mode="codex_responses",
            reasoning_config=kwargs.get("reasoning_override") or {"enabled": True, "effort": "medium"},
        )

    monkeypatch.setattr(manager, "_make_agent", construct)
    state = manager.create_session(cwd=str(tmp_path))
    state.history = [{"role": "user", "content": "Preserve context 89271"}]
    server = HermesACPAgent(manager)
    monkeypatch.setattr(server, "_build_model_state", lambda state: SessionModelState(
        current_model_id="openai-codex:" + state.model,
        available_models=[ModelInfo(model_id="openai-codex:gpt-6.1-sol", name="Sol")],
    ))
    yield server, manager, state, construct, db
    db.close()


@pytest.mark.parametrize("provider,model,base_url", [
    ("openai", "gpt-4o-mini", "https://api.openai.com/v1"),
    ("custom", "gpt-4o-mini", "https://api.openai.com/v1"),
    ("custom", "unknown-model", "http://localhost:11434/v1"),
    ("custom:private", "gpt-6.1-sol", "https://relay.example/v1"),
    ("unknown-provider", "unknown-model", ""),
])
def test_non_reasoning_and_unknown_models_have_no_effort_picker(provider, model, base_url):
    assert reasoning_efforts(SimpleNamespace(provider=provider, model=model, base_url=base_url)) == ()


@pytest.mark.asyncio
@pytest.mark.parametrize("config_id,value", [("model", "not-advertised"), ("reasoning_effort", "ultra"),
                                           ("unknown", "high")])
async def test_invalid_option_leaves_session_unchanged(configured_session, config_id, value):
    server, _, state, _, _ = configured_session
    original = state.agent
    with pytest.raises(acp.RequestError):
        await server.set_config_option(config_id, state.session_id, value)
    assert state.agent is original
    assert state.model == "gpt-6.1-sol"
    assert state.reasoning_override is None


@pytest.mark.asyncio
async def test_busy_session_cannot_change_reasoning(configured_session):
    server, _, state, _, _ = configured_session
    state.is_running = True
    original = state.agent
    with pytest.raises(acp.RequestError):
        await server.set_config_option("reasoning_effort", state.session_id, "high")
    assert state.agent is original
    assert state.reasoning_override is None
    assert state.is_running is True


@pytest.mark.asyncio
async def test_failed_reasoning_rebuild_preserves_runtime(configured_session, monkeypatch):
    server, manager, state, _, _ = configured_session
    original = state.agent

    def fail(**kwargs):
        raise RuntimeError("Provider unavailable")

    monkeypatch.setattr(manager, "_make_agent", fail)
    with pytest.raises(RuntimeError, match="Provider unavailable"):
        await server.set_config_option("reasoning_effort", state.session_id, "high")
    assert state.agent is original
    assert state.reasoning_override is None
    assert state.command_op is False


@pytest.mark.asyncio
async def test_reasoning_and_context_survive_fresh_manager(configured_session, monkeypatch):
    server, _, state, construct, db = configured_session
    await server.set_config_option("reasoning_effort", state.session_id, "high")
    restored_manager = SessionManager(db=db)
    monkeypatch.setattr(restored_manager, "_make_agent", construct)
    restored = restored_manager.get_session(state.session_id)
    assert restored.session_id == state.session_id
    assert [(message["role"], message["content"]) for message in restored.history] == [
        ("user", "Preserve context 89271"),
    ]
    assert restored.agent.reasoning_config == {"enabled": True, "effort": "high"}
    assert restored.reasoning_override == {"enabled": True, "effort": "high"}


@pytest.mark.parametrize("configured_provider,detected_provider,default_choice", [
    ("ollama", "ollama", "custom:ollama:local-model"),
    ("auto", "openai-codex", "openai-codex:local-model"),
    ("", "openai-codex", "openai-codex:local-model"),
])
def test_configured_default_identity_does_not_follow_session_switch(
    monkeypatch, configured_provider, detected_provider, default_choice,
):
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {
        "model": {"default": "local-model", "provider": configured_provider},
    })
    monkeypatch.setattr("acp_adapter.auth.detect_provider", lambda: detected_provider)
    models = SessionModelState(current_model_id="other:switched-model", available_models=[
        ModelInfo(model_id="other:switched-model", name="Switched"),
        ModelInfo(model_id=default_choice, name="Configured"),
    ])
    state = SimpleNamespace(agent=SimpleNamespace(provider="other", model="switched-model"), mode="default")
    option = build_config_options(state, models, HermesACPAgent._MODES)[0]
    assert option.current_value == "other:switched-model"
    assert option.options[0].value == default_choice
    assert option.options[0].name == "Configured Default — Configured"


def test_fork_reasoning_snapshot_matches_runtime_when_source_changes(configured_session, monkeypatch):
    _, manager, original, construct, db = configured_session
    original.reasoning_override = {"enabled": True, "effort": "low"}

    def construct_while_source_changes(**kwargs):
        original.reasoning_override = {"enabled": True, "effort": "high"}
        original.history.append({"role": "user", "content": "After fork began"})
        return construct(**kwargs)

    monkeypatch.setattr(manager, "_make_agent", construct_while_source_changes)
    fork = manager.fork_session(original.session_id, cwd=original.cwd)
    assert fork.agent.reasoning_config == {"enabled": True, "effort": "low"}
    assert fork.reasoning_override == fork.agent.reasoning_config
    assert [message["content"] for message in fork.history] == ["Preserve context 89271"]
    fresh = SessionManager(db=db)
    monkeypatch.setattr(fresh, "_make_agent", construct)
    restored = fresh.get_session(fork.session_id)
    assert restored.agent.reasoning_config == fork.agent.reasoning_config
