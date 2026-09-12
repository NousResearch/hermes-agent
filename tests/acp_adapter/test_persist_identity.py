"""ACP persist identity: named custom providers and turn-scoped fallback."""

from types import SimpleNamespace

from acp_adapter.session import persist_identity, SessionManager, SessionState
from hermes_state import SessionDB


def _agent(**kwargs):
    defaults = {
        "model": "claude-opus-5",
        "provider": "custom",
        "requested_provider": "custom:aiberm",
        "base_url": "https://aiberm.com/v1",
        "api_mode": "chat_completions",
        "_fallback_activated": False,
        "_primary_runtime": {
            "model": "claude-opus-5",
            "provider": "custom",
            "requested_provider": "custom:aiberm",
            "base_url": "https://aiberm.com/v1",
            "api_mode": "chat_completions",
        },
        "_session_db": None,
        "_session_db_created": False,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _state(agent, model="claude-opus-5"):
    return SessionState(
        session_id="s-test",
        agent=agent,
        cwd="/tmp",
        model=model,
        history=[],
    )


def test_persist_auto_keeps_resolved_provider():
    ident = persist_identity(_state(_agent(
        model="anthropic/claude-sonnet-4.5",
        provider="openrouter",
        requested_provider="auto",
        base_url=None,
        _primary_runtime={
            "model": "anthropic/claude-sonnet-4.5",
            "provider": "openrouter",
            "requested_provider": "auto",
            "base_url": None,
            "api_mode": "chat_completions",
        },
    ), model="anthropic/claude-sonnet-4.5"))
    assert ident["model"] == "anthropic/claude-sonnet-4.5"
    assert ident["provider"] == "openrouter"
    assert ident["requested_provider"] == "auto"


def test_persist_keeps_named_custom_provider():
    ident = persist_identity(_state(_agent()))
    assert ident["model"] == "claude-opus-5"
    assert ident["provider"] == "custom:aiberm"
    assert ident["requested_provider"] == "custom:aiberm"
    assert ident["base_url"] == "https://aiberm.com/v1"


def test_persist_does_not_write_fallback_identity():
    agent = _agent(
        model="glm-5.3",
        provider="custom",
        requested_provider="aihubmix",
        base_url="https://api.inferera.com/v1",
        _fallback_activated=True,
    )
    ident = persist_identity(_state(agent, model="glm-5.3"))
    assert ident["model"] == "claude-opus-5"
    assert ident["provider"] == "custom:aiberm"
    assert ident["requested_provider"] == "custom:aiberm"
    assert ident["base_url"] == "https://aiberm.com/v1"


def test_persist_roundtrip_named_provider_survives_restore(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    captured = {}

    def factory(**kwargs):
        captured.update(kwargs)
        return _agent(
            model=kwargs.get("model") or "claude-opus-5",
            requested_provider=kwargs.get("requested_provider") or "custom:aiberm",
            provider=kwargs.get("provider") or "custom",
            base_url=kwargs.get("base_url") or "https://aiberm.com/v1",
        )

    manager = SessionManager(agent_factory=factory, db=db)
    state = manager.create_session(cwd="/tmp")
    state.model = "claude-opus-5"
    state.agent = factory(
        model="claude-opus-5",
        requested_provider="custom:aiberm",
        provider="custom",
        base_url="https://aiberm.com/v1",
    )
    state.history.append({"role": "user", "content": "kept content"})
    manager.save_session(state.session_id)

    row = db.get_session(state.session_id)
    assert row is not None
    assert row["model"] == "claude-opus-5"
    import json
    meta = json.loads(row["model_config"])
    assert meta["provider"] == "custom:aiberm"
    assert meta["requested_provider"] == "custom:aiberm"
    assert meta["base_url"] == "https://aiberm.com/v1"

    manager._sessions.clear()
    restored = manager.get_session(state.session_id)
    assert restored is not None
    assert captured["model"] == "claude-opus-5"
    assert captured["requested_provider"] == "custom:aiberm"
    assert captured["base_url"] == "https://aiberm.com/v1"


def test_fallback_save_does_not_replace_primary_in_db(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")

    def factory(**kwargs):
        return _agent()

    manager = SessionManager(agent_factory=factory, db=db)
    state = manager.create_session(cwd="/tmp")
    state.model = "claude-opus-5"
    state.agent = _agent()
    state.history.append({"role": "user", "content": "kept content"})
    manager.save_session(state.session_id)

    state.agent = _agent(
        model="qwen3.8-flash",
        provider="custom",
        requested_provider="aihubmix",
        base_url="https://api.inferera.com/v1",
        _fallback_activated=True,
    )
    state.model = "qwen3.8-flash"
    manager.save_session(state.session_id)

    row = db.get_session(state.session_id)
    assert row is not None
    assert row["model"] == "claude-opus-5"
    import json
    meta = json.loads(row["model_config"])
    assert meta["provider"] == "custom:aiberm"
    assert meta["base_url"] == "https://aiberm.com/v1"
    assert state.model == "claude-opus-5"


def test_persist_splits_unsplit_named_custom_model(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {
            "providers": {
                "aihubmix": {
                    "name": "AIHubMix",
                    "base_url": "https://api.inferera.com/v1",
                }
            }
        },
    )
    agent = _agent(
        model="aihubmix:qwen3.8-flash",
        provider="custom",
        requested_provider="custom",
        base_url="https://api.inferera.com/v1",
        _primary_runtime={
            "model": "aihubmix:qwen3.8-flash",
            "provider": "custom",
            "requested_provider": "custom",
            "base_url": "https://api.inferera.com/v1",
            "api_mode": "chat_completions",
        },
    )
    ident = persist_identity(_state(agent, model="aihubmix:qwen3.8-flash"))
    assert ident["model"] == "qwen3.8-flash"
    assert ident["provider"] == "custom:aihubmix"
    assert ident["requested_provider"] == "custom:aihubmix"
