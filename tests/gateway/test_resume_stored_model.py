"""Messaging resume restores the target route through the durable store and next turn."""
from unittest.mock import AsyncMock, patch

import pytest

from gateway.config import GatewayConfig
from gateway.session import SessionStore
from tests.gateway.test_resume_command import _make_event, _make_runner


@pytest.fixture
def resume_route(tmp_path, monkeypatch):
    import hermes_state

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", tmp_path / "state.db")
    stores = []

    def new_store():
        store = SessionStore(sessions_dir=tmp_path / "sessions", config=GatewayConfig())
        stores.append(store)
        assert store._db is not None
        return store

    store = new_store()
    event = _make_event(text="/resume target")
    entry = store.get_or_create_session(event.source)
    key = entry.session_key
    store.set_model_override(key, {"model": "departing", "provider": "openrouter"})
    runner = _make_runner(session_db=store._db, event=event)
    runner.config = GatewayConfig()
    runner.session_store = store
    runner._session_model_overrides = {key: {"model": "departing"}}
    yield runner, store, event, key, new_store, tmp_path
    for opened in stores:
        opened._db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", ["nested", "legacy", "model-only", "named-custom"])
async def test_resume_route_survives_db_only_restart_and_next_turn(resume_route, shape):
    runner, store, event, key, new_store, tmp_path = resume_route
    route = {"provider": "openai", "base_url": "https://api.openai.com/v1"}
    if shape == "named-custom":
        route = {"provider": "custom:chosen", "base_url": "https://chosen.example/v1"}
    config = {"gateway_runtime": {**route, "api_key": "not-to-be-restored", "api_mode": "old-wire"}}
    if shape == "legacy":
        config = route
    elif shape == "model-only":
        config = {}
    store._db.create_session("target", "telegram", user_id="12345", chat_id="67890",
                             model="target-model", model_config=config)
    other = store.get_or_create_session(_make_event(chat_id="other").source)
    store.set_model_override(other.session_key, {"model": "unrelated"})
    expected = {"model": "target-model", **(route if shape != "model-only" else {})}

    await runner._handle_resume_command(event)

    assert store.get_or_create_session(event.source).session_id == "target"
    assert store.get_model_override(key) == expected
    assert key not in runner._session_model_overrides
    (tmp_path / "sessions" / "sessions.json").unlink()
    restarted = new_store()
    assert restarted.get_or_create_session(event.source).session_id == "target"
    assert restarted.get_model_override(key) == expected
    assert restarted.get_model_override(other.session_key) == {"model": "unrelated"}
    fresh = _make_runner(session_db=restarted._db, event=event)
    fresh.session_store = restarted
    fresh.config = GatewayConfig()
    runtime = {"provider": route["provider"], "base_url": route["base_url"],
               "api_key": "fresh-credential", "api_mode": "chat_completions"}
    with patch("gateway.run._resolve_runtime_agent_kwargs_for_provider", return_value=runtime), \
         patch("gateway.run._resolve_runtime_agent_kwargs", return_value=runtime):
        model, resolved = fresh._resolve_session_agent_runtime(
            session_key=key, user_config={"model": {"default": "ambient", "provider": "openai"}})
    assert model == "target-model"
    assert resolved["api_key"] == "fresh-credential"
    assert resolved["base_url"] == route["base_url"]
    assert resolved["api_mode"] == "chat_completions"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["absent-model", "blank-model", "missing-row", "read-error"])
async def test_resume_without_readable_model_durably_clears_old_pin(resume_route, mode):
    runner, store, event, key, new_store, tmp_path = resume_route
    store._db.create_session("target", "telegram", user_id="12345", chat_id="67890",
                             model="   " if mode == "blank-model" else None)
    original = runner._session_db.get_session

    async def read_target(sid):
        # Fail only the metadata restore read AFTER authorization and successful switch.
        if store.get_or_create_session(event.source).session_id == "target":
            if mode == "read-error":
                raise OSError("metadata unavailable")
            if mode == "missing-row":
                return None
        return await original(sid)

    with patch.object(runner._session_db, "get_session", side_effect=read_target):
        await runner._handle_resume_command(event)
    assert store.get_model_override(key) is None
    assert key not in runner._session_model_overrides
    (tmp_path / "sessions" / "sessions.json").unlink()
    restarted = new_store()
    assert restarted.get_model_override(key) is None
    fresh = _make_runner(session_db=restarted._db, event=event)
    fresh.session_store = restarted
    with patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"provider": "openai"}):
        model, _ = fresh._resolve_session_agent_runtime(
            session_key=key, user_config={"model": {"default": "ambient"}})
    assert model == "ambient"


@pytest.mark.asyncio
async def test_failed_switch_keeps_departing_pin(resume_route):
    runner, store, event, key, _, _ = resume_route
    store._db.create_session("target", "telegram", user_id="12345", chat_id="67890", model="target-model")
    with patch.object(runner.async_session_store, "switch_session", new=AsyncMock(return_value=None)):
        await runner._handle_resume_command(event)
    assert store.get_model_override(key)["model"] == "departing"


@pytest.mark.asyncio
async def test_resume_uses_compression_tip_model(resume_route):
    runner, store, event, key, _, _ = resume_route
    store._db.create_session("target", "telegram", user_id="12345", chat_id="67890", model="root-model")
    store._db.end_session("target", "compression")
    store._db.create_session("tip", "telegram", user_id="12345", chat_id="67890",
                             parent_session_id="target", model="tip-model")
    await runner._handle_resume_command(event)
    assert store.get_or_create_session(event.source).session_id == "tip"
    assert store.get_model_override(key) == {"model": "tip-model"}


@pytest.mark.asyncio
async def test_resume_unavailable_provider_keeps_coherent_fallback(resume_route):
    runner, store, event, key, _, _ = resume_route
    route = {"provider": "openai-codex", "base_url": "https://chatgpt.com/backend-api/codex"}
    store._db.create_session("target", "telegram", user_id="12345", chat_id="67890",
                             model="target-model", model_config={"gateway_runtime": route})
    await runner._handle_resume_command(event)
    fallback = {"provider": "openai", "base_url": "https://api.openai.com/v1", "api_key": "ambient-key"}
    with patch("gateway.run._resolve_runtime_agent_kwargs_for_provider", side_effect=RuntimeError("gone")), \
         patch("gateway.run._resolve_runtime_agent_kwargs", return_value=dict(fallback)):
        model, runtime = runner._resolve_session_agent_runtime(
            session_key=key, user_config={"model": {"default": "ambient"}})
    assert model == "ambient"
    assert {k: runtime[k] for k in fallback} == fallback
    assert runner._pre_agent_fallback_notice
    assert store.get_model_override(key) == {"model": "target-model", **route}
