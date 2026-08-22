"""Durable, async-safe persistence for session-scoped /reasoning overrides."""
from unittest.mock import AsyncMock
import pytest
from gateway.config import GatewayConfig, Platform
from gateway.session import SessionSource, SessionStore, sanitize_reasoning_override

def _source():
    return SessionSource(platform=Platform.TELEGRAM, user_id="u1", chat_id="c1", chat_type="dm")

@pytest.fixture
def store_factory(tmp_path, monkeypatch):
    import hermes_state
    def _disabled(): raise RuntimeError("SQLite disabled in test")
    monkeypatch.setattr(hermes_state, "SessionDB", _disabled)
    return lambda: SessionStore(sessions_dir=tmp_path, config=GatewayConfig())

def test_round_trips_full_effort_ladder(store_factory):
    from hermes_constants import VALID_REASONING_EFFORTS
    store = store_factory(); entry = store.get_or_create_session(_source())
    for effort in VALID_REASONING_EFFORTS:
        store.set_reasoning_override(entry.session_key, {"enabled": True, "effort": effort, "unexpected": "discard"})
        assert store_factory().get_reasoning_override(entry.session_key) == {"enabled": True, "effort": effort}

def test_clear_and_expiry_survive_restart(store_factory):
    store = store_factory(); entry = store.get_or_create_session(_source())
    store.set_reasoning_override(entry.session_key, {"enabled": True, "effort": "ultra"})
    store.set_reasoning_override(entry.session_key, None)
    assert store_factory().get_reasoning_override(entry.session_key) is None
    store.set_reasoning_override(entry.session_key, {"enabled": True, "effort": "max"})
    store.set_expiry_finalized(entry)
    assert store_factory().get_reasoning_override(entry.session_key) is None

def test_state_db_round_trip_without_json_mirror(tmp_path, monkeypatch):
    import hermes_state

    real_session_db = hermes_state.SessionDB
    db_path = tmp_path / "state.db"
    monkeypatch.setattr(
        hermes_state,
        "SessionDB",
        lambda: real_session_db(db_path=db_path),
    )
    config = GatewayConfig()
    config.write_sessions_json = False
    sessions_dir = tmp_path / "sessions"

    store = SessionStore(sessions_dir=sessions_dir, config=config)
    entry = store.get_or_create_session(_source())
    store.set_reasoning_override(
        entry.session_key, {"enabled": True, "effort": "ultra"}
    )
    store._db.close()

    restored = SessionStore(sessions_dir=sessions_dir, config=config)
    try:
        assert restored.get_reasoning_override(entry.session_key) == {
            "enabled": True,
            "effort": "ultra",
        }
    finally:
        restored._db.close()


def test_sanitizer_rejects_malformed_shape():
    assert sanitize_reasoning_override(None) is None
    assert sanitize_reasoning_override({"enabled": "false", "effort": "max"}) is None
    assert sanitize_reasoning_override({"enabled": True, "effort": "unknown"}) is None
    assert sanitize_reasoning_override({"enabled": False, "effort": "ultra"}) == {"enabled": False}

@pytest.mark.asyncio
async def test_command_persists_through_async_store():
    import gateway.run as gateway_run
    runner = object.__new__(gateway_run.GatewayRunner)
    runner._session_reasoning_overrides = {}; runner._reasoning_config = None
    runner._show_reasoning = False; runner._running_agents = {}
    runner.session_store = object()
    runner._async_session_store = AsyncMock()
    runner._async_session_store._store = runner.session_store
    runner._evict_cached_agent = lambda _key: None
    runner._save_gateway_config_key = lambda *_args: True
    assert await runner._apply_reasoning_selection("agent:main:telegram:dm:u1", "telegram", "ultra")
    runner._async_session_store.set_reasoning_override.assert_awaited_once_with("agent:main:telegram:dm:u1", {"enabled": True, "effort": "ultra"})

def test_rehydrate_copies_durable_override():
    import gateway.run as gateway_run
    runner = object.__new__(gateway_run.GatewayRunner); runner._session_reasoning_overrides = {}
    entry = type("Entry", (), {"session_key": "agent:main:telegram:dm:u1", "reasoning_override": {"enabled": True, "effort": "max"}})()
    runner._rehydrate_session_reasoning_override(entry)
    assert runner._resolve_session_reasoning_config(session_key=entry.session_key) == {"enabled": True, "effort": "max"}
