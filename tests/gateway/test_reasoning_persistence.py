"""Durable, async-safe persistence for session-scoped /reasoning overrides."""
from unittest.mock import AsyncMock
import pytest
from gateway.config import GatewayConfig, Platform
from gateway.session import SessionSource, SessionStore, sanitize_reasoning_override

def _source():
    return SessionSource(platform=Platform.TELEGRAM, user_id="u1", chat_id="c1", chat_type="dm")

# The reasoning setter is merged into the shared runtime-options write.
def _set_reasoning(store, session_key, override):
    return store.set_runtime_options(session_key, reasoning_override=override)

def _get_reasoning(store, session_key):
    return (store.get_runtime_options(session_key) or {}).get("reasoning_override")

@pytest.fixture
def store_factory(tmp_path, monkeypatch):
    import hermes_state
    def _disabled(**_kw): raise RuntimeError("SQLite disabled in test")
    monkeypatch.setattr(hermes_state, "SessionDB", _disabled)
    return lambda: SessionStore(sessions_dir=tmp_path, config=GatewayConfig())

def test_round_trips_full_effort_ladder(store_factory):
    from hermes_constants import VALID_REASONING_EFFORTS
    store = store_factory(); entry = store.get_or_create_session(_source())
    for effort in VALID_REASONING_EFFORTS:
        _set_reasoning(store, entry.session_key, {"enabled": True, "effort": effort, "unexpected": "discard"})
        assert _get_reasoning(store_factory(), entry.session_key) == {"enabled": True, "effort": effort}

def test_clear_and_reset_survive_restart(store_factory):
    store = store_factory(); entry = store.get_or_create_session(_source())
    _set_reasoning(store, entry.session_key, {"enabled": True, "effort": "ultra"})
    _set_reasoning(store, entry.session_key, None)
    assert _get_reasoning(store_factory(), entry.session_key) is None
    _set_reasoning(store, entry.session_key, {"enabled": True, "effort": "max"})
    store.reset_session(entry.session_key)
    assert _get_reasoning(store_factory(), entry.session_key) is None

def test_state_db_round_trip_without_json_mirror(tmp_path, monkeypatch):
    import hermes_state

    real_session_db = hermes_state.SessionDB
    db_path = tmp_path / "state.db"
    monkeypatch.setattr(
        hermes_state,
        "SessionDB",
        lambda **_kw: real_session_db(db_path=db_path),
    )
    config = GatewayConfig()
    config.write_sessions_json = False
    sessions_dir = tmp_path / "sessions"

    store = SessionStore(sessions_dir=sessions_dir, config=config)
    entry = store.get_or_create_session(_source())
    _set_reasoning(
        store, entry.session_key, {"enabled": True, "effort": "ultra"}
    )
    store._db.close()

    restored = SessionStore(sessions_dir=sessions_dir, config=config)
    try:
        assert _get_reasoning(restored, entry.session_key) == {
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
    from unittest.mock import MagicMock
    runner = object.__new__(gateway_run.GatewayRunner)
    runner._session_reasoning_overrides = {}; runner._reasoning_config = None
    runner._show_reasoning = False; runner._running_agents = {}
    # The durable write runs on the gateway executor against the store itself.
    runner.session_store = MagicMock()
    runner.session_store.lookup_by_session_key.return_value = type("Entry", (), {"session_id": "s1"})()
    runner.session_store.get_runtime_options.return_value = None
    runner.session_store.get_model_override.return_value = None
    runner.session_store.set_runtime_options.return_value = True
    runner._evict_cached_agent = lambda _key: None
    runner._save_gateway_config_key = lambda *_args: True
    assert await runner._apply_reasoning_selection("agent:main:telegram:dm:u1", "telegram", "ultra")
    runner.session_store.set_runtime_options.assert_called_once_with("agent:main:telegram:dm:u1", expected_session_id="s1", reasoning_override={"enabled": True, "effort": "ultra"})
    assert runner._session_reasoning_overrides["agent:main:telegram:dm:u1"] == {"enabled": True, "effort": "ultra"}

def test_rehydrate_copies_durable_override():
    import gateway.run as gateway_run
    runner = object.__new__(gateway_run.GatewayRunner); runner._session_reasoning_overrides = {}
    session_key = "agent:main:telegram:dm:u1"
    runner.session_store = type("Store", (), {
        "get_model_override": lambda self, _key: None,
        "get_runtime_options": lambda self, _key: {"reasoning_override": {"enabled": True, "effort": "max"}},
    })()
    runner._rehydrate_session_runtime_options(session_key)
    assert runner._resolve_session_reasoning_config(session_key=session_key) == {"enabled": True, "effort": "max"}
