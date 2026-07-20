"""Per-session /model overrides must survive gateway restarts (#3659 salvage).

``GatewayRunner._session_model_overrides`` is in-memory, so before persistence
a gateway restart silently reverted every session to the global default model.
The non-secret parts (model/provider/base_url/responses_transport) are written through to the
session store (``SessionEntry.model_override`` in sessions.json) and lazily
rehydrated on first use after a restart, with credentials re-resolved through
the normal runtime provider resolution.

Covers:
  - the override survives a simulated restart (a second SessionStore instance
    reading the same sessions dir, and a fresh runner rehydrating from it)
  - /new (SessionStore.reset_session) clears the persisted override so a
    restart cannot resurrect it
  - api_key is NEVER serialized to sessions.json
"""
import json
from unittest.mock import patch

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.session import (
    SessionEntry,
    SessionSource,
    SessionStore,
    sanitize_model_override,
)

OVERRIDE = {
    "model": "gpt-5o",
    "provider": "openai",
    "api_key": "sk-SUPER-SECRET-do-not-persist",
    "base_url": "https://api.openai.example/v1",
    "api_mode": "responses",
    "responses_transport": "websocket-cached",
}


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


@pytest.fixture
def store_factory(tmp_path, monkeypatch):
    """Build SessionStores over a shared sessions dir, without SQLite."""

    def _raise():
        raise RuntimeError("SQLite disabled in test")

    import hermes_state

    monkeypatch.setattr(hermes_state, "SessionDB", _raise)

    def _make() -> SessionStore:
        store = SessionStore(sessions_dir=tmp_path, config=GatewayConfig())
        assert store._db is None
        return store

    return _make


def _sessions_json(tmp_path) -> str:
    return (tmp_path / "sessions.json").read_text(encoding="utf-8")


def test_override_persists_and_survives_restart(store_factory, tmp_path):
    store = store_factory()
    entry = store.get_or_create_session(_make_source())
    session_key = entry.session_key

    store.set_model_override(session_key, OVERRIDE)

    # Simulated restart: a brand-new store instance reads the same dir.
    store2 = store_factory()
    persisted = store2.get_model_override(session_key)
    assert persisted == {
        "model": "gpt-5o",
        "provider": "openai",
        "base_url": "https://api.openai.example/v1",
        "responses_transport": "websocket-cached",
    }


def test_api_key_never_serialized(store_factory, tmp_path):
    store = store_factory()
    entry = store.get_or_create_session(_make_source())

    store.set_model_override(entry.session_key, OVERRIDE)

    raw = _sessions_json(tmp_path)
    assert "sk-SUPER-SECRET-do-not-persist" not in raw
    assert "api_key" not in raw
    # api_mode is re-derived from provider resolution; not persisted either.
    data = json.loads(raw)
    stored = data[entry.session_key]["model_override"]
    assert set(stored) == {
        "model",
        "provider",
        "base_url",
        "responses_transport",
    }


def test_from_dict_strips_api_key_from_tampered_json():
    """Even a hand-edited sessions.json with an api_key must not load one."""
    store_entry = SessionEntry.from_dict(
        {
            "session_key": "k1",
            "session_id": "s1",
            "created_at": "2026-01-01T00:00:00",
            "updated_at": "2026-01-01T00:00:00",
            "model_override": {
                "model": "m1",
                "provider": "p1",
                "api_key": "sk-injected",
                "api_mode": "chat_completions",
            },
        }
    )
    assert store_entry.model_override == {"model": "m1", "provider": "p1"}


def test_new_clears_persisted_override(store_factory, tmp_path):
    """/new resets the session; the persisted override must not survive it."""
    store = store_factory()
    entry = store.get_or_create_session(_make_source())
    session_key = entry.session_key

    store.set_model_override(session_key, OVERRIDE)
    assert store.get_model_override(session_key) is not None

    # /new path -> SessionStore.reset_session creates a fresh entry.
    new_entry = store.reset_session(session_key)
    assert new_entry is not None
    assert store.get_model_override(session_key) is None

    # Restart after /new must NOT resurrect the override.
    store2 = store_factory()
    assert store2.get_model_override(session_key) is None
    assert "gpt-5o" not in _sessions_json(tmp_path)


def _make_runner(store):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._session_model_overrides = {}
    runner.session_store = store
    return runner


def test_runner_rehydrates_override_after_restart(store_factory):
    store = store_factory()
    entry = store.get_or_create_session(_make_source())
    session_key = entry.session_key
    store.set_model_override(session_key, OVERRIDE)

    # Simulated restart: fresh store + fresh runner with an empty in-memory
    # override map, credentials re-resolved via runtime provider resolution.
    runner = _make_runner(store_factory())
    with patch(
        "gateway.run._resolve_runtime_agent_kwargs_for_provider",
        return_value={
            "api_key": "sk-fresh-from-keychain",
            "api_mode": "responses",
            "base_url": "https://api.openai.example/v1",
            "provider": "openai",
            "responses_transport": "sse",
        },
    ):
        runner._rehydrate_session_model_override(session_key)

    override = runner._session_model_overrides[session_key]
    assert override["model"] == "gpt-5o"
    assert override["provider"] == "openai"
    assert override["base_url"] == "https://api.openai.example/v1"
    # Credentials come from live resolution, never from disk.
    assert override["api_key"] == "sk-fresh-from-keychain"
    assert override["api_mode"] == "responses"
    assert override["responses_transport"] == "websocket-cached"


def test_runner_rehydrate_keeps_live_override(store_factory):
    """An in-memory override (live gateway state) always wins over disk."""
    store = store_factory()
    entry = store.get_or_create_session(_make_source())
    session_key = entry.session_key
    store.set_model_override(session_key, OVERRIDE)

    runner = _make_runner(store)
    live = {"model": "live-model", "provider": "anthropic"}
    runner._session_model_overrides[session_key] = live

    runner._rehydrate_session_model_override(session_key)

    assert runner._session_model_overrides[session_key] is live


def test_runner_rehydrate_noop_without_persisted_override(store_factory):
    store = store_factory()
    entry = store.get_or_create_session(_make_source())

    runner = _make_runner(store)
    runner._rehydrate_session_model_override(entry.session_key)

    assert runner._session_model_overrides == {}


def test_runner_rehydrate_survives_credential_resolution_failure(store_factory):
    """Missing credentials degrade to a credential-less override, not a crash."""
    store = store_factory()
    entry = store.get_or_create_session(_make_source())
    session_key = entry.session_key
    store.set_model_override(session_key, OVERRIDE)

    runner = _make_runner(store)
    with patch(
        "gateway.run._resolve_runtime_agent_kwargs_for_provider",
        side_effect=RuntimeError("no credentials"),
    ):
        runner._rehydrate_session_model_override(session_key)

    override = runner._session_model_overrides[session_key]
    assert override["model"] == "gpt-5o"
    assert override.get("api_key") is None


def test_sanitize_model_override():
    assert sanitize_model_override(None) is None
    assert sanitize_model_override({}) is None
    assert sanitize_model_override({"api_key": "sk-x", "api_mode": "chat"}) is None
    assert sanitize_model_override(OVERRIDE) == {
        "model": "gpt-5o",
        "provider": "openai",
        "base_url": "https://api.openai.example/v1",
        "responses_transport": "websocket-cached",
    }
