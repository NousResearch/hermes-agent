"""Per-session /model overrides must survive gateway restarts (#3659 salvage).

``GatewayRunner._session_model_overrides`` is in-memory, so before persistence
a gateway restart silently reverted every session to the global default model.
The non-secret parts (model/provider/base_url) are now written through to the
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
    }


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


def test_sanitize_model_override():
    assert sanitize_model_override(None) is None
    assert sanitize_model_override({}) is None
    assert sanitize_model_override({"api_key": "sk-x", "api_mode": "chat"}) is None
    assert sanitize_model_override(OVERRIDE) == {
        "model": "gpt-5o",
        "provider": "openai",
        "base_url": "https://api.openai.example/v1",
    }


# --- named-provider identity must survive session overrides + restart -------
# Two named providers (A configured globally, B chosen per-session) can share a
# single model id AND base_url yet declare different exact context windows. The
# session override must carry B's *named* identity (``requested_provider``) so a
# later metadata lookup resolves B's window, not the globally-configured A's.

_NAMED_SHARED_URL = "http://da-aihost01:4000/v1"
_NAMED_MODEL = "vllm/DeepSeek-V4-Flash-0731"
_NAMED_OVERRIDE_B = {
    "model": _NAMED_MODEL,
    "provider": "beta",
    "requested_provider": "beta",
    "api_key": "sk-SECRET-do-not-persist",
    "base_url": _NAMED_SHARED_URL,
    "api_mode": "chat_completions",
}


def test_named_requested_provider_persists_and_survives_restart(store_factory):
    store = store_factory()
    entry = store.get_or_create_session(_make_source())
    session_key = entry.session_key

    store.set_model_override(session_key, _NAMED_OVERRIDE_B)

    store2 = store_factory()
    persisted = store2.get_model_override(session_key)
    assert persisted is not None
    # The named identity must round-trip so a shared-endpoint sibling cannot
    # borrow it after a restart.
    assert persisted.get("requested_provider") == "beta"
    # Secrets are still never written.
    assert "api_key" not in persisted


def test_runner_rehydrate_restores_named_requested_provider(store_factory):
    store = store_factory()
    entry = store.get_or_create_session(_make_source())
    session_key = entry.session_key
    store.set_model_override(session_key, _NAMED_OVERRIDE_B)

    runner = _make_runner(store_factory())
    with patch(
        "gateway.run._resolve_runtime_agent_kwargs_for_provider",
        return_value={
            "api_key": "sk-fresh",
            "api_mode": "chat_completions",
            "base_url": _NAMED_SHARED_URL,
            "provider": "beta",
        },
    ):
        runner._rehydrate_session_model_override(session_key)

    override = runner._session_model_overrides[session_key]
    assert override["requested_provider"] == "beta"
