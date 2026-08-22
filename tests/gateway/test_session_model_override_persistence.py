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
from types import SimpleNamespace
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


def test_structured_runtime_options_persist_atomically_and_reset(store_factory, tmp_path):
    store = store_factory()
    entry = store.get_or_create_session(_make_source())
    session_key = entry.session_key

    assert store.set_runtime_options(
        session_key,
        model_override=OVERRIDE,
        reasoning_override={"enabled": True, "effort": "high"},
        service_tier_override="normal",
    )

    restarted = store_factory()
    assert restarted.get_runtime_options(session_key) == {
        "model_override": {
            "model": "gpt-5o",
            "provider": "openai",
            "base_url": "https://api.openai.example/v1",
        },
        "reasoning_override": {"enabled": True, "effort": "high"},
        "service_tier_override": "normal",
    }
    assert "sk-SUPER-SECRET" not in _sessions_json(tmp_path)

    restarted.reset_session(session_key)
    assert restarted.get_runtime_options(session_key) == {
        "model_override": None,
        "reasoning_override": None,
        "service_tier_override": None,
    }


def test_runner_rehydrates_all_structured_runtime_options(store_factory):
    store = store_factory()
    entry = store.get_or_create_session(_make_source())
    session_key = entry.session_key
    store.set_runtime_options(
        session_key,
        model_override=OVERRIDE,
        reasoning_override={"enabled": False},
        service_tier_override="normal",
    )

    runner = _make_runner(store_factory())
    with patch(
        "gateway.run._resolve_runtime_agent_kwargs_for_provider",
        return_value={"api_key": "sk-live", "provider": "openai"},
    ):
        runner._rehydrate_session_runtime_options(session_key)

    state = runner._session_state(session_key)
    assert state.conversation.model_override["api_key"] == "sk-live"
    assert state.conversation.reasoning_override == {"enabled": False}
    assert state.conversation.service_tier_override is None
    assert state.persistent.runtime_options_rehydrated is True


@pytest.mark.asyncio
async def test_public_session_options_api_applies_without_a_message_turn(store_factory):
    from gateway.run import GatewayRunner

    source = _make_source()
    store = store_factory()
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner.session_store = store
    runner._session_options_locks = {}
    runner._resolve_session_agent_runtime = lambda **_kwargs: (
        "anthropic/claude-sonnet-4",
        {"provider": "openrouter", "base_url": "", "api_key": ""},
    )
    runner._evict_cached_agent = lambda _session_key: None
    runner._session_db = None

    result = await runner.apply_session_options(
        source,
        {"reasoning_effort": "high", "fast": False, "initial": True},
    )

    assert result["status"] == "accepted"
    assert result["applied"] == ["reasoning_effort", "fast"]
    session_key = result["session_key"]
    assert store.get_runtime_options(session_key) == {
        "model_override": None,
        "reasoning_override": {"enabled": True, "effort": "high"},
        "service_tier_override": "normal",
    }
    assert runner._session_reasoning_overrides[session_key] == {
        "enabled": True,
        "effort": "high",
    }
    assert runner._session_service_tier_overrides[session_key] is None


@pytest.mark.asyncio
async def test_public_session_options_api_uses_native_model_validation(store_factory):
    from gateway.run import GatewayRunner

    source = _make_source()
    store = store_factory()
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner.session_store = store
    runner._session_options_locks = {}
    runner._resolve_session_agent_runtime = lambda **_kwargs: (
        "old-model",
        {"provider": "openrouter", "base_url": "", "api_key": ""},
    )
    runner._evict_cached_agent = lambda _session_key: None
    runner._session_db = None
    selected = SimpleNamespace(
        success=True,
        new_model="gpt-5",
        target_provider="openai",
        api_key="sk-live-only",
        base_url="https://api.openai.com/v1",
        api_mode="responses",
        model_info=None,
        warning_message="",
    )

    with (
        patch("gateway.run._load_gateway_config", return_value={}),
        patch("hermes_cli.model_switch.switch_model", return_value=selected),
        patch(
            "hermes_cli.model_selection_guards.combined_selection_warning",
            return_value=None,
        ),
    ):
        result = await runner.apply_session_options(
            source,
            {
                "model": "gpt-5",
                "provider": "openai",
                "confirm_model_selection": True,
                "initial": True,
            },
        )

    assert result["status"] == "accepted"
    assert result["effective"]["model"] == "gpt-5"
    persisted = store.get_runtime_options(result["session_key"])
    assert persisted["model_override"] == {
        "model": "gpt-5",
        "provider": "openai",
        "base_url": "https://api.openai.com/v1",
    }
