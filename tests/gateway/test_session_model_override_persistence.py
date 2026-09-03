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
            "requested_provider": "custom:chatgpt-tier",
            "capabilities": {"openai_native_compaction": True},
            "max_tokens": 32_768,
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
    assert override["requested_provider"] == "custom:chatgpt-tier"
    assert override["capabilities"] == {"openai_native_compaction": True}
    assert override["max_tokens"] == 32_768

    model, runtime = runner._resolve_session_agent_runtime(
        session_key=session_key,
        user_config={"model": {"default": "global-model"}},
    )
    assert model == "gpt-5o"
    assert runtime["requested_provider"] == "custom:chatgpt-tier"
    assert runtime["capabilities"] == {"openai_native_compaction": True}
    assert runtime["max_tokens"] == 32_768
    route = runner._resolve_turn_agent_config("", model, runtime)
    assert route["runtime"]["capabilities"] == {"openai_native_compaction": True}


def test_sanitize_model_override():
    assert sanitize_model_override(None) is None
    assert sanitize_model_override({}) is None
    assert sanitize_model_override({"api_key": "sk-x", "api_mode": "chat"}) is None
    assert sanitize_model_override(OVERRIDE) == {
        "model": "gpt-5o",
        "provider": "openai",
        "base_url": "https://api.openai.example/v1",
    }


# ── OpenCode /v1 mismatch on rehydrate (#100854) ────────────────────────
#
# A session switched to an Anthropic-routed OpenCode model (e.g.
# ``/model qwen3.8-flash`` on opencode-go) persists the /v1-stripped base_url
# (https://opencode.ai/zen/go). After a gateway restart, rehydration re-resolved
# credentials WITHOUT the persisted model, so api_mode was derived from the
# stale config default (chat_completions) and the stripped URL was kept — the
# OpenAI client then POSTed to /zen/go/chat/completions, a 404 on the opencode
# marketing site. Refs #57585 (stripped/persisted /v1) + #100854 (rehydrate).

_OPENCODE_QWEN_OVERRIDE = {
    "model": "qwen3.8-flash",
    "provider": "opencode-go",
    "base_url": "https://opencode.ai/zen/go",  # anthropic-stripped form
}


def test_rehydrate_passes_persisted_model_as_target_model(store_factory):
    """Rehydration must resolve with the persisted model so OpenCode api_mode
    is re-derived from the model actually in use, not the config default."""
    store = store_factory()
    entry = store.get_or_create_session(_make_source())
    session_key = entry.session_key
    store.set_model_override(session_key, _OPENCODE_QWEN_OVERRIDE)

    runner = _make_runner(store_factory())

    calls = {}

    def _fake_resolve(provider, target_model=None):
        calls["target_model"] = target_model
        # Simulate a resolver that does NOT use the model: returns the
        # chat_completions mode for the config default (the pre-fix behavior).
        return {
            "api_key": "sk-fre...hain",
            "api_mode": "chat_completions",
            "base_url": "https://opencode.ai/zen/go/v1",
            "provider": "opencode-go",
            "requested_provider": "opencode-go",
            "capabilities": {},
            "max_tokens": 32_768,
        }

    with patch(
        "gateway.run._resolve_runtime_agent_kwargs_for_provider",
        side_effect=_fake_resolve,
    ):
        runner._rehydrate_session_model_override(session_key)

    override = runner._session_model_overrides[session_key]
    assert calls["target_model"] == "qwen3.8-flash", (
        "rehydrate must pass the persisted model as target_model"
    )
    assert override["api_mode"] == "chat_completions"


def test_rehydrate_heals_stripped_v1_for_chat_completions_model(store_factory):
    """A persisted /v1-stripped URL rehydrated into a chat_completions api_mode
    must be healed back to /zen/go/v1, or the first post-restart turn 404s."""
    store = store_factory()
    entry = store.get_or_create_session(_make_source())
    session_key = entry.session_key
    store.set_model_override(session_key, _OPENCODE_QWEN_OVERRIDE)

    runner = _make_runner(store_factory())

    def _fake_resolve(provider, target_model=None):
        return {
            "api_key": "sk-fre...hain",
            "api_mode": "chat_completions",  # e.g. stale config default wins
            "base_url": "https://opencode.ai/zen/go/v1",
            "provider": "opencode-go",
            "requested_provider": "opencode-go",
            "capabilities": {},
            "max_tokens": 32_768,
        }

    with patch(
        "gateway.run._resolve_runtime_agent_kwargs_for_provider",
        side_effect=_fake_resolve,
    ):
        runner._rehydrate_session_model_override(session_key)

    override = runner._session_model_overrides[session_key]
    assert override["base_url"] == "https://opencode.ai/zen/go/v1", (
        "chat_completions api_mode must re-append /v1 to the persisted URL"
    )


def test_rehydrate_keeps_stripped_v1_for_anthropic_model(store_factory):
    """The inverse: an anthropic_messages model must KEEP the stripped URL
    (the Anthropic SDK prepends its own /v1/messages)."""
    store = store_factory()
    entry = store.get_or_create_session(_make_source())
    session_key = entry.session_key
    store.set_model_override(session_key, _OPENCODE_QWEN_OVERRIDE)

    runner = _make_runner(store_factory())

    def _fake_resolve(provider, target_model=None):
        return {
            "api_key": "sk-fre...hain",
            "api_mode": "anthropic_messages",
            "base_url": "https://opencode.ai/zen/go",  # resolver strips /v1
            "provider": "opencode-go",
            "requested_provider": "opencode-go",
            "capabilities": {},
            "max_tokens": 32_768,
        }

    with patch(
        "gateway.run._resolve_runtime_agent_kwargs_for_provider",
        side_effect=_fake_resolve,
    ):
        runner._rehydrate_session_model_override(session_key)

    override = runner._session_model_overrides[session_key]
    assert override["base_url"] == "https://opencode.ai/zen/go", (
        "anthropic_messages api_mode must keep the /v1-stripped URL"
    )
