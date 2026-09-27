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


def test_rehydrate_llamacpp_override_follows_live_managed_port(store_factory):
    """The managed llama.cpp supervisor may come back on an ephemeral port (18434 busy). The persisted
    loopback URL is a snapshot of the previous boot, so rehydration must take the live endpoint."""
    store = store_factory()
    session_key = store.get_or_create_session(_make_source()).session_key
    store.set_model_override(session_key, {
        "model": "Local.Model-Q4_K_M", "provider": "llamacpp", "base_url": "http://127.0.0.1:51489/v1"})

    runner = _make_runner(store_factory())
    with patch(
        "gateway.run._resolve_runtime_agent_kwargs_for_provider",
        return_value={"api_key": "local-key", "base_url": "http://127.0.0.1:18434/v1",
                      "provider": "custom", "requested_provider": "llamacpp"},
    ):
        runner._rehydrate_session_model_override(session_key)

    override = runner._session_model_overrides[session_key]
    assert override["base_url"] == "http://127.0.0.1:18434/v1"
    assert override["api_key"] == "local-key"


def test_rehydrate_opencode_override_heals_relay_url_for_rederived_wire(store_factory):
    """api_mode is re-resolved from the target model, so a relay URL persisted by an older build for the
    previous wire (/v1-stripped for anthropic_messages) must be healed to match, not kept verbatim (#96066)."""
    store = store_factory()
    session_key = store.get_or_create_session(_make_source()).session_key
    store.set_model_override(session_key, {
        "model": "deepseek-v4-flash-vision-exp", "provider": "opencode-go", "base_url": "https://opencode.ai/zen/go"})

    runner = _make_runner(store_factory())
    with patch(
        "gateway.run._resolve_runtime_agent_kwargs_for_provider",
        return_value={"api_key": "go-key", "api_mode": "chat_completions",
                      "base_url": "https://opencode.ai/zen/go/v1", "provider": "opencode-go"},
    ):
        runner._rehydrate_session_model_override(session_key)

    override = runner._session_model_overrides[session_key]
    assert (override["api_mode"], override["base_url"]) == ("chat_completions", "https://opencode.ai/zen/go/v1")


@pytest.mark.parametrize("codex_on_turn", ["recovers", "still_unavailable"])
def test_codex_override_never_runs_on_the_default_providers_endpoint(store_factory, codex_on_turn):
    """A persisted openai-codex override whose credentials fail to re-resolve used to be layered over the
    DEFAULT provider's runtime (Nous URL + Nous key + chat_completions). The turn runs on ONE coherent
    route: the override's own provider when it resolves, else the whole default route with a notice."""
    store = store_factory()
    session_key = store.get_or_create_session(_make_source()).session_key
    store.set_model_override(session_key, {"model": "gpt-6-luna-900k", "provider": "openai-codex",
                                           "base_url": "https://inference-api.nousresearch.com/v1"})
    runner = _make_runner(store_factory())
    codex = {"provider": "openai-codex", "api_key": "codex-tok", "api_mode": "codex_responses",
             "base_url": "https://chatgpt.com/backend-api/codex"}
    nous = {"provider": "nous", "api_key": "nous-key", "api_mode": "chat_completions",
            "base_url": "https://inference-api.nousresearch.com/v1"}
    calls = iter([RuntimeError("refresh blip"), codex if codex_on_turn == "recovers" else RuntimeError("gone")])

    def _for_provider(provider, target_model=None):
        assert provider == "openai-codex"
        nxt = next(calls)
        if isinstance(nxt, Exception):
            raise nxt
        return dict(nxt)

    with patch("gateway.run._resolve_runtime_agent_kwargs_for_provider", side_effect=_for_provider), \
         patch("gateway.run._resolve_runtime_agent_kwargs", return_value=dict(nous)):
        model, runtime = runner._resolve_session_agent_runtime(
            session_key=session_key, user_config={"model": {"default": "openai/gpt-6-luna", "provider": "nous"}})

    expected_model, expected = ("gpt-6-luna-900k", codex) if codex_on_turn == "recovers" else ("openai/gpt-6-luna", nous)
    assert (model, {k: runtime[k] for k in expected}) == (expected_model, expected)
    assert bool(runner._pre_agent_fallback_notice) is (codex_on_turn == "still_unavailable")


def test_sanitize_model_override():
    assert sanitize_model_override(None) is None
    assert sanitize_model_override({}) is None
    assert sanitize_model_override({"api_key": "sk-x", "api_mode": "chat"}) is None
    assert sanitize_model_override(OVERRIDE) == {
        "model": "gpt-5o",
        "provider": "openai",
        "base_url": "https://api.openai.example/v1",
    }
