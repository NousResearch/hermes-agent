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


# -- Shared strict runtime-options write (model + reasoning + /fast tier) -----------------------


@pytest.fixture
def db_store_factory(tmp_path, monkeypatch):
    """SessionStores over a real state.db (primary) plus the sessions.json mirror."""
    import hermes_state

    real_session_db = hermes_state.SessionDB
    db_path = tmp_path / "state.db"
    monkeypatch.setattr(hermes_state, "SessionDB", lambda **_kw: real_session_db(db_path=db_path))
    stores = []

    def _make() -> SessionStore:
        for previous in stores:  # a "restart": the previous process let go of the file
            if previous._db is not None:
                previous._db.close()
        store = SessionStore(sessions_dir=tmp_path / "sessions", config=GatewayConfig())
        assert store._db is not None
        stores.append(store)
        return store

    yield _make
    if stores and stores[-1]._db is not None:
        stores[-1]._db.close()


@pytest.mark.parametrize("tier", ["auto", "cold"])
def test_runtime_options_round_trip_and_reset_clear(db_store_factory, tmp_path, tier):
    store = db_store_factory()
    session_key = store.get_or_create_session(_make_source()).session_key

    assert store.set_runtime_options(
        session_key, model_override=OVERRIDE,
        reasoning_override={"enabled": True, "effort": "high", "extra": "drop"},
        service_tier_override=tier,
    ) is True

    restarted = db_store_factory()
    assert restarted.get_runtime_options(session_key) == {
        "model_override": sanitize_model_override(OVERRIDE),
        "reasoning_override": {"enabled": True, "effort": "high"},
        "service_tier_override": tier,
    }
    # Credentials reach neither the primary copy nor the legacy mirror.
    durable = restarted._routing_db.load_gateway_routing_entries(scope=restarted._routing_scope())
    assert "sk-SUPER-SECRET" not in json.dumps(durable)
    assert "sk-SUPER-SECRET" not in (tmp_path / "sessions" / "sessions.json").read_text(encoding="utf-8")

    # Explicit normal stays distinct from inherit across a restart.
    assert restarted.set_runtime_options(session_key, service_tier_override="normal") is True
    assert db_store_factory().get_runtime_options(session_key)["service_tier_override"] == "normal"

    # /new publishes a fresh entry: all three fields are gone, on disk too.
    reset_store = db_store_factory()
    reset_store.reset_session(session_key)
    assert db_store_factory().get_runtime_options(session_key) == {
        "model_override": None, "reasoning_override": None, "service_tier_override": None,
    }


def test_state_db_failure_is_not_hidden_by_json_mirror(db_store_factory, monkeypatch):
    """F1 in its state.db form: state.db (the primary copy) fails while the sessions.json mirror
    is healthy. On main the mirror save hides that failure and a restart brings back the old
    value. The strict write must raise instead, with memory unchanged and a restart matching it."""
    store = db_store_factory()
    session_key = store.get_or_create_session(_make_source()).session_key
    assert store.set_runtime_options(session_key, reasoning_override={"enabled": True, "effort": "low"})

    def _db_down(*_a, **_kw):
        raise RuntimeError("database is locked")

    with monkeypatch.context() as m:
        m.setattr(store._routing_db, "replace_gateway_routing_entries", _db_down)
        with pytest.raises(OSError, match="state.db routing save failed"):
            store.set_runtime_options(
                session_key, model_override=OVERRIDE,
                reasoning_override={"enabled": True, "effort": "max"}, service_tier_override="cold",
            )
    in_memory = store.get_runtime_options(session_key)
    assert in_memory == {
        "model_override": None,
        "reasoning_override": {"enabled": True, "effort": "low"},
        "service_tier_override": None,
    }
    assert db_store_factory().get_runtime_options(session_key) == in_memory
