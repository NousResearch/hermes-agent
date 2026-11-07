"""Ambient-served sessions must resume on their LAST-SERVED model after a gateway restart.

Only an explicit ``/model`` used to be persisted (``SessionEntry.model_override``). A session
served by the ambient/config default — with no mid-session ``/model`` — therefore had nothing
to restore, so after a restart the resumed turn was silently re-derived from the restarted
gateway's warmup model (often a lower-tier one). The identity that actually served the last
successful turn is now recorded as ``SessionEntry.last_served`` (non-secret keys only) and
rehydrated on first use, flagged ``restored_from_served`` so it is never mistaken for a
user-chosen override.

Covers:
  - an ambient-served identity persists and is restored on a simulated restart
  - the restored identity is flagged, not re-persisted as an explicit override
  - a last-served provider unavailable after restart keeps the ambient route (no dead identity)
  - an explicit /model still wins over a previously-restored ambient identity
  - api_key is NEVER serialized
"""
from unittest.mock import patch

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.session import SessionSource, SessionStore


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.DISCORD,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


SERVED = {
    "model": "upstage/solar-pro4:free",
    "provider": "openrouter",
    "base_url": "https://openrouter.ai/api/v1",
    "api_key": "sk-SECRET-do-not-persist",
}

WARMUP_RUNTIME = {
    "api_key": "live-cred",
    "api_mode": "chat_completions",
    "base_url": "https://openrouter.ai/api/v1",
    "provider": "openrouter",
    "requested_provider": "openrouter",
    "capabilities": {},
    "max_tokens": 32_768,
}


@pytest.fixture
def store_factory(tmp_path, monkeypatch):
    def _raise(*a, **k):
        raise RuntimeError("SQLite disabled in test")

    import hermes_state

    monkeypatch.setattr(hermes_state, "SessionDB", _raise)

    def _make() -> SessionStore:
        store = SessionStore(sessions_dir=tmp_path, config=GatewayConfig())
        assert store._db is None
        return store

    return _make


def _make_runner(store):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._session_model_overrides = {}
    runner.session_store = store
    return runner


def test_last_served_persists_without_explicit_override(store_factory):
    store = store_factory()
    session_key = store.get_or_create_session(_make_source()).session_key
    # No explicit /model: only the ambient-served identity is recorded.
    store.set_last_served(session_key, SERVED)
    assert store.get_model_override(session_key) is None

    store2 = store_factory()  # simulated restart reads the same dir
    assert store2.get_last_served(session_key) == {
        "model": "upstage/solar-pro4:free",
        "provider": "openrouter",
        "base_url": "https://openrouter.ai/api/v1",
    }


def test_runner_resumes_ambient_session_on_last_served_model(store_factory):
    store = store_factory()
    session_key = store.get_or_create_session(_make_source()).session_key
    store.set_last_served(session_key, SERVED)

    runner = _make_runner(store_factory())
    with patch(
        "gateway.run._resolve_runtime_agent_kwargs_for_provider",
        return_value=dict(WARMUP_RUNTIME),
    ):
        runner._rehydrate_served_identity(session_key)

    state = runner._peek_session_state(session_key)
    assert state.conversation.restored_from_served is True
    restored = state.conversation.model_override
    assert restored["model"] == "upstage/solar-pro4:free"
    assert restored["api_key"] == "live-cred"  # credentials from live resolution, never disk

    # The full runtime resolution serves the resumed turn on the restored model, not the warmup one.
    with patch(
        "gateway.run._resolve_runtime_agent_kwargs_for_provider",
        return_value=dict(WARMUP_RUNTIME),
    ):
        model, _runtime = runner._resolve_session_agent_runtime(
            session_key=session_key,
            user_config={"model": {"default": "z-ai/glm-5.3-flash"}},
        )
    assert model == "upstage/solar-pro4:free"


def test_unavailable_last_served_provider_keeps_ambient_route(store_factory):
    store = store_factory()
    session_key = store.get_or_create_session(_make_source()).session_key
    store.set_last_served(session_key, SERVED)

    runner = _make_runner(store_factory())
    with patch(
        "gateway.run._resolve_runtime_agent_kwargs_for_provider",
        side_effect=RuntimeError("provider gone"),
    ):
        runner._rehydrate_served_identity(session_key)

    state = runner._peek_session_state(session_key)
    # A dead serving identity must not be restored: nothing armed, ambient model stays.
    assert state.conversation.model_override is None
    assert state.conversation.restored_from_served is False


def test_explicit_override_wins_over_restored_ambient_identity(store_factory):
    store = store_factory()
    session_key = store.get_or_create_session(_make_source()).session_key
    store.set_last_served(session_key, SERVED)

    runner = _make_runner(store_factory())
    with patch(
        "gateway.run._resolve_runtime_agent_kwargs_for_provider",
        return_value=dict(WARMUP_RUNTIME),
    ):
        runner._rehydrate_served_identity(session_key)
    assert runner._peek_session_state(session_key).conversation.restored_from_served is True

    # The user then explicitly picks a model: persist it on a fresh store (as a /model command does).
    explicit = {
        "model": "gpt-5o",
        "provider": "openai",
        "base_url": "https://api.openai.example/v1",
    }
    store_explicit = store_factory()
    store_explicit.set_model_override(session_key, explicit)
    runner.session_store = store_explicit

    explicit_runtime = {
        "api_key": "openai-cred", "api_mode": "responses",
        "base_url": "https://api.openai.example/v1", "provider": "openai",
        "requested_provider": "openai", "capabilities": {}, "max_tokens": 16_384,
    }
    with patch(
        "gateway.run._resolve_runtime_agent_kwargs_for_provider",
        return_value=explicit_runtime,
    ):
        runner._rehydrate_session_model_override(session_key)

    state = runner._peek_session_state(session_key)
    assert state.conversation.restored_from_served is False
    assert state.conversation.model_override["model"] == "gpt-5o"
