"""A /model pin belongs to the chat route (session_key), not to one session_id (#119864).

``SessionStore.switch_session()`` re-points a key at another session id for non-boundary
reasons too (async-delegation re-pin, compression-tip binding heal, CLI handoff, /branch), so
the persisted ``model_override`` must ride along. ``reset_session()`` (/new) is a deliberate
conversation boundary and must keep dropping it — otherwise the next turn's
``_rehydrate_session_model_override`` resurrects the override the user reset away.
"""
from unittest.mock import patch

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.session import SessionSource, SessionStore

OVERRIDE = {"model": "nous/hermes-4", "provider": "nous"}


@pytest.fixture
def store(tmp_path, monkeypatch):
    import hermes_state

    def _raise():
        raise RuntimeError("SQLite disabled in test")

    monkeypatch.setattr(hermes_state, "SessionDB", _raise)
    with patch("gateway.session.SessionStore._ensure_loaded"):
        s = SessionStore(sessions_dir=tmp_path / "sessions", config=GatewayConfig())
    s._loaded = True
    return s


def _pinned_key(store):
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="c1", chat_type="dm",
                           user_id="u1", user_name="tester")
    key = store.get_or_create_session(source).session_key
    store.set_model_override(key, OVERRIDE)
    assert store.get_model_override(key) == OVERRIDE
    return key


def test_switch_session_keeps_model_override_on_the_route(store):
    key = _pinned_key(store)
    switched = store.switch_session(key, "other_session_id")
    assert switched is not None and switched.session_id == "other_session_id"
    assert switched.model_override == OVERRIDE
    assert store.get_model_override(key) == OVERRIDE


def test_reset_session_still_drops_persisted_model_override(store):
    """Control: /new is a boundary; a restart or the next turn must not resurrect the pin."""
    key = _pinned_key(store)
    fresh = store.reset_session(key)
    assert fresh is not None and fresh.is_fresh_reset
    assert fresh.model_override is None
    assert store.get_model_override(key) is None
