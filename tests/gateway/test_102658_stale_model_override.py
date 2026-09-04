"""Stale session-model pins follow fleet default-model changes (#102658).

A ``/model`` pin that merely echoed the config default goes stale when the
operator migrates ``model.default``/``model.provider``: the runner drops it
(live + post-restart) so the session follows the new default.  An explicit
divergent pin still wins; legacy pins without a snapshot keep winning.
"""
from unittest.mock import patch

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.session import SessionSource, SessionStore

OLD = {"model": {"default": "old-m", "provider": "old-p"}}
NEW = {"model": {"default": "new-m", "provider": "new-p"}}


@pytest.fixture
def store(tmp_path, monkeypatch):
    import hermes_state

    def _raise(*a, **k):
        raise RuntimeError("SQLite disabled in test")

    monkeypatch.setattr(hermes_state, "SessionDB", _raise)
    return SessionStore(sessions_dir=tmp_path, config=GatewayConfig())


def _key(store):
    src = SessionSource(platform=Platform.TELEGRAM, user_id="u", chat_id="c",
                        user_name="t", chat_type="dm")
    return store.get_or_create_session(src).session_key


def _runner(store):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.session_store = store
    return runner


def _rehydrate(store, key, cfg):
    runner = _runner(store)
    with patch("hermes_cli.config.load_config", return_value=cfg):
        runner._rehydrate_session_model_override(key)
    return runner


def _live_override(runner, key):
    state = runner._peek_session_state(key)
    return state.conversation.model_override if state else None


def test_stale_echo_pin_follows_default_change(store):
    key = _key(store)
    with patch("hermes_cli.config.load_config", return_value=OLD):
        store.set_model_override(key, {"model": "old-m", "provider": "old-p"})
    runner = _rehydrate(store, key, NEW)
    assert _live_override(runner, key) is None
    assert store.get_model_override(key) is None


def test_explicit_divergent_pin_survives_default_change(store):
    key = _key(store)
    with patch("hermes_cli.config.load_config", return_value=OLD):
        store.set_model_override(key, {"model": "other-m", "provider": "other-p"})
    runner = _rehydrate(store, key, NEW)
    assert (_live_override(runner, key) or {}).get("model") == "other-m"


def test_legacy_pin_without_snapshot_keeps_winning(store):
    key = _key(store)
    with patch("hermes_cli.config.load_config", return_value={}):
        store.set_model_override(key, {"model": "old-m", "provider": "old-p"})
    runner = _rehydrate(store, key, NEW)
    assert (_live_override(runner, key) or {}).get("model") == "old-m"
