"""Phase 3 (drain-window recovery): the INV-9 answerability primitive.

Unit-proves SessionStore.has_platform_message_id_answerable returns the
tri-state (answered, present) — distinguishing "provably absent" from
"could not determine" — which the backfill relies on to fail toward no-dup.
"""
from unittest.mock import patch

import pytest


def _make_store(tmp_path, with_db=True):
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from hermes_state import SessionDB

    config = GatewayConfig()
    with patch("gateway.session.SessionStore._ensure_loaded"):
        store = SessionStore(sessions_dir=tmp_path / "sessions", config=config)
    if with_db:
        db = SessionDB(db_path=tmp_path / "state.db")
        store._db = db
    else:
        store._db = None
    store._loaded = True
    return store


def test_answerable_present_true(tmp_path):
    store = _make_store(tmp_path)
    store._db.create_session(session_id="s1", source="test")
    store._db.append_message(
        session_id="s1", role="user", content="hi", platform_message_id="42"
    )
    answered, present = store.has_platform_message_id_answerable("s1", "42")
    assert (answered, present) == (True, True)


def test_answerable_positively_absent(tmp_path):
    store = _make_store(tmp_path)
    store._db.create_session(session_id="s1", source="test")
    answered, present = store.has_platform_message_id_answerable("s1", "does-not-exist")
    assert (answered, present) == (True, False), "a real DB miss must be answered=True, present=False"


def test_unanswerable_when_no_db(tmp_path):
    store = _make_store(tmp_path, with_db=False)
    answered, present = store.has_platform_message_id_answerable("s1", "42")
    assert (answered, present) == (False, False), "no DB ⇒ cannot answer"


def test_unanswerable_when_lookup_raises(tmp_path):
    store = _make_store(tmp_path)

    def _boom(*a, **k):
        raise RuntimeError("db locked")
    store._db.has_platform_message_id = _boom

    answered, present = store.has_platform_message_id_answerable("s1", "42")
    assert (answered, present) == (False, False), "a raised lookup ⇒ cannot answer"


def test_plain_wrapper_still_collapses_to_false(tmp_path):
    """Regression: the existing plain has_platform_message_id keeps its
    fail-to-False behavior (the #47237 re-persist guard depends on it)."""
    store = _make_store(tmp_path, with_db=False)
    assert store.has_platform_message_id("s1", "42") is False
