"""Tests for SessionRecoveryMixin.resolve_session_id_for_key — flush-recovery
session_key → session_id resolution (profile aware)."""

from unittest.mock import patch

from gateway.config import GatewayConfig
from gateway.session import SessionStore


class _FakeGatewayDB:
    """Minimal SessionDB stand-in exposing only the peer finder the resolver uses."""

    def __init__(self, rows_by_key):
        self.rows_by_key = rows_by_key
        self.queries = []

    def find_latest_gateway_session_for_peer(self, *, source, session_key=None, **kwargs):
        self.queries.append(session_key)
        return self.rows_by_key.get(session_key)


def _store(tmp_path, db):
    config = GatewayConfig(multiplex_profiles=True)
    with patch("gateway.session.SessionStore._ensure_loaded"):
        store = SessionStore(sessions_dir=tmp_path, config=config)
    store._loaded = True
    store._db = db  # pin: _db_for_key returns this db for every key (no profile I/O)
    return store


def test_resolve_profile_namespaced_key_does_not_adopt_main_row(tmp_path):
    """A profile-namespaced key must never resolve into the root store's
    ``agent:main`` row: the exact-key finder only matches the namespaced key."""
    db = _FakeGatewayDB({"agent:main:whatsapp:dm:15551234567": {"id": "main-row"}})
    store = _store(tmp_path, db)
    assert (
        store.resolve_session_id_for_key("agent:test-bot:whatsapp:dm:15551234567")
        is None
    )
    assert db.queries and all("agent:test-bot" in q for q in db.queries)
    assert not any(q.startswith("agent:main") for q in db.queries)


def test_resolve_db_fallback_rejects_row_started_after_flush(tmp_path):
    """A row created after the flush timestamp cannot be the message's origin; only a row that
    already existed at flush time is adopted, and it comes back with its owning db."""
    key = "agent:main:telegram:dm:42"
    db = _FakeGatewayDB({key: {"id": "late-row", "started_at": 1700000100.0}})
    store = _store(tmp_path, db)
    assert store.resolve_session_id_for_key(key, not_after=1700000000) is None
    assert store.resolve_session_id_for_key(key, not_after=1700000100) == ("late-row", db)
