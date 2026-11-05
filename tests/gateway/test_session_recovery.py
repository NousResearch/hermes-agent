"""Tests for SessionRecoveryMixin.resolve_session_id_for_key — flush-recovery
session_key → session_id resolution (profile aware)."""

from types import SimpleNamespace
from unittest.mock import patch

from gateway.config import GatewayConfig
from gateway.session import SessionStore


class _FakeGatewayDB:
    """Minimal store stand-in (legacy handle and storage handle) exposing only the peer finder."""

    def __init__(self, rows_by_key):
        self.rows_by_key, self.queries, self.routing = rows_by_key, [], self

    def latest_session_for_peer(self, *, source, session_key=None, **kwargs):
        self.queries.append(session_key)
        row = self.rows_by_key.get(session_key)
        return row and SimpleNamespace(has_messages=False, session=SimpleNamespace(
            session_id=row["id"], session_key=session_key, started_at=row.get("started_at"),
            last_activity_at=None, message_count=0, origin=None))


def _store(tmp_path, db):
    config = GatewayConfig(multiplex_profiles=True)
    with patch("gateway.session.SessionStore._ensure_loaded"):
        store = SessionStore(sessions_dir=tmp_path, config=config)
    store._loaded = True
    store._db = db  # pin: _persistence_for_key returns the handle beside this db for every key (no profile I/O)
    store._persistence_beside = lambda pinned: pinned
    return store


def test_resolve_profile_namespaced_key_does_not_adopt_main_row(tmp_path, monkeypatch):
    """A profile-namespaced key must never resolve into the root store's
    ``agent:main`` row: the exact-key finder only matches the namespaced key, and an
    unresolvable profile store (``_db_for_key`` → None) yields None, never the root store."""
    db = _FakeGatewayDB({"agent:main:whatsapp:dm:15551234567": {"id": "main-row"}})
    store = _store(tmp_path, db)
    assert (
        store.resolve_session_id_for_key("agent:test-bot:whatsapp:dm:15551234567")
        is None
    )
    assert db.queries and all("agent:test-bot" in q for q in db.queries)
    assert not any(q.startswith("agent:main") for q in db.queries)
    # Fail-closed store: when the profile's home is unresolvable, _persistence_for_key returns None and the
    # resolver must answer None too — even with a routing-map hit — so the flush file is preserved
    # instead of being appended to the ambient root store (#66887/#102157 split-identity class).
    key = "agent:test-bot:whatsapp:dm:15551234567"
    monkeypatch.setattr(store, "peek_session_id", lambda session_key: "routed-sid")
    monkeypatch.setattr(store, "_persistence_for_key", lambda session_key: None)
    assert store.resolve_session_id_for_key(key) is None
    assert store.resolve_session_id_for_key(key, not_after=1700000000) is None


def test_resolve_db_fallback_rejects_row_started_after_flush(tmp_path):
    """A row created after the flush timestamp cannot be the message's origin; only a row that
    already existed at flush time is adopted, and it comes back with its owning db."""
    key = "agent:main:telegram:dm:42"
    db = _FakeGatewayDB({key: {"id": "late-row", "started_at": 1700000100.0}})
    store = _store(tmp_path, db)
    assert store.resolve_session_id_for_key(key, not_after=1700000000) is None
    assert store.resolve_session_id_for_key(key, not_after=1700000100) == ("late-row", db)
    # The flush ``ts`` is a whole second while ``started_at`` is a REAL: a row minted later in the
    # SAME second as the flush is still a valid origin, not a post-flush row.
    same_second = _FakeGatewayDB({key: {"id": "same-sec", "started_at": 1700000000.818}})
    assert _store(tmp_path / "b", same_second).resolve_session_id_for_key(key, not_after=1700000000) == ("same-sec", same_second)
