"""Tests for SessionRecoveryMixin.resolve_session_id_for_key — flush-recovery
session_key → session_id resolution (profile- and WhatsApp-alias aware)."""

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


def _no_aliases(identifier):
    return {identifier}


def test_resolve_exact_key_returns_row_id(tmp_path):
    db = _FakeGatewayDB({"agent:main:telegram:dm:99": {"id": "tel-row"}})
    store = _store(tmp_path, db)
    assert store.resolve_session_id_for_key("agent:main:telegram:dm:99") == "tel-row"
    assert db.queries == ["agent:main:telegram:dm:99"]


def test_resolve_whatsapp_phone_key_to_lid_row(tmp_path, monkeypatch):
    lid = "1234567890123456789"
    key = f"agent:test-bot:whatsapp:dm:{lid}"
    db = _FakeGatewayDB({key: {"id": "lid-row"}})
    store = _store(tmp_path, db)
    monkeypatch.setattr(
        "gateway.whatsapp_identity.expand_whatsapp_aliases",
        lambda ident: {"15551234567", lid} if ident == "15551234567" else {ident},
    )
    resolved = store.resolve_session_id_for_key(
        "agent:test-bot:whatsapp:dm:15551234567"
    )
    assert resolved == "lid-row"
    assert key in db.queries


def test_resolve_profile_namespaced_key_does_not_adopt_main_row(tmp_path, monkeypatch):
    """A profile-namespaced key must never resolve into the root store's
    ``agent:main`` row: the exact-key finder only matches the namespaced key."""
    db = _FakeGatewayDB({"agent:main:whatsapp:dm:15551234567": {"id": "main-row"}})
    store = _store(tmp_path, db)
    monkeypatch.setattr("gateway.whatsapp_identity.expand_whatsapp_aliases", _no_aliases)
    assert (
        store.resolve_session_id_for_key("agent:test-bot:whatsapp:dm:15551234567")
        is None
    )
    assert db.queries and all("agent:test-bot" in q for q in db.queries)
    assert not any(q.startswith("agent:main") for q in db.queries)


def test_resolve_returns_none_when_no_matching_row(tmp_path, monkeypatch):
    db = _FakeGatewayDB({})
    store = _store(tmp_path, db)
    monkeypatch.setattr("gateway.whatsapp_identity.expand_whatsapp_aliases", _no_aliases)
    assert (
        store.resolve_session_id_for_key("agent:test-bot:whatsapp:dm:15551234567")
        is None
    )
