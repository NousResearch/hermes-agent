"""Contract cases for the receiving adapter profile a gateway ingress stamps on a session row: ``NewSession.transport_profile`` on create and ``record_session_peer(transport_profile=)``, read back as ``SessionRecord.transport_profile``.

Every case runs on every registered backend, seeds through public protocol operations only and observes through ``list_sessions``.
"""

from __future__ import annotations

from typing import Optional

from persistence import SessionId
from persistence.conversation.sessions import NewSession, SessionQuery, SessionWindow


def _transport(backend, scope, sid: str) -> Optional[str]:
    store = backend.open(scope)
    rows = store.sessions.list_sessions(SessionQuery(), SessionWindow(limit=50))
    [row] = [r for r in rows if r.session.session_id == sid]
    return row.session.transport_profile


def _peer(store, sid: str, **kw) -> None:
    store.routing.record_session_peer(sid, source="telegram", session_key="agent:main:telegram:dm:1", chat_id="1", chat_type="dm", **kw)


def test_create_stamps_transport_profile_and_upsert_keeps_the_first_writer(backend, scope_a):
    store_a = backend.open(scope_a)
    sid = SessionId("tp-create")
    store_a.sessions.create(NewSession(session_id=sid, source="telegram", transport_profile="bot-a"))
    assert _transport(backend, scope_a, sid) == "bot-a"
    # Upsert: an existing value is never overwritten, and a create carrying none leaves it.
    store_a.sessions.create(NewSession(session_id=sid, source="telegram", transport_profile="bot-b"))
    store_a.sessions.create(NewSession(session_id=sid, source="telegram"))
    assert _transport(backend, scope_a, sid) == "bot-a"


def test_upsert_fills_a_row_created_without_transport_profile(backend, scope_a):
    store_a = backend.open(scope_a)
    sid = SessionId("tp-fill")
    store_a.sessions.create(NewSession(session_id=sid, source="telegram"))
    assert _transport(backend, scope_a, sid) is None
    store_a.sessions.create(NewSession(session_id=sid, source="telegram", transport_profile="bot-a"))
    assert _transport(backend, scope_a, sid) == "bot-a"


def test_peer_verb_replaces_on_value_and_keeps_on_none(backend, scope_a):
    store_a = backend.open(scope_a)
    sid = SessionId("tp-peer")
    store_a.sessions.create(NewSession(session_id=sid, source="telegram"))
    _peer(store_a, sid, transport_profile="bot-a")
    assert _transport(backend, scope_a, sid) == "bot-a"
    # A per-turn refresh that knows no transport leaves the stamp.
    _peer(store_a, sid)
    assert _transport(backend, scope_a, sid) == "bot-a"
    _peer(store_a, sid, transport_profile="bot-b")
    assert _transport(backend, scope_a, sid) == "bot-b"


def test_peer_verb_self_heal_insert_carries_transport_profile(backend, scope_a):
    store_a = backend.open(scope_a)
    sid = SessionId("tp-heal")
    _peer(store_a, sid, transport_profile="bot-a")
    assert _transport(backend, scope_a, sid) == "bot-a"


def test_transport_profile_is_scope_isolated(backend, scope_a, scope_b):
    store_a = backend.open(scope_a)
    store_b = backend.open(scope_b)
    sid = SessionId("tp-iso")
    store_a.sessions.create(NewSession(session_id=sid, source="telegram", transport_profile="bot-a"))
    store_b.sessions.create(NewSession(session_id=sid, source="telegram"))
    assert _transport(backend, scope_b, sid) is None
    _peer(store_b, sid, transport_profile="bot-b")
    assert _transport(backend, scope_a, sid) == "bot-a"
    assert _transport(backend, scope_b, sid) == "bot-b"
