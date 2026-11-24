"""A reopen must not erase the compression stamp a continuation hangs off (#148047).

Every case runs against every registered backend through public protocol verbs only.
"""

from __future__ import annotations

from persistence import SessionId
from persistence.conversation.sessions import END_REASON_ADOPTED, END_REASON_COMPRESSION, NewSession, SessionLineageKind


def _new(session_id: str, **kwargs) -> NewSession:
    return NewSession(session_id=SessionId(session_id), source="cli", **kwargs)


def _compressed(store, parent: str, child: str) -> None:
    store.sessions.create(_new(parent))
    store.sessions.end(SessionId(parent), END_REASON_COMPRESSION)
    store.sessions.create(_new(child, parent_session_id=SessionId(parent)))


def test_reopen_keeps_the_compression_stamp_a_continuation_hangs_off(backend, scope_a):
    store = backend.open(scope_a)
    _compressed(store, "rcb-p", "rcb-c")
    before = store.sessions.get(SessionId("rcb-p"))

    store.sessions.reopen(SessionId("rcb-p"))

    # CONTRACT: the stamp and its time survive, so the chain still walks to the continuation.
    after = store.sessions.get(SessionId("rcb-p"))
    assert after.end_reason == END_REASON_COMPRESSION and after.ended_at == before.ended_at
    assert store.lineage.tip(SessionId("rcb-p")) == SessionId("rcb-c")


def test_reopen_clears_a_compression_stamp_with_only_a_fork_child(backend, scope_a):
    store = backend.open(scope_a)
    store.sessions.create(_new("rcb-o"))
    store.sessions.end(SessionId("rcb-o"), END_REASON_COMPRESSION)
    store.sessions.create(_new("rcb-o-b", parent_session_id=SessionId("rcb-o"), lineage_kind=SessionLineageKind.BRANCH))

    store.sessions.reopen(SessionId("rcb-o"))

    row = store.sessions.get(SessionId("rcb-o"))
    assert row.ended_at is None and row.end_reason is None


def test_reopen_clears_other_reasons_and_drop_lets_adoption_restamp(backend, scope_a):
    store = backend.open(scope_a)
    store.sessions.create(_new("rcb-x"))
    store.sessions.end(SessionId("rcb-x"), "tui_close")
    store.sessions.reopen(SessionId("rcb-x"))
    assert store.sessions.get(SessionId("rcb-x")).end_reason is None

    # CONTRACT: the adoption donor retirement clears the stamp on purpose and lands its own.
    _compressed(store, "rcb-d", "rcb-d-c")
    store.sessions.reopen(SessionId("rcb-d"), drop_compression_boundary=True)
    store.sessions.end(SessionId("rcb-d"), END_REASON_ADOPTED)
    assert store.sessions.get(SessionId("rcb-d")).end_reason == END_REASON_ADOPTED
