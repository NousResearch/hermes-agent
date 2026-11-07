"""Contract cases for the conversation reads ``LineageRepository.is_compression_child``, ``SessionRepository.records_created_surface`` and ``SessionRepository.last_active``, and the ``last_active`` the listings carry.

Each new operation is fetched through ``operation`` before any seeding.
"""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime

from persistence import SessionId
from persistence.conversation.messages import NewMessage
from persistence.conversation.sessions import END_REASON_COMPRESSION, NewSession, SessionLineageKind, SessionOrder, SessionQuery
from tests.contracts._staged import operation

T0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC).timestamp()
# Outside the trusted epoch window: a garbage double salvaged from a damaged page.
UNTRUSTED = 5e246


def _create(store, session_id: str, source: str = "cli", **fields) -> None:
    store.sessions.create(NewSession(session_id=SessionId(session_id), source=source, **fields))


# ── is_compression_child ──────────────────────────────────────────────────


def test_a_continuation_is_a_child_of_a_compression_ended_parent(backend, scope_a):
    store = backend.open(scope_a)
    is_child = operation(store, "lineage", "is_compression_child")
    _create(store, "P")
    store.sessions.end(SessionId("P"), END_REASON_COMPRESSION)
    _create(store, "Q")
    store.sessions.end(SessionId("Q"), "user_exit")
    children = {
        "plain": ("cli", "P", SessionLineageKind.NONE, None),
        "other-parent": ("cli", "Q", SessionLineageKind.NONE, None),
        "reset": ("cli", "P", SessionLineageKind.RESET, None),
        "branch": ("cli", "P", SessionLineageKind.BRANCH, None),
        "delegate": ("cli", "P", SessionLineageKind.DELEGATE, None),
        "marker-elsewhere": ("cli", "P", SessionLineageKind.NONE, {"_branched_from": "someone-else"}),
        "tool": ("tool", "P", SessionLineageKind.NONE, None),
        "root": ("cli", None, SessionLineageKind.NONE, None),
    }
    for sid, (source, parent, kind, config) in children.items():
        _create(store, sid, source, parent_session_id=SessionId(parent) if parent else None, lineage_kind=kind, model_config=config)
    got = {sid: is_child(store.sessions.get(SessionId(sid))) for sid in children}
    assert got == {
        # CONTRACT: a child that is no fork of a compression-ended parent continues it.
        "plain": True,
        # CONTRACT: any other end reason is not a compression.
        "other-parent": False,
        # CONTRACT: a reset fork of a compression-ended parent is its own conversation.
        "reset": False,
        "branch": False,
        "delegate": False,
        # CONTRACT: a marker counts only when it names the parent (a continuation inherits its parent's markers).
        "marker-elsewhere": True,
        "tool": False,
        "root": False,
    }


def test_the_parent_is_read_when_asked_and_a_missing_parent_is_no_compression(backend, scope_a):
    store = backend.open(scope_a)
    is_child = operation(store, "lineage", "is_compression_child")
    _create(store, "P")
    _create(store, "C", parent_session_id=SessionId("P"))
    record = store.sessions.get(SessionId("C"))
    assert is_child(record) is False
    store.sessions.end(SessionId("P"), END_REASON_COMPRESSION)
    # CONTRACT: the caller's record is not re-read; the parent row is read at call time.
    assert is_child(record) is True
    # CONTRACT: a parent the store does not hold is not a compression parent.
    assert is_child(dataclasses.replace(record, parent_session_id=SessionId("ghost"))) is False


# ── records_created_surface ───────────────────────────────────────────────


def test_a_current_store_records_the_creating_surface(backend, scope_a):
    store = backend.open(scope_a)
    records = operation(store, "sessions", "records_created_surface")
    # CONTRACT: a current store holds created_source (an older file opened read-only answering False is a SQLite-only case).
    assert records() is True
    _create(store, "s1")
    assert records() is True


# ── last_active ───────────────────────────────────────────────────────────


def _message(store, session_id: str, at: float) -> None:
    store.messages.append(SessionId(session_id), NewMessage(role="user", content=f"at {at}", timestamp=at))


def test_last_active_is_the_freshest_trusted_stamp(backend, scope_a):
    store = backend.open(scope_a)
    last_active = operation(store, "sessions", "last_active")
    _create(store, "activity-newer")
    _message(store, "activity-newer", T0 + 10)
    store.sessions.touch_activity(SessionId("activity-newer"), T0 + 50)
    _create(store, "message-newer")
    store.sessions.touch_activity(SessionId("message-newer"), T0 + 10)
    _message(store, "message-newer", T0 + 50)
    _create(store, "neither")
    # CONTRACT: the freshest of the activity stamp and the newest message timestamp.
    assert last_active(SessionId("activity-newer")) == T0 + 50
    assert last_active(SessionId("message-newer")) == T0 + 50
    # CONTRACT: neither present falls back to started_at.
    assert last_active(SessionId("neither")) == store.sessions.get(SessionId("neither")).started_at
    # CONTRACT: a missing row is None.
    assert last_active(SessionId("nobody")) is None


def test_a_cell_outside_the_trusted_window_is_skipped(backend, scope_a):
    store = backend.open(scope_a)
    last_active = operation(store, "sessions", "last_active")
    _create(store, "bad-activity")
    _message(store, "bad-activity", T0 + 20)
    _create(store, "bad-activity-only")
    _create(store, "bad-started")
    rows = backend.scope_store(scope_a)["sessions"]
    rows["bad-activity"]["last_activity_at"] = UNTRUSTED
    rows["bad-activity-only"]["last_activity_at"] = UNTRUSTED
    rows["bad-started"]["started_at"] = UNTRUSTED
    # CONTRACT: an untrusted activity stamp is skipped, not ranked first.
    assert last_active(SessionId("bad-activity")) == T0 + 20
    assert last_active(SessionId("bad-activity-only")) == store.sessions.get(SessionId("bad-activity-only")).started_at
    # CONTRACT: a row with no trusted cell has no recency.
    assert last_active(SessionId("bad-started")) is None


def test_a_listed_row_with_no_trusted_cell_has_no_recency(backend, scope_a):
    store = backend.open(scope_a)
    _create(store, "good")
    _message(store, "good", T0 + 20)
    _create(store, "bad")
    rows = backend.scope_store(scope_a)["sessions"]
    rows["bad"]["last_activity_at"] = UNTRUSTED
    rows["bad"]["started_at"] = UNTRUSTED
    listed = store.sessions.list_sessions(SessionQuery(order=SessionOrder.LAST_ACTIVE))
    recency = store.sessions.search_sessions()
    summary = store.sessions.summary(SessionId("bad"))
    # CONTRACT: a row with no trusted cell lists with no recency, and its listing row carries null.
    assert summary is not None and summary.last_active is None and summary.as_list_row()["last_active"] is None
    assert {str(entry.session.session_id): entry.last_active for entry in listed} == {"good": T0 + 20, "bad": None}
    assert {str(entry.session.session_id): entry.last_active for entry in recency} == {"good": T0 + 20, "bad": None}
    # CONTRACT: a row with no recency ranks below every row with one.
    assert [str(entry.session.session_id) for entry in listed] == ["good", "bad"]
    assert [str(entry.session.session_id) for entry in recency] == ["good", "bad"]
