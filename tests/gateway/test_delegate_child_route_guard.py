"""A delegate child must never win gateway routing recovery for its parent's chat (#138210).

Incident shape (2026-10-14, production Telegram DM thread): a background ``delegate_task``
child session carried its parent's routing identity (``session_key`` / ``chat_id`` /
``thread_id`` / ``user_id``), so ``find_latest_gateway_session_for_peer`` ranked the
subagent as the chat's newest live session. The user's follow-up messages in that thread
were then routed into the subagent, which answered as if it were the main agent — and the
parent, having ended ``session_switch`` while children still ran, could not re-take the
thread.

The create path already refuses to copy routing columns into a fork
(``tests/hermes_state/test_delegate_child_routing_inheritance.py``, #92859/#116322), but
recovery had no equivalent guard, so a row that predates that fix — or any writer that set
the columns, or a restored DB, whose values the sessions upsert preserves — keeps the
poison live. Recovery is the last gate before user traffic is routed, so it defends the
conversation against rows from every writer, past and future.

The exclusion is anchored on the ``model_config`` lineage marker, not the ``source`` label:
a delegate child built on a gateway turn is stamped with the parent's source
(``session_source_for`` reads the session env before the platform argument), so a source
filter would not exclude it and would also misclassify one-shot children (#112550).

Real ``SessionDB`` on a temp path, no mocks.
"""

from __future__ import annotations

import uuid

import pytest

from hermes_state import SessionDB

THREAD_ID = "54287"
PEER = dict(
    source="telegram",
    user_id="u1",
    session_key="agent:main:telegram:dm:42",
    chat_id="42",
    chat_type="dm",
    thread_id=THREAD_ID,
)


@pytest.fixture()
def db(tmp_path):
    session_db = SessionDB(db_path=tmp_path / "state.db")
    try:
        yield session_db
    finally:
        session_db.close()


def _parent(db: SessionDB) -> str:
    sid = "20261014_020931_" + uuid.uuid4().hex[:6]
    db.create_session(
        sid, source="telegram", session_key=PEER["session_key"], chat_id=PEER["chat_id"],
        chat_type=PEER["chat_type"], thread_id=THREAD_ID, user_id=PEER["user_id"],
    )
    db.append_message(sid, "user", "host apps on the cluster with CLI access")
    return sid


def _delegate_child(db: SessionDB, parent_id: str) -> str:
    """A child holding the parent's full routing identity — the reporter's row."""
    sid = "20261014_040222_" + uuid.uuid4().hex[:6]
    db.create_session(
        sid, source="telegram", parent_session_id=parent_id,
        session_key=PEER["session_key"], chat_id=PEER["chat_id"], chat_type=PEER["chat_type"],
        thread_id=THREAD_ID, user_id=PEER["user_id"],
        model_config={"_delegate_from": parent_id},
    )
    db.append_message(sid, "assistant", "subagent working on the storage backend")
    return sid


def _route(db: SessionDB) -> str | None:
    row = db.find_latest_gateway_session_for_peer(**PEER)
    return str(row["id"]) if row else None


def test_delegate_child_holding_the_thread_never_wins_the_route(db: SessionDB):
    parent = _parent(db)
    _delegate_child(db, parent)
    assert _route(db) == parent, "the parent conversation must keep its own thread"


def test_parent_ended_session_switch_does_not_hand_the_thread_to_the_child(db: SessionDB):
    """The reporter's 04:19 state: the parent ended while children were still running.
    Recovery must mint a fresh session rather than route the user into the subagent."""
    parent = _parent(db)
    child = _delegate_child(db, parent)
    db.end_session(parent, "session_switch")
    assert _route(db) is None, "a closed boundary must never resurrect into a subagent"
    assert child and _route(db) != child


def test_route_falls_back_to_the_parent_when_the_child_ended(db: SessionDB):
    """A finished subagent must not strand the live conversation: with the child gone the
    parent is still the routing answer."""
    parent = _parent(db)
    child = _delegate_child(db, parent)
    db.end_session(child, "agent_close")
    assert _route(db) == parent


def test_fresh_conversation_after_reset_is_still_reachable(db: SessionDB):
    """The exclusion must not break ordinary continuity: after a /new boundary the chat's
    NEW conversation is still reachable (the successor's own activity postdates the
    boundary, so the reset fence still passes it)."""
    parent = _parent(db)
    db.end_session(parent, "session_reset")
    successor = "20261014_050000_" + uuid.uuid4().hex[:6]
    db.create_session(
        successor, source="telegram", session_key=PEER["session_key"], chat_id=PEER["chat_id"],
        chat_type=PEER["chat_type"], thread_id=THREAD_ID, user_id=PEER["user_id"],
    )
    db.append_message(successor, "user", "a brand new conversation")
    assert _route(db) == successor
