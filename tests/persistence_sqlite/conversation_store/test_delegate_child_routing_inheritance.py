"""A delegate/branch child must never inherit its parent's gateway routing columns.

``_inherit_parent_session_metadata`` states the rule in its own docstring — routing columns are
inherited ONLY across a compression fork, because "peer recovery could repoint traffic into a
subagent's session" — but the SQL gated on the PARENT's ``end_reason`` alone.  A delegate child
whose gateway parent had already rotated on compression (long batch, queued child, detached unit)
therefore took the chat's ``session_key``/``chat_id``/``user_id``, leaving two live rows holding one
routing key (NousResearch/hermes-agent#116322, first reported as #92859).

The SQLite backend's ``create_session`` on a ``StoreHolder`` over a temp path, no mocks: the contract
asserted here is the RELATIONSHIP between the two child kinds — a compression continuation keeps
inheriting, a delegate/branch fork does not.
"""

from __future__ import annotations

import pytest

from persistence.sqlite._holder import StoreHolder
from persistence.sqlite.conversation._store.sessions import create_session, end_session, get_session

ROUTING_COLUMNS = ("session_key", "chat_id", "chat_type", "thread_id", "user_id")


@pytest.fixture()
def db(tmp_path):
    holder = StoreHolder(tmp_path / "state.db")
    try:
        yield holder
    finally:
        holder.close()


def _compressed_gateway_parent(db: StoreHolder) -> dict:
    """A gateway conversation that rotated on compression, i.e. the one case that inherits."""
    create_session(
        db, "parent", source="telegram", session_key="agent:main:telegram:dm:42",
        chat_id="42", chat_type="dm", thread_id="7", user_id="u1",
    )
    end_session(db, "parent", "compression")
    return get_session(db, "parent")


@pytest.mark.parametrize("marker", ["_delegate_from", "_branched_from"])
def test_delegate_and_branch_children_do_not_take_over_the_parent_route(db: StoreHolder, marker: str) -> None:
    parent = _compressed_gateway_parent(db)

    create_session(
        db, "worker", source="subagent", parent_session_id="parent", model_config={marker: "parent"},
    )
    create_session(db, "continuation", source="telegram", parent_session_id="parent")

    worker, continuation = get_session(db, "worker"), get_session(db, "continuation")
    for column in ROUTING_COLUMNS:
        assert parent[column], f"fixture must seed {column}"
        # The continuation IS the conversation; the worker is an internal transcript.
        assert continuation[column] == parent[column], column
        assert worker[column] is None, column


def test_compression_continuation_of_a_branch_child_keeps_inheriting(db: StoreHolder) -> None:
    """A continuation copies model_config verbatim — the branch marker comes along — so the
    exclusion must match the marker's VALUE against the parent id, not its mere presence."""
    create_session(db, "root", source="telegram")
    create_session(
        db, "branch", source="telegram", parent_session_id="root", model_config={"_branched_from": "root"},
        session_key="agent:main:telegram:dm:42", chat_id="42", chat_type="dm", user_id="u1",
    )
    end_session(db, "branch", "compression")

    create_session(
        db, "continuation", source="telegram", parent_session_id="branch", model_config={"_branched_from": "root"},
    )

    continuation = get_session(db, "continuation")
    assert continuation["session_key"] == "agent:main:telegram:dm:42"
    assert continuation["chat_id"] == "42" and continuation["user_id"] == "u1"
