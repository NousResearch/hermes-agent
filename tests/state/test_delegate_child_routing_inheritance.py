"""A delegate/branch child must never inherit its parent's gateway routing columns.

``_inherit_parent_session_metadata`` states the rule in its own docstring — routing columns are
inherited ONLY across a compression fork, because "peer recovery could repoint traffic into a
subagent's session" — but the SQL gated on the PARENT's ``end_reason`` alone.  A delegate child
whose gateway parent had already rotated on compression (long batch, queued child, detached unit)
therefore took the chat's ``session_key``/``chat_id``/``user_id``, leaving two live rows holding one
routing key (NousResearch/hermes-agent#92859).

Real ``SessionDB`` on a temp path, no mocks: the contract asserted here is the RELATIONSHIP between
the two child kinds — a compression continuation keeps inheriting, a delegate/branch fork does not.
"""

from __future__ import annotations

import pytest

from hermes_state import SessionDB

ROUTING_COLUMNS = ("session_key", "chat_id", "chat_type", "thread_id", "user_id")


@pytest.fixture()
def db(tmp_path):
    session_db = SessionDB(db_path=tmp_path / "state.db")
    try:
        yield session_db
    finally:
        session_db.close()


def _compressed_gateway_parent(db: SessionDB) -> dict:
    """A gateway conversation that rotated on compression, i.e. the one case that inherits."""
    db.create_session(
        "parent", source="telegram", session_key="agent:main:telegram:dm:42",
        chat_id="42", chat_type="dm", thread_id="7", user_id="u1",
    )
    db.end_session("parent", "compression")
    return db.get_session("parent")


@pytest.mark.parametrize("marker", ["_delegate_from", "_branched_from"])
def test_delegate_and_branch_children_do_not_take_over_the_parent_route(db: SessionDB, marker: str) -> None:
    parent = _compressed_gateway_parent(db)

    db.create_session(
        "worker", source="subagent", parent_session_id="parent", model_config={marker: "parent"},
    )
    db.create_session("continuation", source="telegram", parent_session_id="parent")

    worker, continuation = db.get_session("worker"), db.get_session("continuation")
    for column in ROUTING_COLUMNS:
        assert parent[column], f"fixture must seed {column}"
        # The continuation IS the conversation; the worker is an internal transcript.
        assert continuation[column] == parent[column], column
        assert worker[column] is None, column
