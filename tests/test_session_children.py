"""Tests for SessionDB.list_child_sessions — the Observatory historical
timeline's child-lane data source.

The critical invariant: only ``source='subagent'`` children are returned.
``parent_session_id`` is also set by ``/branch`` forks and compression/handoff
continuations, so a naive ``WHERE parent_session_id = ?`` would leak unrelated
conversations into the delegation timeline.
"""

import pytest

from hermes_state import SessionDB


@pytest.fixture()
def db(tmp_path):
    session_db = SessionDB(db_path=tmp_path / "state.db")
    yield session_db
    session_db.close()


def _set_timing(db, session_id, started_at, ended_at=None, tool_call_count=0):
    """Stamp explicit timing on a row the way real subagent completion does."""

    def _do(conn):
        conn.execute(
            "UPDATE sessions SET started_at = ?, ended_at = ?, tool_call_count = ? WHERE id = ?",
            (started_at, ended_at, tool_call_count, session_id),
        )

    db._execute_write(_do)


class TestListChildSessions:
    def test_returns_only_subagent_children(self, db):
        db.create_session(session_id="parent", source="cli", model="opus")
        db.create_session(
            session_id="child-sub-1", source="subagent", model="gpt-5.5", parent_session_id="parent"
        )
        db.create_session(
            session_id="child-sub-2", source="subagent", model="gpt-5.5", parent_session_id="parent"
        )
        # A /branch fork ALSO sets parent_session_id — must be excluded.
        db.create_session(
            session_id="branch-fork", source="telegram", model="opus", parent_session_id="parent"
        )
        # A compression/handoff continuation (cli) — must be excluded.
        db.create_session(
            session_id="cli-cont", source="cli", model="opus", parent_session_id="parent"
        )

        children = db.list_child_sessions("parent")
        ids = [c["id"] for c in children]

        assert ids == ["child-sub-1", "child-sub-2"]
        assert all(c["status"] in {"completed", "running"} for c in children)

    def test_ordered_by_started_at_ascending(self, db):
        db.create_session(session_id="parent", source="cli", model="opus")
        db.create_session(session_id="late", source="subagent", parent_session_id="parent")
        db.create_session(session_id="early", source="subagent", parent_session_id="parent")

        _set_timing(db, "late", started_at=2_000.0)
        _set_timing(db, "early", started_at=1_000.0)

        children = db.list_child_sessions("parent")

        assert [c["id"] for c in children] == ["early", "late"]

    def test_status_from_ended_at_not_end_reason(self, db):
        db.create_session(session_id="parent", source="cli", model="opus")
        db.create_session(session_id="done", source="subagent", parent_session_id="parent")
        db.create_session(session_id="live", source="subagent", parent_session_id="parent")

        # 'done' has ended_at but NO end_reason (the common real shape) -> completed.
        _set_timing(db, "done", started_at=1_000.0, ended_at=1_060.0, tool_call_count=7)
        # 'live' has no ended_at -> running.
        _set_timing(db, "live", started_at=1_010.0, ended_at=None, tool_call_count=3)

        children = {c["id"]: c for c in db.list_child_sessions("parent")}

        assert children["done"]["status"] == "completed"
        assert children["done"]["ended_at"] == 1_060.0
        assert children["done"]["tool_call_count"] == 7
        assert children["live"]["status"] == "running"
        assert children["live"]["ended_at"] is None
        assert children["live"]["tool_call_count"] == 3

    def test_no_children_returns_empty(self, db):
        db.create_session(session_id="lonely", source="cli", model="opus")

        assert db.list_child_sessions("lonely") == []

    def test_unknown_parent_returns_empty(self, db):
        assert db.list_child_sessions("does-not-exist") == []

    def test_carries_timing_and_model_fields(self, db):
        db.create_session(session_id="parent", source="cli", model="opus")
        db.create_session(
            session_id="c1", source="subagent", model="gpt-5.5", parent_session_id="parent"
        )
        _set_timing(db, "c1", started_at=1_700_000_000.0, ended_at=1_700_000_042.0, tool_call_count=22)

        [child] = db.list_child_sessions("parent")

        assert child["started_at"] == 1_700_000_000.0
        assert child["ended_at"] == 1_700_000_042.0
        assert child["tool_call_count"] == 22
        assert child["model"] == "gpt-5.5"
