"""Stored subagents of a conversation, for the classic CLI's subagent viewer."""
import sqlite3

import pytest


@pytest.fixture
def db(tmp_path):
    from hermes_state import SessionDB

    store = SessionDB(tmp_path / "state.db")
    yield store
    store.close()


def _child(db, child_id, parent_id, goal, *, started_at):
    db.create_session(child_id, source="subagent", parent_session_id=parent_id,
                      model_config={"_delegate_from": parent_id}, model="child-model")
    db.append_message(child_id, "user", goal)
    _set_started(db, child_id, started_at)


def _set_started(db, session_id, started_at):
    with sqlite3.connect(db.db_path) as conn:
        conn.execute("UPDATE sessions SET started_at=? WHERE id=?", (started_at, session_id))


def test_lists_direct_subagents_oldest_first_with_their_brief(db):
    db.create_session("master", "cli")
    _child(db, "late", "master", "Second task", started_at=200.0)
    _child(db, "early", "master", "First task", started_at=100.0)

    rows = db.list_subagent_sessions("master")

    assert [r["session_id"] for r in rows] == ["early", "late"]
    assert rows[0]["goal"] == "First task"
    assert rows[0]["parent_session_id"] == "master"
    assert rows[0]["depth"] == 1
    assert rows[0]["model"] == "child-model"


def test_ignores_children_that_are_not_subagents(db):
    db.create_session("master", "cli")
    db.create_session("branch", "cli", parent_session_id="master")
    _child(db, "worker", "master", "Do it", started_at=100.0)

    assert [r["session_id"] for r in db.list_subagent_sessions("master")] == ["worker"]


def test_includes_subagents_from_earlier_compression_segments(db):
    db.create_session("root", "cli")
    _set_started(db, "root", 10.0)
    _child(db, "before-rotation", "root", "Old task", started_at=20.0)
    db.end_session("root", "compression")
    db.create_session("tip", "cli", parent_session_id="root")
    _set_started(db, "tip", 30.0)
    _child(db, "after-rotation", "tip", "New task", started_at=40.0)

    rows = db.list_subagent_sessions("tip")

    assert [r["session_id"] for r in rows] == ["before-rotation", "after-rotation"]


def test_does_not_reach_into_the_conversation_a_branch_came_from(db):
    db.create_session("original", "cli")
    _child(db, "original-worker", "original", "Original task", started_at=20.0)
    db.create_session("branch", "cli", parent_session_id="original")

    assert db.list_subagent_sessions("branch") == []


def test_includes_nested_subagents_with_their_depth(db):
    db.create_session("master", "cli")
    _child(db, "orchestrator", "master", "Coordinate", started_at=100.0)
    _child(db, "leaf", "orchestrator", "Leaf work", started_at=150.0)

    rows = {r["session_id"]: r for r in db.list_subagent_sessions("master")}

    assert rows["leaf"]["depth"] == 2
    assert rows["leaf"]["parent_session_id"] == "orchestrator"


def test_reports_when_a_subagent_ended(db):
    db.create_session("master", "cli")
    _child(db, "done", "master", "Finish", started_at=100.0)
    _child(db, "busy", "master", "Still going", started_at=110.0)
    db.end_session("done", "agent_close")

    rows = {r["session_id"]: r for r in db.list_subagent_sessions("master")}

    assert rows["done"]["ended_at"] is not None
    assert rows["done"]["end_reason"] == "agent_close"
    assert rows["busy"]["ended_at"] is None


def test_unknown_or_empty_session_has_no_subagents(db):
    assert db.list_subagent_sessions("") == []
    assert db.list_subagent_sessions("missing") == []
