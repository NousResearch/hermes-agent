"""Tests for SessionDB.get_conversation_root — stable conversation id resolution.

The conversation root is the Nous Portal ``conversation=`` tag value: one
stable id per user-facing conversation, surviving context-compression
session rotation and covering delegate subagent trees.
"""
import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    return SessionDB(tmp_path / "state.db")


def test_root_of_standalone_session_is_itself(db):
    db.create_session("solo", source="cli")
    assert db.get_conversation_root("solo") == "solo"


def test_root_of_unknown_session_is_itself(db):
    # No DB row at all (e.g. subagent's first turn before create_session).
    assert db.get_conversation_root("ghost") == "ghost"


def test_root_follows_compression_rotation_chain(db):
    # root -> seg2 -> seg3 (two compression rotations)
    db.create_session("root", source="cli")
    db.create_session("seg2", source="cli", parent_session_id="root")
    db.create_session("seg3", source="cli", parent_session_id="seg2")
    assert db.get_conversation_root("seg3") == "root"
    assert db.get_conversation_root("seg2") == "root"
    assert db.get_conversation_root("root") == "root"


def test_root_covers_delegate_child_sessions(db):
    db.create_session("parent", source="cli")
    db.create_session("child", source="delegate", parent_session_id="parent")
    assert db.get_conversation_root("child") == "parent"


def test_root_handles_parent_cycle_without_hanging(db):
    # Defensive: a corrupted parent chain with a cycle must terminate.
    db.create_session("a", source="cli")
    db.create_session("b", source="cli", parent_session_id="a")
    with db._lock:
        db._conn.execute(
            "UPDATE sessions SET parent_session_id = ? WHERE id = ?", ("b", "a")
        )
        db._conn.commit()
    root = db.get_conversation_root("b")
    assert root in ("a", "b")  # terminated, returned a chain member


def test_root_empty_session_id_passthrough(db):
    assert db.get_conversation_root("") == ""
