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






def test_root_covers_delegate_child_sessions(db):
    db.create_session("parent", source="cli")
    db.create_session("child", source="delegate", parent_session_id="parent")
    assert db.get_conversation_root("child") == "parent"




