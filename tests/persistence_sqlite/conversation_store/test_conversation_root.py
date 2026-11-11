"""Tests for the SQLite backend's get_conversation_root: stable conversation id resolution.

The conversation root is the Nous Portal ``conversation=`` tag value: one
stable id per user-facing conversation, surviving context-compression
session rotation and covering delegate subagent trees.
"""
import pytest

from persistence.sqlite._holder import StoreHolder
from persistence.sqlite.conversation._store.messages import get_conversation_root
from persistence.sqlite.conversation._store.sessions import create_session


@pytest.fixture
def db(tmp_path):
    holder = StoreHolder(tmp_path / "state.db")
    yield holder
    holder.close()


def test_root_of_standalone_session_is_itself(db):
    create_session(db, "solo", source="cli")
    assert get_conversation_root(db, "solo") == "solo"


def test_root_covers_delegate_child_sessions(db):
    create_session(db, "parent", source="cli")
    create_session(db, "child", source="delegate", parent_session_id="parent")
    assert get_conversation_root(db, "child") == "parent"
