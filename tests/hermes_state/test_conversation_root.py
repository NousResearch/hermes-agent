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


def test_branch_and_its_compression_tip_include_the_parent_in_general_lineage(db):
    """General-usage lineage (Nous Portal tagging, title generation, bot-mode features)
    deliberately traverses an explicit /branch back to its parent — unlike workspace
    ownership, which is a separate, narrower concept
    (hermes_cli.cli_conversation_worktree_mixin._conversation_worktree_root) that must
    stop at the branch so a resume never lands in the parent's worktree."""
    db.create_session("parent", source="cli")
    db.create_session(
        "branch",
        source="cli",
        parent_session_id="parent",
        model_config={"_branched_from": "parent"},
    )
    db.create_session("branch-tip", source="cli", parent_session_id="branch")

    assert db.get_conversation_root("branch") == "parent"
    assert db.get_conversation_root("branch-tip") == "parent"


