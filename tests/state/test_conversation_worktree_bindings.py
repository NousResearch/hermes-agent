"""Durable root-session worktree binding behavior."""

from __future__ import annotations

import pytest

from hermes_state import ConversationWorktreeConflict, SessionDB


@pytest.fixture
def db(tmp_path):
    session_db = SessionDB(tmp_path / "state.db")
    try:
        yield session_db
    finally:
        session_db.close()


def _claim(db: SessionDB, root_session_id: str = "root"):
    return db.claim_conversation_worktree(
        root_session_id=root_session_id,
        worktree_path=f"/repo/.worktrees/{root_session_id}",
        branch=f"hermes/session/{root_session_id}",
        base_commit="a" * 40,
        repo_common_dir="/repo/.git",
    )


def test_binding_claim_precedes_lazy_session_row(db):
    record = _claim(db, "root-1")

    assert record.root_session_id == "root-1"
    assert record.state == "creating"
    assert db.get_session("root-1") is None


def test_identical_second_claim_is_idempotent(db):
    first = _claim(db)
    second = _claim(db)

    assert second == first


def test_conflicting_second_claim_is_rejected(db):
    _claim(db)

    with pytest.raises(ConversationWorktreeConflict):
        db.claim_conversation_worktree(
            root_session_id="root",
            worktree_path="/other",
            branch="hermes/session/root",
            base_commit="a" * 40,
            repo_common_dir="/repo/.git",
        )


def test_ready_binding_survives_reopen(tmp_path):
    path = tmp_path / "state.db"
    first = SessionDB(path)
    try:
        _claim(first)
        ready = first.mark_conversation_worktree_ready("root")
        assert ready.state == "ready"
    finally:
        first.close()

    reopened = SessionDB(path)
    try:
        record = reopened.get_conversation_worktree("root")
        assert record is not None
        assert record.state == "ready"
    finally:
        reopened.close()


def test_failed_and_removed_transitions_preserve_identity(db):
    claimed = _claim(db)
    failed = db.mark_conversation_worktree_failed(
        "root", failure_phase="bootstrap", failure_message="exit status 7"
    )
    removed = db.mark_conversation_worktree_removed("root")

    assert failed.state == "creation_failed"
    assert failed.failure_phase == "bootstrap"
    assert failed.failure_message == "exit status 7"
    assert removed.state == "removed"
    assert removed.worktree_path == claimed.worktree_path
    assert removed.branch == claimed.branch
    assert removed.failure_phase is None
    assert removed.failure_message is None


def test_removed_binding_cannot_be_reactivated(db):
    _claim(db)
    db.mark_conversation_worktree_removed("root")

    with pytest.raises(ConversationWorktreeConflict, match="from 'removed' to 'ready'"):
        db.mark_conversation_worktree_ready("root")
