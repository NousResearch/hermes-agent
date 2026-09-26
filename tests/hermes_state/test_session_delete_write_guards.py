"""Atomic delete guards for live session writers (#123583).

The merged agent-side heal recovers after a row disappears. User-facing
destructive deletes should still refuse to remove a row while a turn or
compression owns it, and the guard must share the DELETE transaction.
"""

import os

import pytest

from hermes_state import SessionCompressionInProgressError, SessionDB, SessionTurnLeaseLostError


def _holder(label: str) -> str:
    return f"pid={os.getpid()}:{label}"


@pytest.mark.parametrize("kind", ["turn", "compression"])
def test_delete_session_refuses_live_write_guard(tmp_path, kind):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("live", source="cli")
        holder = _holder(kind)
        if kind == "turn":
            assert db.try_acquire_session_turn_lease("live", holder, ttl_seconds=60)
            error = SessionTurnLeaseLostError
        else:
            assert db.try_acquire_compression_lock("live", holder, ttl_seconds=60)
            error = SessionCompressionInProgressError

        with pytest.raises(error):
            db.delete_session("live", reject_active_write_guards=True)
        assert db.get_session("live") is not None

        if kind == "turn":
            db.release_session_turn_lease("live", holder)
        else:
            db.release_compression_lock("live", holder)
        assert db.delete_session("live", reject_active_write_guards=True) is True
    finally:
        db.close()


def test_guard_covers_bulk_and_cascaded_delegate_children_atomically(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("safe", source="cli")
        db.create_session("parent", source="cli")
        db.create_session(
            "delegate", source="subagent", parent_session_id="parent",
            model_config={"_delegate_from": "parent"},
        )
        holder = _holder("delegate-turn")
        assert db.try_acquire_session_turn_lease("delegate", holder, ttl_seconds=60)

        with pytest.raises(SessionTurnLeaseLostError):
            db.delete_session("parent", reject_active_write_guards=True)
        assert db.get_session("parent") is not None
        assert db.get_session("delegate") is not None

        with pytest.raises(SessionTurnLeaseLostError):
            db.delete_sessions(["safe", "parent"], reject_active_write_guards=True)
        assert db.get_session("safe") is not None
        assert db.get_session("parent") is not None

        db.release_session_turn_lease("delegate", holder)
        assert db.delete_sessions(["safe", "parent"], reject_active_write_guards=True) == 2
        assert db.get_session("safe") is None
        assert db.get_session("parent") is None
        assert db.get_session("delegate") is None
    finally:
        db.close()
