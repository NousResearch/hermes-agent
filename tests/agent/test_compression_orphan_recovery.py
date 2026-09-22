"""Recovery for legacy compression parents with no continuation child."""

from types import SimpleNamespace

from agent.conversation_compression import recover_rotated_compression_session
from hermes_state import SessionDB
from hermes_state_errors import CompressionSessionClosedError


def test_recover_rotated_compression_session_reopens_legacy_orphan(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("orphan", source="cli")
        db.append_message("orphan", "user", "before compression")
        db.end_session("orphan", "compression")
        agent = SimpleNamespace(_session_db=db, session_id="orphan")

        assert recover_rotated_compression_session(agent) is None
        db.append_message("orphan", "user", "after recovery")
    finally:
        db.close()


def test_recover_rotated_compression_session_keeps_parent_closed_with_child(
    tmp_path,
):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("parent", source="cli")
        db.append_message("parent", "user", "before compression")
        db.end_session("parent", "compression")
        db.create_session("child", source="cli", parent_session_id="parent")
        agent = SimpleNamespace(_session_db=db, session_id="parent")

        assert recover_rotated_compression_session(agent) is None
        try:
            db.append_message("parent", "user", "must stay closed")
        except CompressionSessionClosedError:
            pass
        else:
            raise AssertionError("compression parent with child was reopened")
    finally:
        db.close()
