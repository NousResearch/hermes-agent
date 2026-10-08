"""Offline compaction must fence live writers and stale transcript selection."""

import sqlite3

import pytest

from hermes_state import SessionDB, SessionCompressionInProgressError
from hermes_state_errors import CompressionSessionClosedError, SessionTurnLeaseLostError


def test_guarded_compaction_is_atomic_for_leases_stale_selection_and_insert_failure(tmp_path):
    with SessionDB(db_path=tmp_path / "state.db") as db:
        db.create_session("session", "cli")
        db.append_message("session", "user", "Original")
        db.append_message("session", "assistant", "Original answer")
        expected = db.get_active_message_ids("session")
        replacement = [{"role": "user", "content": "Reviewed summary", "_compressed_summary": True}]

        for kind, exception in [
            ("turn", SessionTurnLeaseLostError),
            ("compression", SessionCompressionInProgressError),
        ]:
            if kind == "turn":
                assert db.try_acquire_session_turn_lease("session", "live-writer")
            else:
                assert db.try_acquire_compression_lock("session", "live-writer")
            with pytest.raises(exception):
                db.archive_and_compact("session", replacement, expected_active_ids=expected)
            assert db.get_active_message_ids("session") == expected
            assert db.message_count("session") == len(expected)
            if kind == "turn":
                db.release_session_turn_lease("session", "live-writer")
            else:
                db.release_compression_lock("session", "live-writer")

        db.append_message("session", "user", "Concurrent turn")
        current = db.get_active_message_ids("session")
        with pytest.raises(RuntimeError, match="active transcript changed"):
            db.archive_and_compact("session", replacement, expected_active_ids=expected)
        assert db.get_active_message_ids("session") == current
        assert db.message_count("session") == len(current)

        # A native SQLite failure after archiving must roll the entire transaction back.
        db._conn.execute(
            "CREATE TRIGGER refuse_summary BEFORE INSERT ON messages "
            "WHEN NEW._compressed_summary = 1 BEGIN SELECT RAISE(ABORT, 'summary refused'); END",
        )
        with pytest.raises(sqlite3.IntegrityError, match="summary refused"):
            db.archive_and_compact("session", replacement, expected_active_ids=current)
        assert db.get_active_message_ids("session") == current
        assert db.message_count("session") == len(current)
        db._conn.execute("DROP TRIGGER refuse_summary")
        db.end_session("session", "compression")
        with pytest.raises(CompressionSessionClosedError):
            db.archive_and_compact("session", replacement, expected_active_ids=current)
        assert db.get_active_message_ids("session") == current
