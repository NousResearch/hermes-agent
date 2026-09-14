import sqlite3
import tempfile
from pathlib import Path
from hermes_state import SessionDB


def test_missing_parent_session_id_nullified_in_isolated_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "state.db"
        db = SessionDB(db_path=db_path)

        # Create a session with a non-existent parent_session_id
        sid = "child_session_123"
        non_existent_parent = "parent_missing_999"
        try:
            db.create_session(
                session_id=sid,
                source="cli",
                parent_session_id=non_existent_parent,
                profile_name="isolated_profile",
            )
        finally:
            db.close()

        # Inspect database row directly
        with sqlite3.connect(str(db_path)) as conn:
            row = conn.execute("SELECT id, parent_session_id FROM sessions WHERE id = ?", (sid,)).fetchone()
            assert row is not None
            assert row[0] == sid
            assert row[1] is None, "Missing parent_session_id should be nullified"
