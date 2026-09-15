"""Exact room identity must stay within its source, including legacy duplicate rows."""

import pytest

from hermes_state import SessionDB


@pytest.mark.parametrize("source", ["bot_room", "room' OR 1=1 --"])
def test_source_title_lookup_prefers_stable_message_bearing_session(tmp_path, source):
    db = SessionDB(tmp_path / "state.db")
    try:
        # Legacy stores may lack the title uniqueness index. Seed that state directly
        # so the lookup contract covers duplicate rows without changing schema policy.
        db._conn.execute("DROP INDEX idx_sessions_title_unique")
        for sid, surface, started, messages in (
            ("foreign", "cli", 0, 4),
            ("empty", source, 1, 0),
            ("later", source, 3, 2),
            ("stable-b", source, 2, 1),
            ("stable-a", source, 2, 1),
        ):
            db.create_session(sid, source=surface)
            db._conn.execute(
                "UPDATE sessions SET title = ?, started_at = ?, message_count = ? WHERE id = ?",
                ("Room 'quoted'", started, messages, sid),
            )
        db._conn.commit()

        for _ in range(2):
            assert db.get_session_by_title("Room 'quoted'", source=source)["id"] == "stable-a"
        assert db.get_session_by_title("Room 'quoted'", source="cli")["id"] == "foreign"
        assert db.get_session_by_title("Room 'quoted'", source="missing") is None
        assert db.get_session_by_title("' OR 1=1 --", source=source) is None
        assert db.get_session_by_title("Room 'quoted'") is not None
        assert db.get_session_by_title("absent") is None
    finally:
        db.close()
