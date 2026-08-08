"""Regression tests for the read-only compacted transcript scope."""

from hermes_state import SessionDB


def test_compacted_scope_excludes_live_and_rewound_rows(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        session_id = "compacted-scope"
        db.create_session(session_id, "test")
        db.append_messages_batch(
            session_id,
            [
                {"role": "user", "content": "archived prompt"},
                {"role": "assistant", "content": "archived answer"},
            ],
        )
        db.archive_and_compact(
            session_id,
            [
                {"role": "assistant", "content": "compaction summary"},
                {"role": "user", "content": "live prompt"},
            ],
        )

        live_user = db.append_message(session_id, "user", "rewound prompt")
        db.rewind_to_message(session_id, live_user)

        archived = db.get_messages(session_id, compacted_only=True)
        assert [row["content"] for row in archived] == ["archived prompt", "archived answer"]
        assert all(row["active"] == 0 and row["compacted"] == 1 for row in archived)
    finally:
        db.close()
