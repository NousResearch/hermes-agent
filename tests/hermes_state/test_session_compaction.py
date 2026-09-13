"""Contracts for selective in-place session compaction."""

from hermes_state import SessionDB


def test_compact_session_archives_old_rows_and_recounts_active_history(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("s1", source="cli")
        for role, content in (
            ("user", "first request"),
            ("assistant", "first answer"),
            ("user", "second request"),
            ("assistant", "second answer"),
        ):
            db.append_message("s1", role=role, content=content)

        result = db.compact_session("s1", keep_last=2)

        assert result == {"compacted": 2, "remaining": 2}
        assert [m["content"] for m in db.get_messages_as_conversation("s1")] == [
            "second request", "second answer"
        ]
        rows = db._conn.execute(
            "SELECT active, compacted FROM messages WHERE session_id = ? ORDER BY id", ("s1",)
        ).fetchall()
        assert [(row["active"], row["compacted"]) for row in rows] == [(0, 1), (0, 1), (1, 0), (1, 0)]
        assert db.get_session("s1")["message_count"] == 2
    finally:
        db.close()


def test_compact_session_dry_run_does_not_mutate(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("s1", source="cli")
        db.append_message("s1", role="user", content="keep", timestamp=1.0)
        db.append_message("s1", role="assistant", content="drop", timestamp=2.0)

        result = db.compact_session("s1", keep_until=1.5, dry_run=True)

        assert result == {"compacted": 1, "remaining": 1}
        assert db.get_messages_as_conversation("s1")[0]["content"] == "keep"
        assert db._conn.execute(
            "SELECT COUNT(*) FROM messages WHERE active = 0"
        ).fetchone()[0] == 0
    finally:
        db.close()
