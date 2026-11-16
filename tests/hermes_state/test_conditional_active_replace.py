"""Conditional transcript replacement for the standalone CAS follow-up #88799."""

from hermes_state import SessionDB


def _seed(db):
    db.create_session("s1", "test")
    for role, content in [("user", "first"), ("assistant", "reply"), ("user", "second")]:
        db.append_message("s1", role, content)


def test_conditional_replace_rejects_cross_connection_append(tmp_path):
    path = tmp_path / "state.db"
    db, writer = SessionDB(path), SessionDB(path)
    try:
        _seed(db)
        expected, _ = db.get_resume_conversations("s1")
        writer.append_message("s1", "assistant", "concurrent reply")
        before = db.get_messages("s1", include_inactive=True)

        assert db.replace_active_messages_if_unchanged(
            "s1", expected, expected[:2], archive_dropped=True) is False
        assert db.get_messages("s1", include_inactive=True) == before
    finally:
        writer.close()
        db.close()


def test_conditional_replace_preserves_survivor_ids_and_archives_only_suffix(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        _seed(db)
        expected, _ = db.get_resume_conversations("s1")
        before = db.get_messages("s1")

        assert db.replace_active_messages_if_unchanged(
            "s1", expected, expected[:2], archive_dropped=True) is True
        assert [row["id"] for row in db.get_messages("s1")] == [row["id"] for row in before[:2]]
        archived = [row for row in db.get_messages("s1", include_inactive=True) if not row["active"]]
        assert [row["id"] for row in archived] == [before[-1]["id"]]
        assert db.get_session("s1")["message_count"] == 2
    finally:
        db.close()
