from hermes_state import SessionDB


def test_list_sessions_rich_can_scope_by_source_and_user_id(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("alice_tg", "telegram", user_id="alice")
        db.set_session_title("alice_tg", "Alice Telegram")
        db.create_session("alice_cli", "cli", user_id="alice")
        db.set_session_title("alice_cli", "Alice CLI")
        db.create_session("bob_tg", "telegram", user_id="bob")
        db.set_session_title("bob_tg", "Bob Telegram")

        sessions = db.list_sessions_rich(source="telegram", user_id="alice", limit=20)
        ids = [session["id"] for session in sessions]

        assert ids == ["alice_tg"]
    finally:
        db.close()


def test_resolve_session_id_and_title_respect_scope(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("alice_secret_001", "telegram", user_id="alice")
        db.set_session_title("alice_secret_001", "Alice Secret")

        db.create_session("bob_secret_001", "telegram", user_id="bob")
        db.set_session_title("bob_secret_001", "Bob Secret")

        assert db.resolve_session_id("bob_secret_001", source="telegram", user_id="alice") is None
        assert db.resolve_session_id("bob_secret", source="telegram", user_id="alice") is None
        assert db.resolve_session_by_title("Bob Secret", source="telegram", user_id="alice") is None

        assert db.resolve_session_id("bob_secret", source="telegram", user_id="bob") == "bob_secret_001"
        assert db.resolve_session_by_title("Bob Secret", source="telegram", user_id="bob") == "bob_secret_001"
    finally:
        db.close()
