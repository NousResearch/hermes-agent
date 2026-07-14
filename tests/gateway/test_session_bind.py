"""SessionStore direct binding for newly-created visible threads."""
from __future__ import annotations

from gateway.config import GatewayConfig, Platform
from gateway.session import SessionSource, SessionStore, build_session_key


def test_bind_session_persists_existing_target_without_throwaway_row(tmp_path, monkeypatch):
    import hermes_state

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", tmp_path / "state.db")
    sessions_dir = tmp_path / "sessions"
    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="new-thread",
        chat_name="Fork Lane",
        chat_type="thread",
        user_id="user-1",
        thread_id="new-thread",
        parent_chat_id="parent-forum",
    )

    store = SessionStore(sessions_dir=sessions_dir, config=GatewayConfig())
    db = store._db
    assert db is not None
    db.create_session(
        "child-session",
        source="discord",
        user_id="user-1",
    )
    before = db.session_count()

    entry = store.bind_session(source, "child-session", display_name="Fork Lane")

    assert entry.session_key == build_session_key(source)
    assert entry.session_id == "child-session"
    assert store.peek_session_id(entry.session_key) == "child-session"
    assert db.session_count() == before
    row = db.get_session("child-session")
    assert row is not None
    assert row["session_key"] == build_session_key(source)
    assert row["chat_id"] == "new-thread"
    assert row["chat_type"] == "thread"
    assert row["thread_id"] == "new-thread"

    # The durable routing index must survive a fresh SessionStore instance.
    restored = SessionStore(sessions_dir=sessions_dir, config=GatewayConfig())
    restored_db = restored._db
    assert restored_db is not None
    restored_entry = restored.get_or_create_session(source)
    assert restored_entry.session_id == "child-session"
    assert restored_db.session_count() == before

    restored_db.close()
    db.close()
