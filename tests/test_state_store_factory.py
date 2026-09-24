"""Backend-neutral narrow State Store factory contracts."""

from __future__ import annotations

from state_store import open_state_store


def test_sqlite_factory_preserves_session_message_and_end_contract(tmp_path):
    store = open_state_store({}, db_path=tmp_path / "state.db")
    try:
        assert store.ensure_session("slice", source="test") == "slice"
        message_id = store.append_message("slice", role="user", content="hello")
        assert message_id > 0
        assert [(message["role"], message["content"]) for message in store.get_messages("slice")] == [("user", "hello")]
        store.end_session("slice", "done")
        assert store.get_session("slice")["end_reason"] == "done"
    finally:
        store.close()


def test_sqlite_factory_preserves_session_creation_metadata(tmp_path):
    store = open_state_store({}, db_path=tmp_path / "state.db")
    try:
        store.ensure_session(
            "metadata", source="test", metadata={"user_id": "user", "chat_id": "chat", "model": "model", "cwd": "/work"},
        )
        session = store.get_session("metadata")
        assert {key: session[key] for key in ("user_id", "chat_id", "model", "cwd")} == {
            "user_id": "user", "chat_id": "chat", "model": "model", "cwd": "/work",
        }
    finally:
        store.close()
