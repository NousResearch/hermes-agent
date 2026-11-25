"""Topic-binding reads must tolerate columns this build does not know.

A ``state.db`` written by a newer or downstream schema can carry extra columns
on ``telegram_dm_topic_bindings``. ``SELECT *`` + ``TopicBinding(**row)`` raised
``TypeError`` on them, and the swallowed read errors silently disabled every
topic-binding verb (#160070).
"""

from __future__ import annotations

import contextlib
from pathlib import Path

from persistence import ProfileScope, SessionId
from persistence.conversation.sessions import NewSession
from persistence.sqlite import SQLiteBackend

CHAT = "208214988"


@contextlib.contextmanager
def _store(db_path: Path):
    """A writable SQLite store on *db_path*, closed on exit."""
    scope = ProfileScope(profile_dir=db_path.parent, name=db_path.parent.name)
    backend = SQLiteBackend({scope.hermes_home_key: db_path})
    try:
        yield backend.open(scope)
    finally:
        backend.close()


def test_topic_binding_reads_ignore_columns_this_build_does_not_know(tmp_path: Path):
    with _store(tmp_path / "state.db") as store:
        store.sessions.create(NewSession(session_id=SessionId("s1"), source="telegram"))
        assert store.routing.bind_topic(
            chat_id=CHAT, thread_id="99", user_id=CHAT,
            session_key="tg:dm:99", session_id="s1",
        )

        # A newer/downstream build added a column; this build still reads the row.
        def _add_unknown_column(conn):
            conn.execute(
                "ALTER TABLE telegram_dm_topic_bindings ADD COLUMN last_user_inbound_at REAL"
            )
            conn.execute(
                "UPDATE telegram_dm_topic_bindings SET last_user_inbound_at = 123.0"
            )

        store.holder()._execute_write(_add_unknown_column)

        binding = store.routing.get_topic_binding(chat_id=CHAT, thread_id="99")
        assert binding is not None
        assert (binding.profile_name, binding.session_id, binding.session_key) == (
            "default", "s1", "tg:dm:99",
        )

        listed = store.routing.list_topic_bindings_for_chat(chat_id=CHAT)
        assert [row.session_id for row in listed] == ["s1"]

        by_session = store.routing.get_topic_binding_for_session(session_id="s1")
        assert by_session is not None
        assert by_session.thread_id == "99"
