"""Deterministic chat-origin ownership for whole-chat title claims (SessionDB semantics).

Ordering is session opening order in the store (started_at DESC, id DESC tiebreak), never
callback completion time. Regression coverage for the Telegram group-title newest-wins slice.
"""

import time

from hermes_state import SessionDB


def _db(tmp_path):
    return SessionDB(tmp_path / "state.db")


def _open_session(db, session_id, *, chat_id="-101", thread_id=None, started_at=None):
    db.create_session(
        session_id, source="telegram", session_key=f"agent:main:telegram:group:{session_id}",
        chat_id=chat_id, chat_type="group", thread_id=thread_id,
    )
    if started_at is not None:
        db._write_sql("UPDATE sessions SET started_at = ? WHERE id = ?", (started_at, session_id))


def test_newest_live_session_owns_chat_origin(tmp_path):
    db = _db(tmp_path)
    try:
        _open_session(db, "old", started_at=time.time() - 30)
        _open_session(db, "new", started_at=time.time())
        assert db.session_owns_chat_origin(session_id="new", platform="telegram", chat_id="-101")
        assert not db.session_owns_chat_origin(session_id="old", platform="telegram", chat_id="-101")
    finally:
        db.close()


def test_ended_newer_session_leaves_ownership_with_live_older(tmp_path):
    db = _db(tmp_path)
    try:
        _open_session(db, "older", started_at=time.time() - 30)
        _open_session(db, "newer", started_at=time.time())
        db.end_session("newer", "session_reset")
        assert db.session_owns_chat_origin(session_id="older", platform="telegram", chat_id="-101")
    finally:
        db.close()


def test_missing_row_keeps_claim(tmp_path):
    db = _db(tmp_path)
    try:
        assert db.session_owns_chat_origin(session_id="ghost", platform="telegram", chat_id="-101")
    finally:
        db.close()


def test_compression_continuation_keeps_conversation_claim(tmp_path):
    db = _db(tmp_path)
    try:
        base = time.time() - 60
        _open_session(db, "gen1", started_at=base)
        db.end_session("gen1", "compression")
        db.create_session(
            "gen2", source="telegram", session_key=f"agent:main:telegram:group:gen2",
            chat_id="-101", chat_type="group", parent_session_id="gen1",
        )
        db._write_sql("UPDATE sessions SET started_at = ? WHERE id = ?", (base + 30, "gen2"))
        db.end_session("gen2", "compression")
        db.create_session(
            "gen3", source="telegram", session_key=f"agent:main:telegram:group:gen3",
            chat_id="-101", chat_type="group", parent_session_id="gen2",
        )
        db._write_sql("UPDATE sessions SET started_at = ? WHERE id = ?", (base + 60, "gen3"))
        # The delayed final title of the pre-compression session still owns its conversation.
        assert db.session_owns_chat_origin(session_id="gen1", platform="telegram", chat_id="-101")
        assert db.session_owns_chat_origin(session_id="gen2", platform="telegram", chat_id="-101")
        assert db.session_owns_chat_origin(session_id="gen3", platform="telegram", chat_id="-101")
        # An unrelated later session in the same chat takes ownership away.
        _open_session(db, "unrelated", started_at=base + 90)
        assert not db.session_owns_chat_origin(session_id="gen1", platform="telegram", chat_id="-101")
        assert db.session_owns_chat_origin(session_id="unrelated", platform="telegram", chat_id="-101")
    finally:
        db.close()


def test_started_at_tie_broken_by_row_id(tmp_path):
    db = _db(tmp_path)
    try:
        stamp = time.time()
        _open_session(db, "session-aaa", started_at=stamp)
        _open_session(db, "session-bbb", started_at=stamp)
        assert db.session_owns_chat_origin(session_id="session-bbb", platform="telegram", chat_id="-101")
        assert not db.session_owns_chat_origin(session_id="session-aaa", platform="telegram", chat_id="-101")
    finally:
        db.close()


def test_whole_chat_claim_spans_forum_topics(tmp_path):
    db = _db(tmp_path)
    try:
        _open_session(db, "topic-42", thread_id="42", started_at=time.time() - 30)
        _open_session(db, "topic-7", thread_id="7", started_at=time.time())
        # Group ownership ignores the originating topic: newest across ALL topics wins the chat.
        assert db.session_owns_chat_origin(session_id="topic-7", platform="telegram", chat_id="-101")
        assert not db.session_owns_chat_origin(session_id="topic-42", platform="telegram", chat_id="-101")
        # Same chat_id but a different chat entirely never competes.
        _open_session(db, "other-chat", chat_id="-202", started_at=time.time())
        assert db.session_owns_chat_origin(session_id="other-chat", platform="telegram", chat_id="-202")
    finally:
        db.close()
