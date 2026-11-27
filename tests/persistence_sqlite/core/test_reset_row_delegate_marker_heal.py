"""A chat's /new continuation (``_reset_from``) that picked up a stray ``_delegate_from`` is healed when the store opens: the marker hid the row from every session list while the gateway kept routing the chat into it (#109073), and kept its messages out of the trigram index."""

from __future__ import annotations

import json
import sqlite3

import pytest

from persistence.sqlite._holder import StoreHolder
from persistence.sqlite.conversation._store.messages import append_message
from persistence.sqlite.conversation._store.sessions import (
    create_session, end_session, get_session, list_sessions_rich,
)

OLD, LIVE = "20260819_180933_c53aa488", "20260829_143124_a736cc"
PEER = dict(user_id="410541755", session_key="agent:main:telegram:dm:410541755", chat_id="410541755",
            chat_type="dm")


def _config(db: StoreHolder, sid: str) -> dict:
    raw = get_session(db, sid)["model_config"]
    return json.loads(raw) if isinstance(raw, str) else (raw or {})


def _listed(db: StoreHolder) -> list[str]:
    return [s["id"] for s in list_sessions_rich(db, source="telegram", limit=50)]


def _stamp_delegate_marker(path) -> sqlite3.Connection:
    con = sqlite3.connect(path)  # the reporter's row, written by an older build
    con.execute("UPDATE sessions SET model_config = json_set(model_config, '$._delegate_from', ?) WHERE id = ?",
                (OLD, LIVE))
    return con


def test_a_polluted_reset_row_is_listed_again_after_the_store_opens(tmp_path):
    path = tmp_path / "state.db"
    db = StoreHolder(path)
    create_session(db, OLD, "telegram", **PEER)
    append_message(db, OLD, "user", "old")
    end_session(db, OLD, "session_reset")
    create_session(db, LIVE, "telegram", parent_session_id=OLD, model_config={"_reset_from": OLD}, **PEER)
    append_message(db, LIVE, "user", "the live chat")
    db.close()
    con = _stamp_delegate_marker(path)
    con.commit()
    con.close()

    db = StoreHolder(path)
    try:
        assert _config(db, LIVE) == {"_reset_from": OLD}
        assert _listed(db)[0] == LIVE
    finally:
        db.close()


def test_a_keyed_delegate_row_without_a_reset_marker_keeps_its_marker_on_open(tmp_path):
    path = tmp_path / "state.db"
    db = StoreHolder(path)
    create_session(db, OLD, "telegram", **PEER)
    create_session(db, LIVE, "telegram", parent_session_id=OLD, model_config={"_delegate_from": OLD}, **PEER)
    db.close()

    db = StoreHolder(path)
    try:
        assert _config(db, LIVE) == {"_delegate_from": OLD}
    finally:
        db.close()


def test_a_healed_reset_row_is_indexed_for_substring_search(tmp_path):
    """The marker kept the row's messages out of the trigram index; the heal must put them in."""
    path = tmp_path / "state.db"
    db = StoreHolder(path)
    create_session(db, OLD, "telegram", **PEER)
    end_session(db, OLD, "session_reset")
    create_session(db, LIVE, "telegram", parent_session_id=OLD, model_config={"_reset_from": OLD}, **PEER)
    trigram = db._trigram_available
    db.close()
    if not trigram:
        pytest.skip("SQLite build without the trigram tokenizer")
    con = _stamp_delegate_marker(path)  # the marker landed first, so the triggers skip the message
    con.execute("INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, 'user', ?, 1.0)",
                (LIVE, "登录页面打不开怎么办"))
    con.commit()
    con.close()

    db = StoreHolder(path)
    try:
        hits = db._conn.execute(
            "SELECT rowid FROM messages_fts_trigram WHERE messages_fts_trigram MATCH ?", ('"页面打不开"',)).fetchall()
        assert hits, "the healed chat's message is missing from substring search"
        db._conn.execute("INSERT INTO messages_fts_trigram(messages_fts_trigram, rank) VALUES('integrity-check', 1)")
    finally:
        db.close()
