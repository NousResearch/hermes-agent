"""A chat's /new continuation (``_reset_from``) never carries ``_delegate_from``: the marker hides the
row from every session list while the gateway keeps routing the chat into it (#109073)."""

from __future__ import annotations

import json
import sqlite3

from hermes_state import SessionDB

OLD, LIVE = "20260819_180933_c53aa488", "20260829_143124_a736cc"
PEER = dict(user_id="410541755", session_key="agent:main:telegram:dm:410541755", chat_id="410541755",
            chat_type="dm")


def _config(db: SessionDB, sid: str) -> dict:
    raw = db.get_session(sid)["model_config"]
    return json.loads(raw) if isinstance(raw, str) else (raw or {})


def _listed(db: SessionDB) -> list[str]:
    return [s["id"] for s in db.list_sessions_rich(source="telegram", limit=50)]


def _reset_chat(db: SessionDB) -> None:
    db.create_session(OLD, "telegram", **PEER)
    db.append_message(OLD, "user", "old")
    db.end_session(OLD, "session_reset")
    db.create_session(LIVE, "telegram", parent_session_id=OLD, model_config={"_reset_from": OLD}, **PEER)
    db.append_message(LIVE, "user", "the live chat")


def test_a_polluted_reset_row_is_listed_again_after_the_store_opens(tmp_path):
    path = tmp_path / "state.db"
    db = SessionDB(db_path=path)
    _reset_chat(db)
    db.close()
    con = sqlite3.connect(path)  # the reporter's row, written by an older build
    con.execute("UPDATE sessions SET model_config = json_set(model_config, '$._delegate_from', ?) WHERE id = ?",
                (OLD, LIVE))
    con.commit()
    con.close()

    db = SessionDB(db_path=path)
    try:
        assert _config(db, LIVE) == {"_reset_from": OLD}
        assert _listed(db)[0] == LIVE
    finally:
        db.close()


def test_a_config_arriving_with_a_delegate_marker_never_lands_on_a_reset_row(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        _reset_chat(db)
        db.create_session(LIVE, "telegram", model_config={"_delegate_from": OLD, "max_iterations": 90}, **PEER)
        assert _config(db, LIVE) == {"_reset_from": OLD, "max_iterations": 90}
        assert _listed(db)[0] == LIVE
        # A real delegate child (no reset marker) keeps its marker.
        db.create_session("child", "subagent", parent_session_id=LIVE, model_config={"_delegate_from": LIVE})
        assert _config(db, "child") == {"_delegate_from": LIVE}
    finally:
        db.close()

def test_a_new_reset_row_with_a_delegate_marker_is_visible_before_reopen(tmp_path):
    """A newly inserted polluted /new chat must not vanish until the store is reopened (#109073)."""
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session(OLD, "telegram", **PEER)
        db.append_message(OLD, "user", "old chat")
        db.end_session(OLD, "session_reset")
        db.create_session(LIVE, "telegram", parent_session_id=OLD,
                          model_config={"_reset_from": OLD, "_delegate_from": OLD}, **PEER)
        db.append_message(LIVE, "user", "new chat")
        assert _config(db, LIVE) == {"_reset_from": OLD}
        assert _listed(db)[0] == LIVE
    finally:
        db.close()

