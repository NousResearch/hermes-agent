"""A split-slot compaction twin must project as one message (issue #128468).

Trigger94666-max's measured end-state on the read path: a compaction rewrite
soft-archives a turn's rows (active=0, compacted=1) and inserts fresh ones
(active=1) sharing the SAME message_uid -- but the new copy sits in a NEW
display_order slot while the archived copy keeps the OLD one. The page CTE in
_display_rows_from_conn groups by display_order, so one message serves as two
rows (measured 342 rows / 330 distinct uids). No NULL involved, so the Bug 1
NULL-safe join does not cover it; test_display_split_heal only heals on the
compaction write path, leaving legacy rows double-served.
"""
from hermes_state import SessionDB


def _split_slot(db, sid, reply_text, live_text=None):
    ts = 1727000000.0
    db.create_session(sid, source="desktop")
    db.append_message(sid, "user", "q", timestamp=ts)
    orig_id = db.append_message(sid, "assistant", reply_text, timestamp=ts + 1)
    uid = db._read_all(
        "SELECT message_uid FROM messages WHERE id = ?", (orig_id,))[0]["message_uid"]
    assert uid, "append must mint a message_uid"

    def _split(conn):
        conn.execute(
            "UPDATE messages SET active = 0, compacted = 1 WHERE id = ?", (orig_id,))
        cur = conn.execute(
            "INSERT INTO messages (session_id, role, content, timestamp, message_uid,"
            " active, compacted) VALUES (?, 'assistant', ?, ?, ?, 1, 0)",
            (sid, db._encode_content(live_text if live_text is not None else reply_text),
             ts + 1, uid))
        copy_id = cur.lastrowid
        # A generation that computed the identity differently leaves the copy
        # in its own display_order group: order == own id, foreign identity.
        conn.execute(
            "UPDATE messages SET display_order = id, display_identity = zeroblob(32)"
            " WHERE id = ?", (copy_id,))
        return copy_id

    copy_id = db._execute_write(_split)
    assert copy_id != orig_id
    return uid


def _replies(db, sid, **kw):
    rows = db.get_messages(sid, include_compacted=True, **kw)
    return [m for m in rows if m["role"] == "assistant" and "same reply" in (m["content"] or "")]


class TestSplitSlotUidDedupe:
    def test_same_uid_two_slots_projects_once(self, tmp_path):
        from pathlib import Path
        db = SessionDB(Path(tmp_path) / "state.db")
        _split_slot(db, "chat", "same reply")
        # The split-slot arm, not the NULL arm: every row keeps its slot.
        nulls = db._read_all(
            "SELECT COUNT(*) n FROM messages WHERE session_id = ?"
            " AND display_order IS NULL", ("chat",))[0]["n"]
        assert nulls == 0
        assert len(_replies(db, "chat")) == 1

    def test_survivor_is_the_live_copy(self, tmp_path):
        from pathlib import Path
        db = SessionDB(Path(tmp_path) / "state.db")
        _split_slot(db, "chat", "same reply", live_text="same reply live")
        got = _replies(db, "chat")
        assert len(got) == 1
        assert got[0]["active"] == 1
        assert got[0]["content"] == "same reply live"

    def test_rows_without_uid_still_project(self, tmp_path):
        from pathlib import Path
        db = SessionDB(Path(tmp_path) / "state.db")
        _split_slot(db, "chat", "same reply")
        db._execute_write(lambda conn: conn.execute(
            "UPDATE messages SET message_uid = NULL WHERE session_id = ?", ("chat",)))
        assert len(_replies(db, "chat")) == 2
