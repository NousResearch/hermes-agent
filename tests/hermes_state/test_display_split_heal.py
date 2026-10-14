"""A split display generation must still project as one message (issue #122167).

Symptom B: after a compaction tail move, the server-side display projection
(``get_messages(include_compacted=True)``, the ``GROUP BY display_order`` path
in ``_display_rows_from_conn``) hands the renderer the same assistant reply
twice. The reporter's measurement on the real function: both rowids inside the
returned list, ``display_order == own id`` on each side, only
active/compacted/display_order/display_identity differing.

This test rebuilds that measured end-state (compaction carry + a skewed stored
identity/order on the copy, as left behind by a generation that computed the
identity differently), drives one more compaction, and asserts the behavior
contract: one logical message, one projection entry. The heal lives on the
compaction write path (not the read path) so paged display reads stay bounded
in the size of the page, not the transcript.
"""

from hermes_state import SessionDB


def _carry_and_split(db, sid, reply_text):
    ts = 1727000000.0
    db.create_session(sid, source="desktop")
    db.append_message(sid, "user", "q", timestamp=ts)
    db.append_message(sid, "assistant", reply_text, timestamp=ts + 1)
    live = db.get_messages_as_conversation(sid)
    db.archive_and_compact(
        sid, [{"role": "user", "content": "[summary]"}] + live[-2:], tail_count=2)

    def _split(conn):
        copy_id = conn.execute(
            "SELECT MAX(id) FROM messages WHERE session_id = ? AND active = 1 AND role = 'assistant'",
            (sid,)).fetchone()[0]
        # A generation that computed the identity differently leaves the copy
        # in its own display_order group: order == own id, foreign identity.
        conn.execute(
            "UPDATE messages SET display_order = id, display_identity = zeroblob(32) "
            "WHERE id = ?", (copy_id,))
        # ...while the carried original was archived (still display-visible)
        # instead of rewound — the issue's measured end-state: both rowids
        # display-visible, one logical message in two display_order groups.
        conn.execute(
            "UPDATE messages SET compacted = 1 WHERE session_id = ? AND role = 'assistant' "
            "AND active = 0 AND id <> ?", (sid, copy_id))
        return copy_id

    return db._execute_write(_split)


def _compact_again(db, sid, reply_text):
    """One more ordinary compaction; its commit re-folds any split generation.

    ``tail_count=0`` archives (not rewinds) the skewed copy, so it stays
    display-visible: the exact shape that used to project twice."""
    live = db.get_messages_as_conversation(sid)
    tail = [m for m in live if m["role"] == "assistant" and m["content"] == reply_text][-1:]
    db.archive_and_compact(
        sid, [{"role": "user", "content": "[summary 2]"}] + tail, tail_count=0)


def _reply_count(db, sid, reply_text):
    rows = db.get_messages(sid, include_compacted=True)
    return sum(1 for m in rows if m["role"] == "assistant" and m["content"] == reply_text)


class TestSplitDisplayGenerationProjectsOnce:
    def test_text_reply_projects_once(self, tmp_path):
        from pathlib import Path
        db = SessionDB(Path(tmp_path) / "state.db")
        _carry_and_split(db, "chat", "same reply")
        assert _reply_count(db, "chat", "same reply") == 2
        _compact_again(db, "chat", "same reply")

        assert _reply_count(db, "chat", "same reply") == 1
        # The desktop pages this path (limit/offset/latest): paged reads fold too.
        for rows in (db.get_messages("chat", include_compacted=True, latest=True, limit=10),
                     db.get_messages("chat", include_compacted=True, limit=10, offset=0)):
            assert sum(1 for m in rows if m["role"] == "assistant"
                       and m["content"] == "same reply") == 1

    def test_empty_reply_projects_once(self, tmp_path):
        from pathlib import Path
        db = SessionDB(Path(tmp_path) / "state.db")
        _carry_and_split(db, "chat", "")
        assert _reply_count(db, "chat", "") == 2
        _compact_again(db, "chat", "")

        assert _reply_count(db, "chat", "") == 1

    def test_heal_is_durable_not_per_read(self, tmp_path):
        """After the healing compaction, the rows share one group on disk."""
        from pathlib import Path
        db = SessionDB(Path(tmp_path) / "state.db")
        _carry_and_split(db, "chat", "same reply")
        _compact_again(db, "chat", "same reply")
        assert _reply_count(db, "chat", "same reply") == 1

        orders = db._read_all(
            "SELECT DISTINCT display_order FROM messages WHERE session_id = ? "
            "AND role = 'assistant' AND content = ? AND (active = 1 OR compacted = 1)",
            ("chat", db._encode_content("same reply")))

        assert len(orders) == 1
