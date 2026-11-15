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

    def test_rewritten_user_prompt_with_same_uid_projects_once(self, tmp_path):
        """A user prompt whose skill expansion header is stripped (or in-flight restate banner
        prepended) across compaction while keeping the same message_uid must fold into one
        display_order group and project once."""
        from pathlib import Path
        db = SessionDB(Path(tmp_path) / "state.db")
        sid = "chat_user_uid_rewrite"
        db.create_session(sid, source="desktop")
        uid_go = "b40a17b67a85480893b10f3c9ac1da5f"
        uid_rca = "5f33d6f5130e4d919a5879510c36dec3"
        db.append_messages_batch(sid, [
            {"role": "user", "content": "go", "timestamp": 100.0, "message_uid": uid_go},
            {"role": "assistant", "content": "working on it", "timestamp": 101.0},
        ])
        # Compaction 1: 'go' is restated with [STILL IN PROGRESS...] keeping uid_go
        db.archive_and_compact(sid, [
            {"role": "user", "content": "[CONTEXT COMPACTION — REFERENCE ONLY]\nsummary 1", "_compressed_summary": True, "timestamp": 102.0},
            {"role": "user", "content": "[STILL IN PROGRESS — restated]\n\ngo", "timestamp": 100.0, "message_uid": uid_go},
            {"role": "assistant", "content": "done step 1", "timestamp": 103.0},
        ], tail_count=0)
        # User invokes /rca with expanded skill body, then compaction strips skill header to '/rca'
        db.append_messages_batch(sid, [
            {"role": "user", "content": "[IMPORTANT: The user has invoked the \"rca\" skill]\n\n/rca", "timestamp": 200.0, "message_uid": uid_rca},
            {"role": "assistant", "content": "investigating", "timestamp": 201.0},
        ])
        db.archive_and_compact(sid, [
            {"role": "user", "content": "[CONTEXT COMPACTION — REFERENCE ONLY]\nsummary 2", "_compressed_summary": True, "timestamp": 202.0},
            {"role": "user", "content": "/rca", "timestamp": 200.0, "message_uid": uid_rca},
            {"role": "assistant", "content": "final rca report", "timestamp": 203.0},
        ], tail_count=0)

        displayed = db.get_messages(sid, include_compacted=True)
        conv = db.get_messages_as_conversation(sid, include_compacted=True)
        for proj in (displayed, conv):
            uids = [m.get("message_uid") for m in proj if m.get("message_uid") in {uid_go, uid_rca}]
            assert uids.count(uid_go) == 1
            assert uids.count(uid_rca) == 1
        rca_msg = next(m for m in displayed if m.get("message_uid") == uid_rca)
        assert rca_msg["content"] == "/rca"
        assert db.display_message_count(sid) == len(displayed)

    def test_rewound_prefix_carried_across_compacted_turns_keeps_original_display_order(self, tmp_path):
        """When a prior turn's final assistant reply is carried as a non-contiguous prefix
        (rewound to active=0, compacted=0) across a mid-turn compaction that archives later
        user/assistant turns as compacted=1, it must retain its original pre-compaction
        display_order rather than jumping past the compacted turns that followed it."""
        from pathlib import Path
        db = SessionDB(Path(tmp_path) / "state.db")
        sid = "chat_prefix_order"
        db.create_session(sid, source="desktop")
        uid_turn3_reply = "7052ee0718a24ed7bff0c5cd629ed7b6"
        uid_rca = "5f33d6f5130e4d919a5879510c36dec3"
        db.append_messages_batch(sid, [
            {"role": "user", "content": "earlier task", "timestamp": 100.0},
            {"role": "assistant", "content": "Turn 3 final reply", "timestamp": 101.0, "message_uid": uid_turn3_reply},
            {"role": "user", "content": "[IMPORTANT: skill body]\n\n/rca", "timestamp": 200.0, "message_uid": uid_rca},
            {"role": "assistant", "content": "Turn 5 round 1 thinking", "timestamp": 201.0},
            {"role": "assistant", "content": "Turn 5 round 2 thinking", "timestamp": 202.0},
        ])
        live = db.get_messages_as_conversation(sid)
        carried_prefix = [m for m in live if m.get("message_uid") == uid_turn3_reply]
        db.archive_and_compact(
            sid,
            [
                {"role": "user", "content": "[CONTEXT COMPACTION — REFERENCE ONLY] mid-turn summary", "_compressed_summary": True, "timestamp": 203.0},
                {"role": "assistant", "content": "Turn 3 final reply", "timestamp": 101.0, "message_uid": uid_turn3_reply},
                {"role": "user", "content": "/rca", "timestamp": 200.0, "message_uid": uid_rca},
                {"role": "assistant", "content": "Turn 5 final answer", "timestamp": 204.0},
            ],
            carried_messages=carried_prefix,
        )

        displayed = db.get_messages(sid, include_compacted=True)
        conv = db.get_messages_as_conversation(sid, include_compacted=True)
        for proj in (displayed, conv):
            contents = [m["content"] for m in proj]
            assert contents.index("Turn 3 final reply") < contents.index("/rca")
            assert contents.index("/rca") < contents.index("Turn 5 round 1 thinking")
            assert contents.count("Turn 3 final reply") == 1
            assert contents.count("/rca") == 1
