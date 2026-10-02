"""Unresolved carried copies must not leave duplicate display-visible rows (#123985).

In-place compaction carries verbatim copies of early turns forward while
archiving the durable originals. When a carried copy cannot be resolved to its
durable original (ambiguous content+timestamp fallback, or a resume surface
that omits row ids), ``archive_and_compact`` archives the original as
``active=0, compacted=1`` — display-visible — AND inserts the copy as a live
row. Both rows share one ``display_identity`` (and one ``display_order``), so
the session's first messages render twice on desktop.

Invariant: after ``archive_and_compact``, at most one display-visible row per
``display_identity`` exists in the session (the issue's own detection query
returns zero groups).
"""

from hermes_state import SessionDB


def _duplicate_groups(db, sid):
    return db._read_all(
        "SELECT display_identity, COUNT(*) AS copies, GROUP_CONCAT(id) AS ids"
        " FROM messages WHERE session_id = ? AND (active = 1 OR compacted = 1)"
        " AND display_identity IS NOT NULL GROUP BY session_id, display_identity"
        " HAVING copies > 1",
        (sid,),
    )


class TestUnresolvedCarryLeavesNoDuplicateDisplayIdentity:
    def test_ambiguous_carry_collapses_to_one_visible_row(self, tmp_path):
        db = SessionDB(tmp_path / "state.db")
        sid = "20260926_120000_dupcarry"
        db.create_session(sid, "cli", model="test/model")
        first_ts, reply_ts = 1727000000.0, 1727000001.0
        # Two byte-identical user turns (double-submit artifact): the carried
        # "hello" copy matches two durable rows, so the fallback deliberately
        # refuses to resolve it rather than risk a false rewind.
        db.append_message(sid, "user", "hello", timestamp=first_ts)
        db.append_message(sid, "user", "hello", timestamp=first_ts)
        db.append_message(sid, "assistant", "hi there", timestamp=reply_ts)
        held = db.get_messages_as_conversation(sid, include_row_ids=True)
        assert [m["_row_id"] for m in held] == [1, 2, 3]

        # A resume surface that omits row ids: copies keep content+timestamps.
        carried = [{k: v for k, v in m.items() if k != "_row_id"} for m in held]
        compacted = carried + [{"role": "assistant", "content": "[SUMMARY]"}]
        watermark = db.get_active_message_watermark(sid)
        db.archive_and_compact(
            sid, compacted, carried_messages=carried, watermark=watermark,
            covered_ids=[1, 2, 3], unresolved_held=[],
        )

        assert _duplicate_groups(db, sid) == []

        _, display = db.get_resume_conversations(sid)
        texts = [(m["role"], m["content"]) for m in display]
        assert texts.count(("user", "hello")) == 1
        assert ("assistant", "hi there") in texts
        assert ("assistant", "[SUMMARY]") in texts
