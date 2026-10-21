"""Resuming a compressed conversation shows each message once.

In-place compaction can carry a protected-tail message forward wrapped in the summary carrier (the
prior turn's text, then the compaction summary). The carrier keeps the original's ``message_uid``
but not its content, so the display read used to show the archived original AND its text again
inside the carrier: every word of that reply twice after ``/compress`` and on resume (the TUI
resume/compress E2E caught it on the v0.22.3 release attempt). Real SessionDB, real
``archive_and_compact``.
"""
from agent.context_compressor import (
    SUMMARY_PREFIX,
    _MERGED_PRIOR_CONTEXT_HEADER,
    _MERGED_SUMMARY_DELIMITER,
    _SUMMARY_END_MARKER,
)
from hermes_state import SessionDB


def test_summary_carrier_does_not_show_its_message_twice(tmp_path):
    with SessionDB(db_path=tmp_path / "state.db") as db:
        db.create_session("s", source="test")
        history = [
            {"role": "user", "content": "turn one", "timestamp": 1.0},
            {"role": "assistant", "content": "REPLY-ONE words", "timestamp": 2.0},
            {"role": "user", "content": "turn two", "timestamp": 3.0},
            {"role": "assistant", "content": "REPLY-TWO words", "timestamp": 4.0},
        ]
        db.append_messages_batch("s", history)
        stored = db.get_messages_as_conversation("s", include_row_ids=True)
        carried = dict(stored[1])  # the compressor wraps the carried reply around the summary
        carried["content"] = (f"{_MERGED_PRIOR_CONTEXT_HEADER}\nREPLY-ONE words\n\n{_MERGED_SUMMARY_DELIMITER}"
                              f"\n\n{SUMMARY_PREFIX}\nsummary of earlier turns\n\n{_SUMMARY_END_MARKER}")
        carried["_compressed_summary"] = True
        db.archive_and_compact("s", [stored[0], carried, *stored[2:]])

        shown = db.get_messages_as_conversation("s", include_ancestors=True, include_compacted=True)
        listed = db.get_messages("s", include_compacted=True)  # the SQL display read exports use
        export = db.export_session("s", include_compacted=True)
        assert export is not None
        exported = export["messages"]
        for read in (shown, listed, exported):
            assert sum(str(m.get("content")).count("REPLY-ONE") for m in read) == 1, read
            # The other messages are untouched: each still shows exactly once.
            for text in ("turn one", "turn two", "REPLY-TWO"):
                assert sum(str(m.get("content")).count(text) for m in read) == 1, (text, read)
