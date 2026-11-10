"""Resuming a compressed conversation shows each message once.

In-place compaction can carry a protected-tail message forward wrapped in the summary carrier (the
prior turn's text, then the compaction summary). The carrier keeps the original's ``message_uid``
but not its content, so the display read used to show the archived original AND its text again
inside the carrier: every word of that reply twice after ``/compress`` and on resume (the TUI
resume/compress E2E caught it on the v0.22.3 release attempt). Real SQLite store, real
``archive_and_compact``.
"""
from agent.context_compressor import (
    SUMMARY_PREFIX,
    _MERGED_PRIOR_CONTEXT_HEADER,
    _MERGED_SUMMARY_DELIMITER,
    _SUMMARY_END_MARKER,
)
from persistence.sqlite.conversation._store.messages import (
    append_messages_batch, archive_and_compact, get_messages, get_messages_as_conversation,
)
from persistence.sqlite.conversation._store.sessions import create_session


def test_summary_carrier_does_not_show_its_message_twice(backend, scope_a):
    store = backend.open(scope_a)
    holder = store.holder()
    create_session(holder, "s", source="test")
    history = [
        {"role": "user", "content": "turn one", "timestamp": 1.0},
        {"role": "assistant", "content": "REPLY-ONE words", "timestamp": 2.0},
        {"role": "user", "content": "turn two", "timestamp": 3.0},
        {"role": "assistant", "content": "REPLY-TWO words", "timestamp": 4.0},
    ]
    append_messages_batch(holder, "s", history)
    stored = get_messages_as_conversation(holder, "s", include_row_ids=True)
    carried = dict(stored[1])  # the compressor wraps the carried reply around the summary
    carried["content"] = (f"{_MERGED_PRIOR_CONTEXT_HEADER}\nREPLY-ONE words\n\n{_MERGED_SUMMARY_DELIMITER}"
                          f"\n\n{SUMMARY_PREFIX}\nsummary of earlier turns\n\n{_SUMMARY_END_MARKER}")
    carried["_compressed_summary"] = True
    archive_and_compact(holder, "s", [stored[0], carried, *stored[2:]])

    shown = get_messages_as_conversation(holder, "s", include_ancestors=True, include_compacted=True)
    listed = get_messages(holder, "s", include_compacted=True)  # the SQL display read exports use
    export = store.maintenance.export_session("s", include_compacted=True)
    assert export is not None
    exported = [{"content": row.content} for row in export.messages]
    for read in (shown, listed, exported):
        assert sum(str(m.get("content")).count("REPLY-ONE") for m in read) == 1, read
        # The other messages are untouched: each still shows exactly once.
        for text in ("turn one", "turn two", "REPLY-TWO"):
            assert sum(str(m.get("content")).count(text) for m in read) == 1, (text, read)
