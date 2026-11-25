"""Regression for #137197: a compaction must not store a carried message twice.

A row can be both *held* by the compressor (so the compacted set carries it
verbatim) and *above the commit watermark* (flushed after the watermark was
read, or unnamed by the held dicts so coverage fell back to the watermark).
The carried copy is inserted and the watermark path clones the row again, so
the same message lands twice with fresh ids, and the positional tail rewind
then flags a summarized row as a superseded duplicate instead.

These cases drive the backend's ``archive_and_compact`` on a ``StoreHolder``; the
other carried-tail cases run against the store verbs in
``tests/persistence_sqlite/conversation/test_compaction_carried_tail.py``.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from agent.message_metadata import RESTORED_FROM_STORE
from persistence.sqlite._holder import StoreHolder
from persistence.sqlite.conversation._store.messages import (
    append_message,
    append_messages_batch,
    archive_and_compact,
    get_active_message_watermark,
    get_messages,
    get_messages_as_conversation,
)
from persistence.sqlite.conversation._store.search import search_messages
from persistence.sqlite.conversation._store.sessions import create_session

SID = "sess1"
SUMMARY_TEXT = "[CONTEXT COMPACTION] summary of the earlier turns"


def _summary() -> dict:
    """A fresh dict per commit: the insert stamps row id, uid and timestamp onto the dict it stores."""
    return {"role": "user", "content": SUMMARY_TEXT}


@pytest.fixture
def db(tmp_path: Path):
    handle = StoreHolder(tmp_path / "state.db")
    create_session(handle, SID, source="test")
    for i in range(4):
        append_message(handle, SID, role="user" if i % 2 == 0 else "assistant", content=f"old turn {i}")
    try:
        yield handle
    finally:
        handle.close()


def _rows(db: StoreHolder) -> list[dict]:
    conn = sqlite3.connect(db.db_path)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute(
            "SELECT id, active, compacted, content, message_uid FROM messages WHERE session_id = ? ORDER BY id",
            (SID,)).fetchall()]
    finally:
        conn.close()


def _copies(db: StoreHolder, content: str) -> list[dict]:
    return [r for r in _rows(db) if r["content"] == content]


class TestCarriedTailAboveWatermark:
    def test_held_row_above_watermark_is_carried_once(self, db: StoreHolder) -> None:
        """The carried copy stands for the row; no clone lands beside it, and the
        positional tail rewind leaves the summarized rows summarized."""
        watermark = get_active_message_watermark(db, SID)
        append_message(db, SID, role="user", content="steer during the summary call")
        append_message(db, SID, role="assistant", content="reply during the summary call")
        held = get_messages_as_conversation(db, SID, include_row_ids=True)[-2:]
        assert all(m.get("_row_id") and m.get("message_uid") for m in held)

        count = archive_and_compact(db, SID, [_summary(), *held], watermark=watermark, tail_count=2)

        assert count == 3
        live = get_messages_as_conversation(db, SID)
        assert [m["content"] for m in live] == [
            SUMMARY_TEXT, "steer during the summary call", "reply during the summary call"]
        for content in ("steer during the summary call", "reply during the summary call"):
            copies = _copies(db, content)
            assert len(copies) == 2, f"{content!r}: original + carried copy, no clone"
            assert sorted(r["active"] for r in copies) == [0, 1]
            assert len({r["message_uid"] for r in copies}) == 1
            assert len(search_messages(db, content)) == 1
        # The carried dicts named rows above the watermark, so the tail rewind must not
        # reach below it: every summarized row stays recallable as compacted history.
        for i in range(4):
            (row,) = _copies(db, f"old turn {i}")
            assert (row["active"], row["compacted"]) == (0, 1)


def test_a_colliding_stored_row_is_never_dropped(db: StoreHolder) -> None:
    """A carried copy or a restored dict may only stand for the stored row it really is: a different message
    that shares its text and timestamp, its uid with newer content, or bare text keeps a live row or stays
    visible; a restored dict edited since its row was read keeps its edit."""
    append_messages_batch(db, SID, [{"role": "user", "content": "ok", "timestamp": 1000.0},
                                    {"role": "assistant", "content": "draft answer", "timestamp": 1001.0}])
    held_ok, held_draft = get_messages_as_conversation(db, SID, include_row_ids=True)[-2:]
    watermark = get_active_message_watermark(db, SID)
    append_messages_batch(db, SID, [
        {"role": "user", "content": "ok", "timestamp": 1000.0},  # another "ok" at the same second
        {"role": "assistant", "content": "edited answer", "timestamp": 1002.0, "message_uid": held_draft["message_uid"]},
        {"role": "user", "content": "continue", "timestamp": 1003.0}])  # another surface's identical text
    archive_and_compact(db, SID, [_summary(), held_ok, held_draft, {"role": "user", "content": "continue"}],
                        watermark=watermark, tail_count=3)

    live = [m["content"] for m in get_messages_as_conversation(db, SID)]
    assert live.count("ok") == 2 and "edited answer" in live
    assert [m["content"] for m in get_messages(db, SID, include_compacted=True)].count("continue") == 2

    restored = get_messages_as_conversation(db, SID)[-1]
    restored.update({"content": "continue, and also this", RESTORED_FROM_STORE: True})
    restored.pop("_db_persisted", None)
    append_messages_batch(db, SID, [restored])
    assert "continue, and also this" in [m["content"] for m in get_messages_as_conversation(db, SID)]


def _summary_merged_into_named_row(db: StoreHolder) -> tuple:
    """The compressor merged the summary into a row-addressed load of the row (``_row_id``, no digest): nothing
    tells that copy from one taken before the row was shortened, so the row keeps its clone."""
    append_message(db, SID, role="user", content="steer during the summary call")
    row = get_messages_as_conversation(db, SID, include_row_ids=True)[-1]
    merged = f"[prior]\n{row['content']}\n\n{SUMMARY_TEXT}"
    return [{**row, "content": merged, "_compressed_summary": True}], [merged, row["content"]]


def _summary_merged_into_flushed_row(db: StoreHolder) -> tuple:
    """The compressor merged the summary into its copy of a flushed dict whose digest still matches the row; the
    load lens strips the row's padding, the copy inside the merge keeps it."""
    flushed = {"role": "user", "content": "  steer during the summary call"}
    append_messages_batch(db, SID, [flushed])
    merged = f"[prior]\n{flushed['content']}\n\n{SUMMARY_TEXT}"
    return [{**flushed, "content": merged, "_compressed_summary": True}], [merged]


@pytest.mark.parametrize("build", [
    _summary_merged_into_named_row,
    _summary_merged_into_flushed_row,
], ids=lambda build: build.__name__)
def test_a_rewritten_carry_claims_only_what_the_compressor_rewrote(db: StoreHolder, build) -> None:
    """A held copy whose content differs from its tail row stands for that row only when the compressor's own
    rewrite explains the difference: the model never ends up on an older copy or on two copies of one
    message, and every stored version stays visible."""
    watermark = get_active_message_watermark(db, SID)
    compacted, expected_live = build(db)
    versions = [r["content"] for r in _rows(db) if r["active"] and r["id"] > watermark]

    archive_and_compact(db, SID, compacted, watermark=watermark, tail_count=1)

    assert [m["content"] for m in get_messages_as_conversation(db, SID)] == expected_live
    display = "\n".join(str(m["content"]) for m in get_messages(db, SID, include_compacted=True))
    assert versions and all(text in display for text in versions)
