"""Regression for #137197: a compaction must not store a carried message twice.

A row can be both *held* by the compressor (so the compacted set carries it
verbatim) and *above the commit watermark* (flushed after the watermark was
read, or unnamed by the held dicts so coverage fell back to the watermark).
The carried copy is inserted and the watermark path clones the row again, so
the same message lands twice with fresh ids, and the positional tail rewind
then flags a summarized row as a superseded duplicate instead.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from hermes_state import SessionDB

SID = "sess1"
SUMMARY_TEXT = "[CONTEXT COMPACTION] summary of the earlier turns"


def _summary() -> dict:
    """A fresh dict per commit: the insert stamps row id, uid and timestamp onto the dict it stores."""
    return {"role": "user", "content": SUMMARY_TEXT}


@pytest.fixture
def db(tmp_path: Path):
    handle = SessionDB(tmp_path / "state.db")
    handle.create_session(SID, source="test")
    for i in range(4):
        handle.append_message(SID, role="user" if i % 2 == 0 else "assistant", content=f"old turn {i}")
    try:
        yield handle
    finally:
        handle.close()


def _rows(db: SessionDB) -> list[dict]:
    conn = sqlite3.connect(db.db_path)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute(
            "SELECT id, active, compacted, content, message_uid FROM messages WHERE session_id = ? ORDER BY id",
            (SID,)).fetchall()]
    finally:
        conn.close()


def _copies(db: SessionDB, content: str) -> list[dict]:
    return [r for r in _rows(db) if r["content"] == content]


class TestCarriedTailAboveWatermark:
    def test_held_row_above_watermark_is_carried_once(self, db: SessionDB) -> None:
        """The carried copy stands for the row; no clone lands beside it, and the
        positional tail rewind leaves the summarized rows summarized."""
        watermark = db.get_active_message_watermark(SID)
        db.append_message(SID, role="user", content="steer during the summary call")
        db.append_message(SID, role="assistant", content="reply during the summary call")
        held = db.get_messages_as_conversation(SID, include_row_ids=True)[-2:]
        assert all(m.get("_row_id") and m.get("message_uid") for m in held)

        count = db.archive_and_compact(SID, [_summary(), *held], watermark=watermark, tail_count=2)

        assert count == 3
        live = db.get_messages_as_conversation(SID)
        assert [m["content"] for m in live] == [
            SUMMARY_TEXT, "steer during the summary call", "reply during the summary call"]
        for content in ("steer during the summary call", "reply during the summary call"):
            copies = _copies(db, content)
            assert len(copies) == 2, f"{content!r}: original + carried copy, no clone"
            assert sorted(r["active"] for r in copies) == [0, 1]
            assert len({r["message_uid"] for r in copies}) == 1
            assert len(db.search_messages(content)) == 1
        # The carried dicts named rows above the watermark, so the tail rewind must not
        # reach below it: every summarized row stays recallable as compacted history.
        for i in range(4):
            (row,) = _copies(db, f"old turn {i}")
            assert (row["active"], row["compacted"]) == (0, 1)

    def test_content_only_carried_copy_is_not_cloned(self, db: SessionDB) -> None:
        """A carried dict with no id, uid or timestamp still names the row by content."""
        watermark = db.get_active_message_watermark(SID)
        db.append_message(SID, role="assistant", content="verbatim carried body")

        db.archive_and_compact(
            SID, [_summary(), {"role": "assistant", "content": "verbatim carried body"}], watermark=watermark)

        copies = _copies(db, "verbatim carried body")
        assert len(copies) == 2
        assert sorted(r["active"] for r in copies) == [0, 1]

    def test_foreign_identical_row_above_watermark_is_still_cloned(self, db: SessionDB) -> None:
        """A row another surface appended keeps its clone even when a held row below the
        watermark carries the same text: the carried dict names its own row, not the newcomer."""
        db.append_message(SID, role="assistant", content="same text twice")
        held = db.get_messages_as_conversation(SID, include_row_ids=True)[-1]
        watermark = db.get_active_message_watermark(SID)
        db.append_message(SID, role="assistant", content="same text twice")

        db.archive_and_compact(SID, [_summary(), held], watermark=watermark, tail_count=1)

        copies = _copies(db, "same text twice")
        assert len(copies) == 4, "held original + its carried copy, foreign original + its clone"
        assert sorted(r["active"] for r in copies) == [0, 0, 1, 1]
        assert len({r["message_uid"] for r in copies if r["active"]}) == 2
