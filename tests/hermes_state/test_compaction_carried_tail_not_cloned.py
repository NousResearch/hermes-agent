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

from agent.message_metadata import DB_ROW_SNAPSHOT, RESTORED_FROM_STORE
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


def test_a_colliding_stored_row_is_never_dropped(db: SessionDB) -> None:
    """A carried copy or a restored dict may only stand for the stored row it really is: a different message
    that shares its text and timestamp, its uid with newer content, or bare text keeps a live row or stays
    visible; a restored dict edited since its row was read keeps its edit."""
    db.append_messages_batch(SID, [{"role": "user", "content": "ok", "timestamp": 1000.0},
                                   {"role": "assistant", "content": "draft answer", "timestamp": 1001.0}])
    held_ok, held_draft = db.get_messages_as_conversation(SID, include_row_ids=True)[-2:]
    watermark = db.get_active_message_watermark(SID)
    db.append_messages_batch(SID, [
        {"role": "user", "content": "ok", "timestamp": 1000.0},  # another "ok" at the same second
        {"role": "assistant", "content": "edited answer", "timestamp": 1002.0, "message_uid": held_draft["message_uid"]},
        {"role": "user", "content": "continue", "timestamp": 1003.0}])  # another surface's identical text
    db.archive_and_compact(SID, [_summary(), held_ok, held_draft, {"role": "user", "content": "continue"}],
                           watermark=watermark, tail_count=3)

    live = [m["content"] for m in db.get_messages_as_conversation(SID)]
    assert live.count("ok") == 2 and "edited answer" in live
    assert [m["content"] for m in db.get_messages(SID, include_compacted=True)].count("continue") == 2

    restored = db.get_messages_as_conversation(SID)[-1]
    restored.update({"content": "continue, and also this", RESTORED_FROM_STORE: True})
    restored.pop("_db_persisted", None)
    db.append_messages_batch(SID, [restored])
    assert "continue, and also this" in [m["content"] for m in db.get_messages_as_conversation(SID)]


def _row_rewritten_after_hold(db: SessionDB) -> tuple:
    """The compressor took a uid-only copy, then the row was rewritten in place (an @file expansion)."""
    db.append_message(SID, role="user", content="raw @file")
    row = db.get_messages_as_conversation(SID, include_row_ids=True)[-1]
    db.set_user_message_content(SID, row["_row_id"], "expanded file body")
    held = {"role": "user", "content": "raw @file", "message_uid": row["message_uid"]}
    return [_summary(), held], [SUMMARY_TEXT, "raw @file", "expanded file body"]


def _edit_reinserted_with_its_uid(db: SessionDB) -> tuple:
    """Two tail rows share a uid (an edit re-inserted with it); the held copy is the edit."""
    db.append_message(SID, role="assistant", content="v1 draft")
    uid = db.get_messages_as_conversation(SID)[-1]["message_uid"]
    db.append_messages_batch(SID, [{"role": "assistant", "content": "v2 edited", "message_uid": uid}])
    held = {"role": "assistant", "content": "v2 edited", "message_uid": uid}
    return [_summary(), held], [SUMMARY_TEXT, "v2 edited", "v1 draft"]


def _summary_merged_into_named_row(db: SessionDB) -> tuple:
    """The compressor merged the summary into a row-addressed load of the row (``_row_id``, no digest): nothing
    tells that copy from one taken before the row was shortened, so the row keeps its clone."""
    db.append_message(SID, role="user", content="steer during the summary call")
    row = db.get_messages_as_conversation(SID, include_row_ids=True)[-1]
    merged = f"[prior]\n{row['content']}\n\n{SUMMARY_TEXT}"
    return [{**row, "content": merged, "_compressed_summary": True}], [merged, row["content"]]


def _summary_merged_into_flushed_row(db: SessionDB) -> tuple:
    """The compressor merged the summary into its copy of a flushed dict whose digest still matches the row; the
    load lens strips the row's padding, the copy inside the merge keeps it."""
    flushed = {"role": "user", "content": "  steer during the summary call"}
    db.append_messages_batch(SID, [flushed])
    merged = f"[prior]\n{flushed['content']}\n\n{SUMMARY_TEXT}"
    return [{**flushed, "content": merged, "_compressed_summary": True}], [merged]


def _row_shortened_after_flushed_hold(label: str, before: str, after: str, *drop: str):
    """The compressor copied the flushed dict (row id, uid, stored-row digest; minus *drop*), then an in-place
    edit SHORTENED the row: its new text sits inside the older copy, which must not stand for it."""
    def build(db: SessionDB) -> tuple:
        flushed = {"role": "user", "content": before}
        db.append_messages_batch(SID, [flushed])
        held = {key: value for key, value in flushed.items() if key not in drop}
        db.set_user_message_content(SID, flushed["_row_id"], after)
        return [_summary(), held], [SUMMARY_TEXT, before, after]
    build.__name__ = f"shortened_{label}"
    return build


def _row_shortened_after_row_addressed_load(db: SessionDB) -> tuple:
    """A resume-style row-addressed load (``_row_id``, no CAS digest), then an in-place edit SHORTENED the row to
    a whole line of the older copy: without a version digest the copy must not stand for the row."""
    db.append_message(SID, role="user", content="fix bug\nthen delete database")
    held = db.get_messages_as_conversation(SID, include_row_ids=True)[-1]
    db.set_user_message_content(SID, held["_row_id"], "fix bug")
    return [_summary(), held], [SUMMARY_TEXT, "fix bug\nthen delete database", "fix bug"]


@pytest.mark.parametrize("build", [
    _row_rewritten_after_hold, _row_shortened_after_row_addressed_load, _edit_reinserted_with_its_uid, _summary_merged_into_named_row,
    _summary_merged_into_flushed_row,
    _row_shortened_after_flushed_hold("suffix", "fix bug in parser", "fix bug"),
    _row_shortened_after_flushed_hold("instruction", "yes, but first delete the prod database", "yes"),
    _row_shortened_after_flushed_hold("line_digest", "fix bug\nthen drop the prod table", "fix bug"),
    _row_shortened_after_flushed_hold(
        "line_uid_only", "fix bug\nthen drop the prod table", "fix bug", "_row_id", DB_ROW_SNAPSHOT),
], ids=lambda build: build.__name__)
def test_a_rewritten_carry_claims_only_what_the_compressor_rewrote(db: SessionDB, build) -> None:
    """A held copy whose content differs from its tail row stands for that row only when the compressor's own
    rewrite explains the difference: the model never ends up on an older copy or on two copies of one
    message, and every stored version stays visible."""
    watermark = db.get_active_message_watermark(SID)
    compacted, expected_live = build(db)
    versions = [r["content"] for r in _rows(db) if r["active"] and r["id"] > watermark]

    db.archive_and_compact(SID, compacted, watermark=watermark, tail_count=1)

    assert [m["content"] for m in db.get_messages_as_conversation(SID)] == expected_live
    display = "\n".join(str(m["content"]) for m in db.get_messages(SID, include_compacted=True))
    assert versions and all(text in display for text in versions)


def _replayed(db: SessionDB) -> list:
    """What the next provider request replays per live row: the ``api_content`` sidecar, else the content."""
    return [m.get("api_content") or m["content"] for m in db.get_messages_as_conversation(SID)]


def test_matching_text_never_claims_an_independent_append(db: SessionDB) -> None:
    """A carried copy that names no row is not proof of an occurrence: another surface appending the same text
    after the watermark keeps its own model-visible row."""
    db.append_message(SID, role="assistant", content="continue")
    watermark = db.get_active_message_watermark(SID)
    db.append_message(SID, role="assistant", content="continue")

    db.archive_and_compact(SID, [_summary(), {"role": "assistant", "content": "continue"}],
                           watermark=watermark, tail_count=1)

    assert _replayed(db) == [SUMMARY_TEXT, "continue", "continue"]


def test_a_claim_keeps_the_rows_current_replay_sidecar(db: SessionDB) -> None:
    """A carried copy taken before the row's ``api_content`` sidecar was rewritten is an older model input, even
    with the same text and uid: the row's current bytes still reach the next request."""
    watermark = db.get_active_message_watermark(SID)
    flushed = {"role": "user", "content": "hello", "api_content": "first exact provider input"}
    db.append_messages_batch(SID, [flushed])
    held = dict(flushed)
    db.set_message_api_content(SID, flushed["_row_id"], "hello", "new exact provider input")

    db.archive_and_compact(SID, [_summary(), held], watermark=watermark, tail_count=1)

    assert "new exact provider input" in _replayed(db)
