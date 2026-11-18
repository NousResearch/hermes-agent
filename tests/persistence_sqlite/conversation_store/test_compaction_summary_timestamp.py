"""Regression for #156992: a compaction summary must sort with its window.

``archive_and_compact`` inserts the summary at write time (``now_ts``) while the
carried tail keeps its original timestamps. Inside the new window the summary is
first by id / display_order and last by timestamp, so timestamp-ordered views
hoist it to the wrong end. These cases drive the real backend on a temp DB.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest

from agent.message_metadata import INFLIGHT_TASK_RESTATEMENT
from persistence.sqlite._holder import StoreHolder
from persistence.sqlite.conversation._store.messages import (
    append_message,
    archive_and_compact,
    get_messages_as_conversation,
)
from persistence.sqlite.conversation._store.sessions import create_session

SID = "sess1"
BASE_TS = 1_700_000_000.0
SUMMARY_TEXT = "[CONTEXT COMPACTION] summary of the earlier turns"


def _summary(content: str = SUMMARY_TEXT) -> dict:
    """Production summaries set ``_compressed_summary`` and carry no timestamp."""
    return {"role": "user", "content": content, "_compressed_summary": True}


@pytest.fixture
def db(tmp_path: Path):
    handle = StoreHolder(tmp_path / "state.db")
    create_session(handle, SID, source="test")
    try:
        yield handle
    finally:
        handle.close()


def _seed(db: StoreHolder, n: int = 4) -> list[dict]:
    for i in range(n):
        append_message(
            db, SID,
            role="user" if i % 2 == 0 else "assistant",
            content=f"old turn {i}",
            timestamp=BASE_TS + i,
        )
    return get_messages_as_conversation(db, SID, include_row_ids=True)


def _active(db: StoreHolder) -> list[dict]:
    conn = sqlite3.connect(db.db_path)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute(
            "SELECT id, timestamp, content, message_uid, _compressed_summary, active, compacted "
            "FROM messages WHERE session_id = ? AND active = 1 ORDER BY id",
            (SID,),
        ).fetchall()]
    finally:
        conn.close()


def _all_rows(db: StoreHolder) -> list[dict]:
    conn = sqlite3.connect(db.db_path)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute(
            "SELECT id, timestamp, content, message_uid, active, compacted "
            "FROM messages WHERE session_id = ? ORDER BY id",
            (SID,),
        ).fetchall()]
    finally:
        conn.close()


def _hoisted_summaries(db: StoreHolder) -> list[int]:
    """The reporter's check: a live summary whose timestamp is after a later-id live row."""
    conn = sqlite3.connect(db.db_path)
    try:
        return [row[0] for row in conn.execute(
            "SELECT s.id FROM messages s "
            "JOIN messages t ON t.session_id = s.session_id AND t.active = 1 AND t.id > s.id "
            "WHERE s.session_id = ? AND s.active = 1 AND s._compressed_summary = 1 "
            "AND s.timestamp > t.timestamp",
            (SID,),
        ).fetchall()]
    finally:
        conn.close()


def _assert_id_matches_timestamp_order(rows: list[dict]) -> None:
    by_id = [r["id"] for r in rows]
    by_ts = [r["id"] for r in sorted(rows, key=lambda r: (r["timestamp"], r["id"]))]
    assert by_id == by_ts


class TestCompactionSummaryTimestamp:
    def test_summary_sorts_with_the_carried_tail(self, db: StoreHolder) -> None:
        """Summary + tail: id order and (timestamp, id) order agree; summary sits before the tail."""
        seeded = _seed(db)
        last_summarised_ts = seeded[-3]["timestamp"]
        tail = [{k: m[k] for k in ("role", "content", "timestamp", "message_uid") if k in m}
                for m in seeded[-2:]]
        first_tail_ts = tail[0]["timestamp"]

        archive_and_compact(db, SID, [_summary(), *tail], tail_count=2)

        live = _active(db)
        assert [r["content"] for r in live] == [SUMMARY_TEXT, "old turn 2", "old turn 3"]
        _assert_id_matches_timestamp_order(live)
        summary, first_tail = live[0], live[1]
        assert summary["_compressed_summary"] == 1
        assert summary["timestamp"] < first_tail["timestamp"]
        assert summary["timestamp"] >= last_summarised_ts
        assert first_tail["timestamp"] == first_tail_ts
        assert _hoisted_summaries(db) == []

    def test_restated_inflight_turn_still_gets_compaction_time(self, db: StoreHolder) -> None:
        """#121064: a standalone restatement is new at the compaction boundary (not a summary)."""
        seeded = _seed(db)
        tail = [{k: m[k] for k in ("role", "content", "timestamp", "message_uid") if k in m}
                for m in seeded[-2:]]
        restated = {
            "role": "user",
            "content": "[Current task]\nfinish the job",
            "display_metadata": {INFLIGHT_TASK_RESTATEMENT: True},
        }
        before = time.time()
        archive_and_compact(db, SID, [_summary(), *tail, restated], tail_count=2)

        live = _active(db)
        assert [r["content"] for r in live] == [
            SUMMARY_TEXT, "old turn 2", "old turn 3", "[Current task]\nfinish the job"]
        _assert_id_matches_timestamp_order(live)
        summary, last_tail, replay = live[0], live[2], live[3]
        assert summary["_compressed_summary"] == 1
        assert replay["_compressed_summary"] == 0
        assert replay["timestamp"] >= before
        assert replay["timestamp"] > last_tail["timestamp"]
        assert summary["timestamp"] < live[1]["timestamp"]

    def test_empty_tail_uses_last_summarised_timestamp(self, db: StoreHolder) -> None:
        seeded = _seed(db)
        last_summarised_ts = seeded[-1]["timestamp"]
        archive_and_compact(db, SID, [_summary()], tail_count=0)

        live = _active(db)
        assert len(live) == 1
        assert live[0]["content"] == SUMMARY_TEXT
        assert live[0]["timestamp"] is not None
        assert live[0]["timestamp"] == last_summarised_ts
        _assert_id_matches_timestamp_order(live)

    def test_ids_and_message_uids_are_unchanged(self, db: StoreHolder) -> None:
        seeded = _seed(db)
        original_ids = [m["_row_id"] for m in seeded]
        tail_uids = [m["message_uid"] for m in seeded[-2:]]
        tail = [{k: m[k] for k in ("role", "content", "timestamp", "message_uid") if k in m}
                for m in seeded[-2:]]

        archive_and_compact(db, SID, [_summary(), *tail], tail_count=2)

        stored = {r["content"]: r for r in _all_rows(db)}
        assert [r["id"] for r in _all_rows(db) if r["content"].startswith("old turn")][:4] == original_ids
        live = _active(db)
        assert [r["message_uid"] for r in live[1:]] == tail_uids
        assert live[0]["message_uid"] not in tail_uids
        assert stored["old turn 2"]["message_uid"] == tail_uids[0]
        assert stored["old turn 3"]["message_uid"] == tail_uids[1]

    def test_named_coverage_path_stamps_the_same_way(self, db: StoreHolder) -> None:
        """``covered_ids`` takes ``_archive_named_rows``; it must use the same stamp."""
        seeded = _seed(db)
        last_summarised_ts = seeded[-3]["timestamp"]
        tail = [{k: m[k] for k in ("role", "content", "timestamp", "message_uid") if k in m}
                for m in seeded[-2:]]
        archive_and_compact(
            db, SID, [_summary(), *tail], tail_count=2,
            covered_ids=[m["_row_id"] for m in seeded], unresolved_held=[],
        )
        live = _active(db)
        _assert_id_matches_timestamp_order(live)
        assert live[0]["timestamp"] < live[1]["timestamp"]
        assert live[0]["timestamp"] >= last_summarised_ts
        assert _hoisted_summaries(db) == []

    def test_multiple_summaries_stay_before_the_tail(self, db: StoreHolder) -> None:
        seeded = _seed(db)
        tail = [{k: m[k] for k in ("role", "content", "timestamp", "message_uid") if k in m}
                for m in seeded[-2:]]
        archive_and_compact(
            db, SID,
            [_summary("S1"), _summary("S2"), *tail],
            tail_count=2,
        )
        live = _active(db)
        assert [r["content"] for r in live] == ["S1", "S2", "old turn 2", "old turn 3"]
        _assert_id_matches_timestamp_order(live)
        assert live[0]["timestamp"] < live[1]["timestamp"] < live[2]["timestamp"]
        assert _hoisted_summaries(db) == []

    def test_missing_tail_timestamp_does_not_crash(self, db: StoreHolder) -> None:
        seeded = _seed(db)
        tail = [{k: m[k] for k in ("role", "content", "message_uid") if k in m} for m in seeded[-2:]]
        archive_and_compact(db, SID, [_summary(), *tail], tail_count=2)
        live = _active(db)
        assert len(live) == 3
        assert all(r["timestamp"] is not None for r in live)
        _assert_id_matches_timestamp_order(live)

    def test_equal_tail_timestamps_keep_summary_first(self, db: StoreHolder) -> None:
        append_message(db, SID, role="user", content="a", timestamp=BASE_TS)
        append_message(db, SID, role="assistant", content="b", timestamp=BASE_TS + 1)
        append_message(db, SID, role="user", content="c", timestamp=BASE_TS + 10)
        append_message(db, SID, role="assistant", content="d", timestamp=BASE_TS + 10)
        seeded = get_messages_as_conversation(db, SID, include_row_ids=True)
        tail = [{k: m[k] for k in ("role", "content", "timestamp", "message_uid") if k in m}
                for m in seeded[-2:]]
        archive_and_compact(db, SID, [_summary(), *tail], tail_count=2)
        live = _active(db)
        _assert_id_matches_timestamp_order(live)
        assert live[0]["timestamp"] < live[1]["timestamp"]
        assert live[1]["timestamp"] == live[2]["timestamp"]
        assert _hoisted_summaries(db) == []
