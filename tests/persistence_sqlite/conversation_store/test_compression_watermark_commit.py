"""Watermark commit: concurrent appends survive in-place compaction (#75316).

The provider summary call is external and slow. Messages that arrive while it
runs must (a) persist immediately (appends are not fenced by the compression
lock) and (b) survive the commit: ``archive_and_compact(watermark=...)``
re-sequences every active row above the watermark after the compacted set
instead of archiving it. The commit is holder-fenced: a compression whose
lease was reclaimed cannot publish a stale compaction.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time

import pytest

from persistence.sqlite._holder import StoreHolder
from persistence.sqlite._state.errors import SessionCompressionInProgressError
from persistence.sqlite.conversation._store.compression import publish_compression_child
from persistence.sqlite.conversation._store.messages import (
    append_message,
    archive_and_compact,
    get_active_message_watermark,
    get_ancestor_display_prefix,
    get_messages,
    get_messages_as_conversation,
    get_resume_conversations,
)
from persistence.sqlite.conversation._store.sessions import create_session, get_session


@pytest.fixture
def store(backend, scope_a):
    return backend.open(scope_a)


@pytest.fixture
def db(store) -> StoreHolder:
    d = store.holder()
    create_session(d, "sess1", source="test")
    return d


def _seed(db: StoreHolder, n: int = 6) -> None:
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        append_message(db, "sess1", role=role, content=f"turn {i}")


SUMMARY = [
    {"role": "user", "content": "[CONTEXT COMPACTION] summary of turns 0-5"},
    {"role": "assistant", "content": "Continuing from the summary."},
]


class TestWatermarkCommit:
    def test_concurrent_tail_survives_compaction(self, db: StoreHolder) -> None:
        _seed(db)
        watermark = get_active_message_watermark(db, "sess1")
        # Simulate the slow summary window: two messages land after capture.
        append_message(db, "sess1", role="user", content="mid-compression steer")
        append_message(db, "sess1", role="assistant", content="mid-compression reply")

        count = archive_and_compact(db, "sess1", SUMMARY, watermark=watermark)

        live = get_messages(db, "sess1")
        contents = [r["content"] for r in live]
        assert contents == [
            "[CONTEXT COMPACTION] summary of turns 0-5",
            "Continuing from the summary.",
            "mid-compression steer",
            "mid-compression reply",
        ], "tail must follow the summary, in arrival order"
        assert count == 4

    def test_tail_clone_preserves_every_column(self, db: StoreHolder) -> None:
        """The pure-SQL clone must carry sidecar fields byte-exact."""
        _seed(db, 2)
        watermark = get_active_message_watermark(db, "sess1")
        append_message(
            db,
            "sess1",
            role="assistant",
            content="tool caller",
            tool_calls=[{"id": "c1", "type": "function",
                         "function": {"name": "terminal", "arguments": "{}"}}],
        )
        append_message(
            db,
            "sess1", role="tool", content="tool output",
            tool_call_id="c1", tool_name="terminal",
        )

        archive_and_compact(db, "sess1", SUMMARY, watermark=watermark)

        live = get_messages(db, "sess1")
        by_content = {r["content"]: r for r in live}
        caller = by_content["tool caller"]
        result = by_content["tool output"]
        parsed = caller["tool_calls"]
        if isinstance(parsed, str):
            parsed = json.loads(parsed)
        assert parsed and parsed[0]["id"] == "c1"
        assert result["tool_call_id"] == "c1"
        assert result["tool_name"] == "terminal"

    def test_conversation_load_is_correct_after_commit(self, db: StoreHolder) -> None:
        """The live conversation projection sees summary + tail, in order."""
        _seed(db)
        watermark = get_active_message_watermark(db, "sess1")
        append_message(db, "sess1", role="user", content="late arrival")

        archive_and_compact(db, "sess1", SUMMARY, watermark=watermark)

        convo = get_messages_as_conversation(db, "sess1")
        assert [m["content"] for m in convo] == [
            "[CONTEXT COMPACTION] summary of turns 0-5",
            "Continuing from the summary.",
            "late arrival",
        ]

    def test_no_tail_behaves_identically_to_legacy(self, db: StoreHolder) -> None:
        _seed(db)
        watermark = get_active_message_watermark(db, "sess1")
        count = archive_and_compact(db, "sess1", SUMMARY, watermark=watermark)
        assert count == 2
        assert [r["content"] for r in get_messages(db, "sess1")] == [
            SUMMARY[0]["content"], SUMMARY[1]["content"],
        ]

    def test_none_watermark_preserves_historical_behavior(self, db: StoreHolder) -> None:
        """watermark=None archives everything: the pre-#75316 contract."""
        _seed(db)
        append_message(db, "sess1", role="user", content="gets archived")
        count = archive_and_compact(db, "sess1", SUMMARY, watermark=None)
        assert count == 2
        contents = [r["content"] for r in get_messages(db, "sess1")]
        assert "gets archived" not in contents

    def test_archived_rows_stay_recoverable(self, db: StoreHolder) -> None:
        """Originals (snapshot AND tail source rows) survive as archived."""
        _seed(db, 4)
        watermark = get_active_message_watermark(db, "sess1")
        append_message(db, "sess1", role="user", content="tail row")
        archive_and_compact(db, "sess1", SUMMARY, watermark=watermark)

        everything = get_messages(db, "sess1", include_inactive=True)
        archived = [r for r in everything if not r["active"]]
        assert sum(1 for r in archived if r["content"] == "turn 0") == 1
        # The tail original is archived; its clone is the live copy.
        tail_rows = [r for r in everything if r["content"] == "tail row"]
        assert sorted(bool(r["active"]) for r in tail_rows) == [False, True]

    def test_session_counters_include_tail(self, db: StoreHolder) -> None:
        _seed(db)
        watermark = get_active_message_watermark(db, "sess1")
        append_message(
            db,
            "sess1", role="assistant", content="tail with tools",
            tool_calls=[{"id": "t1", "type": "function",
                         "function": {"name": "x", "arguments": "{}"}}],
        )
        archive_and_compact(db, "sess1", SUMMARY, watermark=watermark)
        info = get_session(db, "sess1")
        assert info["message_count"] == 3
        assert info["tool_call_count"] == 1


class TestCommitFence:
    def test_commit_refused_when_lease_lost(self, store, db: StoreHolder) -> None:
        _seed(db)
        watermark = get_active_message_watermark(db, "sess1")
        assert store.coordination.try_acquire_compression_lock(session_id="sess1", holder="worker-A") is True
        # Lease reclaimed by another writer while worker-A's summary ran.
        store.coordination.release_compression_lock(session_id="sess1", holder="worker-A")
        assert store.coordination.try_acquire_compression_lock(session_id="sess1", holder="worker-B") is True

        with pytest.raises(SessionCompressionInProgressError):
            archive_and_compact(
                db,
                "sess1", SUMMARY, watermark=watermark, lock_holder="worker-A"
            )
        # Nothing committed: original transcript intact.
        assert [r["content"] for r in get_messages(db, "sess1")] == [
            f"turn {i}" for i in range(6)
        ]

    def test_commit_refused_when_lease_expired(self, store, db: StoreHolder) -> None:
        _seed(db)
        assert store.coordination.try_acquire_compression_lock(session_id="sess1", holder="worker-A", ttl_seconds=0.05) is True
        time.sleep(0.1)  # health: allow HX013 -- the lease expiring on the wall clock is the refusal under test
        with pytest.raises(SessionCompressionInProgressError):
            archive_and_compact(db, "sess1", SUMMARY, lock_holder="worker-A")

    def test_commit_allowed_for_live_holder(self, store, db: StoreHolder) -> None:
        _seed(db)
        watermark = get_active_message_watermark(db, "sess1")
        assert store.coordination.try_acquire_compression_lock(session_id="sess1", holder="worker-A") is True
        count = archive_and_compact(
            db,
            "sess1", SUMMARY, watermark=watermark, lock_holder="worker-A"
        )
        assert count == 2

    def test_refused_commit_rolls_back_atomically(self, db: StoreHolder) -> None:
        """Failure injection: the fence raise must leave zero partial writes."""
        _seed(db)
        before = get_messages(db, "sess1", include_inactive=True)
        with pytest.raises(SessionCompressionInProgressError):
            archive_and_compact(db, "sess1", SUMMARY, lock_holder="never-held")
        after = get_messages(db, "sess1", include_inactive=True)
        assert len(before) == len(after)
        assert all(r["active"] for r in after)


class TestConcurrentAppendDuringCompaction:
    def test_append_racing_the_commit_transaction(self, db: StoreHolder) -> None:
        """An append serialized behind the commit lands AFTER it, never lost.

        SQLite's write lock serializes the two transactions; whichever side
        wins, the append must end up in the live transcript.
        """
        _seed(db)
        watermark = get_active_message_watermark(db, "sess1")

        barrier = threading.Barrier(2, timeout=10)
        append_err: list = []

        def _racer():
            barrier.wait()
            try:
                append_message(db, "sess1", role="user", content="racer")
            except Exception as exc:  # pragma: no cover
                append_err.append(exc)

        t = threading.Thread(target=_racer, daemon=True)
        t.start()
        barrier.wait()
        archive_and_compact(db, "sess1", SUMMARY, watermark=watermark)
        t.join(timeout=10)

        assert not append_err, f"append died during commit race: {append_err}"
        contents = [r["content"] for r in get_messages(db, "sess1")]
        assert "racer" in contents, "racing append was lost"


class TestRotationPathWatermark:
    """Legacy (non-in-place) compression rotates to a child session:
    the concurrent tail must follow the rotation instead of stranding in
    the closed parent."""

    @pytest.mark.parametrize(
        "tail_content",
        [
            "mid-rotation steer",
            [
                {"type": "text", "text": "inspect this"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,AA=="},
                },
            ],
        ],
        ids=["text", "multimodal"],
    )
    def test_tail_clones_into_the_child(self, store, db: StoreHolder, tail_content) -> None:
        _seed(db)
        watermark = get_active_message_watermark(db, "sess1")
        assert store.coordination.try_acquire_compression_lock(session_id="sess1", holder="rotator") is True
        append_message(db, "sess1", role="user", content=tail_content)
        # Ceiling captured AFTER the foreign append, BEFORE the rotation
        # path's own pre-publish flush (which this test has none of).
        ceiling = get_active_message_watermark(db, "sess1")

        publish_compression_child(
            db,
            parent_session_id="sess1",
            child_session_id="child1",
            source="test",
            messages=SUMMARY,
            compression_lock_holder="rotator",
            require_compression_lease=True,
            watermark=watermark,
            watermark_ceiling=ceiling,
        )

        child = get_messages_as_conversation(db, "child1")
        assert [m["content"] for m in child] == [
            SUMMARY[0]["content"],
            SUMMARY[1]["content"],
            tail_content,
        ]
        model_history, display_history = get_resume_conversations(db, "child1")
        visible_steers = [
            message
            for message in display_history
            if message.get("content") == tail_content
        ]
        assert len(visible_steers) == 1
        assert visible_steers[0]["_row_id"] == model_history[-1]["_row_id"]
        assert all(
            message.get("content") != tail_content
            for message in get_ancestor_display_prefix(db, "child1")
        )
        info = get_session(db, "child1")
        assert info["message_count"] == 3
        # Parent keeps its copy for lineage recovery; parent is closed.
        parent_info = get_session(db, "sess1")
        assert parent_info["end_reason"] == "compression"

    def test_ceiling_excludes_the_rotators_own_flush(self, store, db: StoreHolder) -> None:
        """Rows the rotation path flushes AFTER the ceiling (its own input
        transcript, already inside the handoff) must NOT be cloned."""
        _seed(db)
        watermark = get_active_message_watermark(db, "sess1")
        assert store.coordination.try_acquire_compression_lock(session_id="sess1", holder="rotator") is True
        append_message(db, "sess1", role="user", content="foreign steer")
        ceiling = get_active_message_watermark(db, "sess1")
        # Simulates the #47202 pre-publish flush of the rotator's own input.
        append_message(
            db,
            "sess1", role="user", content="rotator's own flush",
            compression_lock_holder="rotator",
        )

        publish_compression_child(
            db,
            parent_session_id="sess1",
            child_session_id="child1",
            source="test",
            messages=SUMMARY,
            compression_lock_holder="rotator",
            require_compression_lease=True,
            watermark=watermark,
            watermark_ceiling=ceiling,
        )

        child_contents = [
            m["content"] for m in get_messages_as_conversation(db, "child1")
        ]
        assert "foreign steer" in child_contents
        assert "rotator's own flush" not in child_contents

    def test_no_watermark_keeps_historical_rotation(self, store, db: StoreHolder) -> None:
        _seed(db)
        assert store.coordination.try_acquire_compression_lock(session_id="sess1", holder="rotator") is True
        append_message(db, "sess1", role="user", content="stranded either way")
        publish_compression_child(
            db,
            parent_session_id="sess1",
            child_session_id="child1",
            source="test",
            messages=SUMMARY,
            compression_lock_holder="rotator",
            require_compression_lease=True,
        )
        child = get_messages_as_conversation(db, "child1")
        assert [m["content"] for m in child] == [
            SUMMARY[0]["content"], SUMMARY[1]["content"],
        ]
