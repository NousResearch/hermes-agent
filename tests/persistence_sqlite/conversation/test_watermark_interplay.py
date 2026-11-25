"""Regression tests for #86366 × watermark interplay in archive_and_compact.

Two sibling sites of the superseded-duplicate class:

* carried-forward tail originals (tail_count): the compressor's protected
  tail rides inside compacted_messages verbatim; originals must take rewind
  flags (active=0, compacted=0), not compacted=1.
* concurrent-tail originals (watermark clone, #75316): rows appended during
  the summary call are re-inserted byte-exact as live clones; their originals
  are the SAME superseded-duplicate class and must take rewind flags too.

And the interaction bound: with both watermark and tail_count set, the
rewind-target LIMIT walk must not consume concurrent-append rows (above the
watermark) as if they were carried-forward tail.
"""

from __future__ import annotations

import sqlite3

import pytest

from persistence.sqlite.conversation._store.messages import (
    append_message, archive_and_compact, get_active_message_watermark,
)
from persistence.sqlite.conversation._store.search import search_messages
from persistence.sqlite.conversation._store.sessions import create_session


@pytest.fixture
def db(backend, scope_a):
    return backend.open(scope_a).holder()


def _flags(db, session_id: str):
    conn = sqlite3.connect(db.db_path)
    conn.row_factory = sqlite3.Row
    try:
        return [
            dict(r)
            for r in conn.execute(
                "SELECT id, active, compacted, content FROM messages "
                "WHERE session_id = ? ORDER BY id",
                (session_id,),
            ).fetchall()
        ]
    finally:
        conn.close()


class TestWatermarkTailCountInterplay:
    def test_rewind_walk_bounded_at_watermark(self, db) -> None:
        """A concurrent append above the watermark must not steal a rewind
        LIMIT slot from a genuine carried-forward tail original."""
        sid = "s-wm-bound"
        create_session(db, sid, source="cli")
        for i in range(4):
            append_message(db, sid, "user", content=f"seen-{i}")
        watermark = get_active_message_watermark(db, sid)
        # Concurrent append AFTER the compressor snapshotted its input.
        append_message(db, sid, "user", content="concurrent-append")

        archive_and_compact(
            db,
            sid,
            [
                {"role": "user", "content": "SUMMARY"},
                {"role": "user", "content": "seen-2"},
                {"role": "user", "content": "seen-3"},
            ],
            watermark=watermark,
            tail_count=2,
        )

        rows = _flags(db, sid)
        by_content = {}
        for r in rows:
            by_content.setdefault(r["content"], []).append(r)

        # Carried-forward tail originals: rewind flags (hidden from recall).
        for content in ("seen-2", "seen-3"):
            originals = [
                r for r in by_content[content] if r["active"] == 0
            ]
            assert originals, content
            assert all(r["compacted"] == 0 for r in originals), (
                f"{content} original stamped compacted=1: recall duplicate"
            )
            live = [r for r in by_content[content] if r["active"] == 1]
            assert len(live) == 1

        # Summarized-away rows keep discoverability.
        for content in ("seen-0", "seen-1"):
            (row,) = by_content[content]
            assert (row["active"], row["compacted"]) == (0, 1)

    def test_concurrent_tail_original_takes_rewind_flags(
        self, db
    ) -> None:
        """The watermark clone's original is a superseded byte-identical
        duplicate: it must not satisfy recall next to its live clone."""
        sid = "s-wm-clone"
        create_session(db, sid, source="cli")
        for i in range(3):
            append_message(db, sid, "user", content=f"old-{i}")
        watermark = get_active_message_watermark(db, sid)
        append_message(db, sid, "user", content="mid-flight zqx-token")

        archive_and_compact(
            db,
            sid,
            [{"role": "user", "content": "SUMMARY"}],
            watermark=watermark,
        )

        rows = _flags(db, sid)
        copies = [r for r in rows if "zqx-token" in r["content"]]
        assert len(copies) == 2  # archived original + live clone
        original = next(r for r in copies if r["active"] == 0)
        clone = next(r for r in copies if r["active"] == 1)
        assert original["compacted"] == 0, (
            "concurrent-tail original stamped compacted=1: it would be "
            "recalled alongside its live clone (same class as #86366)"
        )
        assert clone["compacted"] == 0

        # Recall filter surfaces exactly ONE copy.
        hits = [
            r
            for r in search_messages(db, "zqx-token")
            if r.get("session_id") == sid
        ]
        assert len(hits) == 1

    def test_recall_stable_across_generations(self, db) -> None:
        """search_messages hit count for a carried-forward message must not
        grow with compaction generations."""
        sid = "s-gen"
        create_session(db, sid, source="cli")
        for i in range(5):
            append_message(db, sid, "user", content=f"turn-{i} gentok-{i}")

        payload = [
            {"role": "user", "content": "SUMMARY"},
            {"role": "user", "content": "turn-3 gentok-3"},
            {"role": "user", "content": "turn-4 gentok-4"},
        ]
        counts = []
        for _ in range(3):
            archive_and_compact(db, sid, list(payload), tail_count=2)
            counts.append(
                len(
                    [
                        r
                        for r in search_messages(db, "gentok-3")
                        if r.get("session_id") == sid
                    ]
                )
            )
        assert counts == [1, 1, 1], counts
