"""Tests for SessionDB.get_messages(include_compacted=...).

In-place compaction archives earlier turns as ``active=0, compacted=1`` rows
that are durable display history, not soft-deleted rows. A transcript read
that drops them silently cuts the user-visible conversation off at the
compaction boundary (#80680): the UI exhausts its active-only window, "Show
earlier messages" disappears, and earlier turns become unreachable even
though they are still on disk.

``include_compacted=True`` must surface those rows while still excluding
soft-deleted Undo/Rewind rows (``active=0, compacted=0``) — that remains the
job of ``include_inactive`` (audit / debug reads).
"""

import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    return SessionDB(tmp_path / "state.db")


def _seed(db, sid="s1"):
    """Session with 4 archived (compacted) turns + 2 live turns + 1 rewound row."""
    db.create_session(sid, source="cli")
    old = [
        {"role": "user", "content": "old q1"},
        {"role": "assistant", "content": "old a1"},
        {"role": "user", "content": "old q2"},
        {"role": "assistant", "content": "old a2"},
    ]
    db.append_messages_batch(sid, old)
    db.archive_and_compact(
        sid,
        [
            {"role": "assistant", "content": "summary of old turns"},
            {"role": "user", "content": "live q1"},
            {"role": "assistant", "content": "live a1"},
        ],
    )
    # Soft-delete the last live user turn (active=0, compacted=0) so every
    # row class is present: active=1/compacted=0, active=0/compacted=1,
    # active=0/compacted=0. rewind_to_message requires a user target.
    live = db.get_messages(sid)
    user_msg = next(m for m in reversed(live) if m["role"] == "user")
    db.rewind_to_message(sid, user_msg["id"])
    return db


def _row_ids(db, sid, **kwargs):
    return [m["id"] for m in db.get_messages(sid, **kwargs)]


class TestIncludeCompacted:
    def test_default_returns_only_active_rows(self, db):
        """Regression guard: the default read must not change behaviour."""
        sid = "s1"
        db = _seed(db, sid)
        msgs = db.get_messages(sid)
        assert all(m["active"] for m in msgs)
        # Only the compaction summary survived (the rewind soft-deleted
        # the live user turn AND everything after it); the 4 archived rows
        # stay hidden.
        assert len(msgs) == 1

    def test_include_compacted_surfaces_archived_rows(self, db):
        sid = "s1"
        db = _seed(db, sid)
        msgs = db.get_messages(sid, include_compacted=True)
        # 4 archived + 1 live (the summary); the 2 rewound rows are excluded.
        assert len(msgs) == 5
        assert all(m["active"] or m["compacted"] for m in msgs)
        # Archived rows are the oldest — they come first in insertion order.
        assert msgs[0]["content"] == "old q1"
        assert msgs[-1]["content"] == "summary of old turns"

    def test_include_compacted_excludes_soft_deleted_rows(self, db):
        """Undo/Rewind rows (active=0, compacted=0) stay hidden."""
        sid = "s1"
        db = _seed(db, sid)
        msgs = db.get_messages(sid, include_compacted=True)
        assert not any(not m["active"] and not m["compacted"] for m in msgs)

    def test_include_inactive_still_returns_everything(self, db):
        """Audit semantics are unchanged: include_inactive wins."""
        sid = "s1"
        db = _seed(db, sid)
        msgs = db.get_messages(sid, include_inactive=True)
        assert len(msgs) == 7  # 4 archived + 1 live + 2 rewound

    def test_latest_page_with_compacted_rows(self, db):
        """latest=True pages back from the newest message, still in order."""
        sid = "s1"
        db = _seed(db, sid)
        ids = _row_ids(db, sid, include_compacted=True, latest=True)
        all_ids = _row_ids(db, sid, include_compacted=True)
        # The whole display history fits one page; latest pages are returned
        # in chronological order (offset measured back from the newest row).
        assert ids == all_ids
        # A bounded page still lands on the newest rows.
        tail = db.get_messages(sid, include_compacted=True, latest=True, limit=3)
        assert [m["id"] for m in tail] == all_ids[-3:]

    def test_pagination_with_compacted_rows(self, db):
        """limit/offset pages over the combined display history."""
        sid = "s1"
        db = _seed(db, sid)
        page = db.get_messages(sid, include_compacted=True, limit=3, offset=2)
        all_ids = _row_ids(db, sid, include_compacted=True)
        assert [m["id"] for m in page] == all_ids[2:5]


class TestDisplayDedupe:
    """Compaction epochs copy the protected tail into each new generation, so
    the same logical message exists as several rows (identical
    role/content/timestamp). The display read must surface it exactly once.
    """

    def _copy_tail_as_new_generation(self, db, sid, ids):
        """Simulate one compaction epoch: duplicate rows as active=0,
        compacted=1 with the SAME content and timestamp (the real
        copy-protected-tail behaviour)."""

        def _do(conn):
            placeholders = ",".join("?" * len(ids))
            conn.execute(
                f"""
                INSERT INTO messages
                    (session_id, role, content, tool_call_id, tool_calls,
                     tool_name, timestamp, active, compacted)
                SELECT session_id, role, content, tool_call_id, tool_calls,
                       tool_name, timestamp, 0, 1
                FROM messages
                WHERE session_id = ? AND id IN ({placeholders})
                """,
                [sid, *ids],
            )

        db._execute_write(_do)

    def test_copied_protected_tail_is_surfaced_once(self, db):
        """A message copied across compaction epochs appears exactly once."""
        sid = "s1"
        db.create_session(sid, source="cli")
        db.append_messages_batch(
            sid,
            [
                {"role": "user", "content": "turn 1"},
                {"role": "assistant", "content": "answer 1"},
            ],
        )
        orig = _row_ids(db, sid)
        self._copy_tail_as_new_generation(db, sid, orig)
        msgs = db.get_messages(sid, include_compacted=True)
        # 2 logical messages, not 4 (the copies are duplicates).
        assert len(msgs) == 2
        assert [m["content"] for m in msgs] == ["turn 1", "answer 1"]

    def test_dedupe_prefers_live_row_then_newest_generation(self, db):
        """When generations conflict, the live row wins; otherwise the newest
        generation (highest id) wins."""
        sid = "s1"
        db.create_session(sid, source="cli")
        db.append_messages_batch(sid, [{"role": "user", "content": "dup q"}])
        gen1 = _row_ids(db, sid)
        self._copy_tail_as_new_generation(db, sid, gen1)  # compacted copy
        msgs = db.get_messages(sid, include_compacted=True)
        assert len(msgs) == 1
        assert msgs[0]["active"] == 1  # live row wins

        # Archive the live row and copy again: the newest compacted copy wins.
        db._execute_write(
            lambda conn: conn.execute(
                "UPDATE messages SET active = 0, compacted = 1 WHERE session_id = ?",
                [sid],
            )
        )
        self._copy_tail_as_new_generation(db, sid, gen1)
        newest_id = max(m["id"] for m in db.get_messages(sid, include_inactive=True))
        msgs = db.get_messages(sid, include_compacted=True)
        assert len(msgs) == 1
        assert msgs[0]["id"] == newest_id  # newest generation wins

    def test_dedupe_applies_before_paging(self, db):
        """Deduping happens over the full display set, not per page, so
        offset paging never surfaces a duplicate."""
        sid = "s1"
        db.create_session(sid, source="cli")
        db.append_messages_batch(
            sid,
            [
                {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "a1"},
                {"role": "user", "content": "q2"},
                {"role": "assistant", "content": "a2"},
            ],
        )
        orig = _row_ids(db, sid)
        self._copy_tail_as_new_generation(db, sid, orig)
        all_ids = _row_ids(db, sid, include_compacted=True)
        assert len(all_ids) == 4  # deduped, no copies
        # Paginate past where the copies would have landed.
        page = db.get_messages(sid, include_compacted=True, limit=2, offset=2)
        assert [m["id"] for m in page] == all_ids[2:]
        assert len(page) == 2

    def test_distinct_tool_calls_with_same_content_are_not_merged(self, db):
        """Two real tool messages that happen to share role/content/timestamp
        must stay separate: the dedupe key includes the tool fields, so only
        genuine compaction copies (which copy those fields verbatim) collapse.
        """
        sid = "s1"
        db.create_session(sid, source="cli")

        def _seed_tool_rows(conn):
            ts = 1700000000.0
            for cid in ("call-1", "call-2"):
                conn.execute(
                    "INSERT INTO messages (session_id, role, content, tool_call_id,"
                    " tool_name, timestamp, active, compacted)"
                    " VALUES (?, ?, ?, ?, ?, ?, 1, 0)",
                    (sid, "tool", "identical result", cid, "search", ts),
                )

        db._execute_write(_seed_tool_rows)
        msgs = db.get_messages(sid, include_compacted=True)
        assert len(msgs) == 2
        assert {m["tool_call_id"] for m in msgs} == {"call-1", "call-2"}

    def test_compaction_copies_of_tool_messages_still_collapse(self, db):
        """Tool rows copied by a compaction epoch (identical tool fields) are
        deduped like any other message, not split by the widened key."""
        sid = "s1"
        db.create_session(sid, source="cli")

        def _seed_tool_row(conn):
            conn.execute(
                "INSERT INTO messages (session_id, role, content, tool_call_id,"
                " tool_name, timestamp, active, compacted)"
                " VALUES (?, ?, ?, ?, ?, ?, 1, 0)",
                (sid, "tool", "result", "call-1", "search", 1700000000.0),
            )

        db._execute_write(_seed_tool_row)
        orig = _row_ids(db, sid)
        self._copy_tail_as_new_generation(db, sid, orig)
        msgs = db.get_messages(sid, include_compacted=True)
        assert len(msgs) == 1
        assert msgs[0]["tool_call_id"] == "call-1"
