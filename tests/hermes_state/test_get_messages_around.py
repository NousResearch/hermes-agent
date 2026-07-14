"""Tests for SessionDB.get_messages_around (anchored-window primitive).

Used by session_search both for the discovery shape (FTS5 match as anchor)
and the scroll shape (user-supplied anchor). Returns a window of messages
around the anchor plus before/after counts so callers can detect session
boundaries.
"""
import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    return SessionDB(tmp_path / "state.db")


def _seed(db, sid="s1", n=10):
    """Create session with n alternating user/assistant messages, return ids ascending."""
    db.create_session(sid, source="cli")
    ids = []
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        # append_message returns the new id
        mid = db.append_message(sid, role=role, content=f"msg {i}")
        ids.append(mid)
    return ids


class TestBasicWindow:
    def test_returns_window_around_anchor(self, db):
        ids = _seed(db, n=10)
        anchor = ids[5]
        view = db.get_messages_around("s1", anchor, window=2)
        # Expected: 2 before + anchor + 2 after = 5 messages
        msgs = view["window"]
        assert len(msgs) == 5
        assert [m["id"] for m in msgs] == [ids[3], ids[4], ids[5], ids[6], ids[7]]
        assert view["messages_before"] == 2
        assert view["messages_after"] == 2

    def test_window_zero_returns_only_anchor(self, db):
        ids = _seed(db, n=5)
        view = db.get_messages_around("s1", ids[2], window=0)
        assert len(view["window"]) == 1
        assert view["window"][0]["id"] == ids[2]
        assert view["messages_before"] == 0
        assert view["messages_after"] == 0

    def test_negative_window_clamps_to_zero(self, db):
        ids = _seed(db, n=5)
        view = db.get_messages_around("s1", ids[2], window=-3)
        # Just anchor, like window=0
        assert len(view["window"]) == 1
        assert view["window"][0]["id"] == ids[2]


class TestBoundaryDetection:
    """messages_before / messages_after tell the agent it's at start/end."""

    def test_at_session_start_messages_before_is_short(self, db):
        ids = _seed(db, n=10)
        # Anchor on first message; ask for window=5
        view = db.get_messages_around("s1", ids[0], window=5)
        assert view["messages_before"] == 0  # nothing before the first msg
        assert view["messages_after"] == 5
        # window contains anchor + 5 after = 6 messages
        assert len(view["window"]) == 6

    def test_at_session_end_messages_after_is_short(self, db):
        ids = _seed(db, n=10)
        view = db.get_messages_around("s1", ids[-1], window=5)
        assert view["messages_before"] == 5
        assert view["messages_after"] == 0
        assert len(view["window"]) == 6

    def test_window_larger_than_session(self, db):
        ids = _seed(db, n=3)
        view = db.get_messages_around("s1", ids[1], window=50)
        # All 3 messages return, both boundaries hit
        assert len(view["window"]) == 3
        assert view["messages_before"] == 1
        assert view["messages_after"] == 1


class TestAnchorValidation:
    def test_missing_anchor_returns_empty(self, db):
        _seed(db, n=5)
        view = db.get_messages_around("s1", 99999, window=5)
        assert view["window"] == []
        assert view["messages_before"] == 0
        assert view["messages_after"] == 0

    def test_anchor_in_different_session_returns_empty(self, db):
        # Two sessions, ask for s1's anchor in s2's namespace
        ids1 = _seed(db, sid="s1", n=5)
        _seed(db, sid="s2", n=5)
        view = db.get_messages_around("s2", ids1[2], window=2)
        assert view["window"] == []


class TestScrollPattern:
    """The forward/backward scroll loop the agent will run."""

    def test_scroll_forward_re_anchored_on_last_id(self, db):
        ids = _seed(db, n=20)
        anchor = ids[5]
        v1 = db.get_messages_around("s1", anchor, window=3)
        last_id = v1["window"][-1]["id"]
        v2 = db.get_messages_around("s1", last_id, window=3)
        # Boundary id (last_id) appears in both windows (in v2 it's the anchor)
        assert last_id in [m["id"] for m in v1["window"]]
        assert last_id in [m["id"] for m in v2["window"]]
        # v2's window extends beyond v1
        assert max(m["id"] for m in v2["window"]) > max(m["id"] for m in v1["window"])

    def test_scroll_backward_re_anchored_on_first_id(self, db):
        ids = _seed(db, n=20)
        anchor = ids[10]
        v1 = db.get_messages_around("s1", anchor, window=3)
        first_id = v1["window"][0]["id"]
        v2 = db.get_messages_around("s1", first_id, window=3)
        assert first_id in [m["id"] for m in v1["window"]]
        assert first_id in [m["id"] for m in v2["window"]]
        assert min(m["id"] for m in v2["window"]) < min(m["id"] for m in v1["window"])


class TestContentHydration:
    def test_content_is_decoded(self, db):
        ids = _seed(db, n=3)
        view = db.get_messages_around("s1", ids[1], window=1)
        for m in view["window"]:
            assert isinstance(m.get("content"), str)
            assert m["content"].startswith("msg ")

    def test_tool_calls_deserialized(self, db):
        db.create_session("s1", source="cli")
        # Message with tool_calls (pass list — append_message JSON-encodes it)
        tc_payload = [{"id": "t1", "function": {"name": "x", "arguments": "{}"}}]
        db.append_message("s1", role="assistant", content="", tool_calls=tc_payload)
        mid = db.append_message("s1", role="tool", content="result", tool_name="x")

        view = db.get_messages_around("s1", mid, window=2)
        # Find the assistant message with tool_calls
        asst = [m for m in view["window"] if m.get("role") == "assistant"]
        assert asst, "expected an assistant message"
        # tool_calls should be a list after hydration, not a string
        assert isinstance(asst[0].get("tool_calls"), list)


class TestSearchVisibility:
    """Window loads must match search_messages discoverability.

    - Rewind/undo: active=0, compacted=0 → hidden
    - Compaction archive: active=0, compacted=1 → still discoverable
    """

    def test_rewound_rows_excluded_from_window(self, db):
        ids = _seed(db, n=10)
        # Soft-delete from ids[6] inclusive (undo/rewind).
        db.rewind_to_message("s1", ids[6])
        view = db.get_messages_around("s1", ids[4], window=5)
        window_ids = [m["id"] for m in view["window"]]
        assert ids[4] in window_ids
        for hidden in ids[6:]:
            assert hidden not in window_ids
        # Boundary count must not count rewound rows as "after".
        assert view["messages_after"] == 1  # only ids[5] remains after anchor

    def test_inactive_rewind_anchor_returns_empty(self, db):
        ids = _seed(db, n=8)
        # rewind_to_message requires a user-role target (even indices in _seed).
        db.rewind_to_message("s1", ids[4])
        # Anchor on a rewound (now inactive) row — must fail closed.
        view = db.get_messages_around("s1", ids[5], window=3)
        assert view["window"] == []
        assert view["messages_before"] == 0
        assert view["messages_after"] == 0

    def test_compacted_rows_remain_discoverable(self, db):
        ids = _seed(db, n=8)
        # Archive live turns as compacted history; insert a summary head.
        db.archive_and_compact(
            "s1",
            [{"role": "user", "content": "summary of earlier turns"}],
        )
        # Pre-compaction rows are active=0, compacted=1 and must still load.
        view = db.get_messages_around("s1", ids[3], window=2)
        window_ids = [m["id"] for m in view["window"]]
        assert ids[3] in window_ids
        assert ids[1] in window_ids
        assert ids[5] in window_ids
        # Fresh summary row is also search-visible (active=1).
        summary_id = max(m["id"] for m in db.get_messages("s1", include_inactive=True))
        assert summary_id not in ids
        view_tail = db.get_messages_around("s1", summary_id, window=2)
        assert any(m["id"] == summary_id for m in view_tail["window"])
