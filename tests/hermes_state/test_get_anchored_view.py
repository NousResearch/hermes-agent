"""Tests for SessionDB.get_anchored_view — anchored window + session bookends.

Used by the discovery shape of session_search: an FTS5 match becomes the
anchor, the call returns goal (bookend_start) + match (window) + resolution
(bookend_end) in a single round trip, no LLM.
"""
import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    return SessionDB(tmp_path / "state.db")


def _seed_long_session(db, sid="s1", n=30):
    """Create a long session with alternating user/assistant prose. Returns ids ascending."""
    db.create_session(sid, source="cli")
    ids = []
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        mid = db.append_message(sid, role=role, content=f"prose msg {i}")
        ids.append(mid)
    return ids


class TestWindowAndBookendShape:
    def test_returns_window_with_bookend_start_and_end(self, db):
        ids = _seed_long_session(db, n=30)
        # Anchor mid-session
        anchor = ids[15]
        view = db.get_anchored_view("s1", anchor, window=3, bookend=3)
        assert len(view["window"]) == 7  # ±3 + anchor
        assert len(view["bookend_start"]) == 3
        assert len(view["bookend_end"]) == 3
        # bookend_start is the first 3 ids of the session
        assert [m["id"] for m in view["bookend_start"]] == ids[:3]
        # bookend_end is the last 3 ids of the session
        assert [m["id"] for m in view["bookend_end"]] == ids[-3:]

    def test_window_anchor_marked_correctly(self, db):
        ids = _seed_long_session(db, n=20)
        anchor = ids[10]
        view = db.get_anchored_view("s1", anchor, window=2, bookend=3)
        # Anchor message is present in the window
        anchor_msgs = [m for m in view["window"] if m["id"] == anchor]
        assert len(anchor_msgs) == 1




class TestRoleFiltering:
    def test_tool_role_filtered_from_window(self, db):
        db.create_session("s1", source="cli")
        user_ids = []
        for i in range(5):
            user_ids.append(db.append_message("s1", role="user", content=f"u{i}"))
            db.append_message("s1", role="tool", content=f"tool output {i}", tool_name="x")
        # Anchor on user message
        view = db.get_anchored_view("s1", user_ids[2], window=5, bookend=0)
        # No tool messages should appear in the window
        roles = [m.get("role") for m in view["window"]]
        assert "tool" not in roles

    def test_anchor_preserved_even_when_tool_role(self, db):
        db.create_session("s1", source="cli")
        db.append_message("s1", role="user", content="ask")
        tool_id = db.append_message("s1", role="tool", content="tool output", tool_name="x")
        db.append_message("s1", role="user", content="follow-up")
        # Anchor on the tool message — should still appear despite default filter
        view = db.get_anchored_view("s1", tool_id, window=5, bookend=0)
        ids_in_window = [m["id"] for m in view["window"]]
        assert tool_id in ids_in_window



class TestEmptyContentFilter:
    """Tool-call-only assistant turns (empty content) should be skipped in bookends."""

    def test_empty_content_messages_excluded_from_bookends(self, db):
        db.create_session("s1", source="cli")
        # Real prose opener
        opener = db.append_message("s1", role="user", content="Let's start the work")
        # Empty content assistant turn (tool-call-only — common in agent loops)
        db.append_message("s1", role="assistant", content="", tool_calls=[{"id": "t1", "function": {"name": "x", "arguments": "{}"}}])
        # More prose
        for i in range(20):
            db.append_message("s1", role="user" if i % 2 == 0 else "assistant", content=f"prose {i}")
        # Another empty assistant near the end
        db.append_message("s1", role="assistant", content="", tool_calls=[{"id": "t2", "function": {"name": "y", "arguments": "{}"}}])
        # Prose closer
        closer = db.append_message("s1", role="assistant", content="Final decision: ship it.")

        # Anchor mid-session
        view = db.get_anchored_view("s1", opener + 15, window=2, bookend=3)
        # Bookend_start should not contain the empty-content tool-call turn
        for m in view["bookend_start"]:
            assert m.get("content"), "bookend_start should skip empty-content messages"
        # Bookend_end should include the closer
        end_contents = [m.get("content") for m in view["bookend_end"]]
        assert any("Final decision" in (c or "") for c in end_contents)


class TestAnchorValidation:
    def test_missing_anchor_returns_empty_view(self, db):
        _seed_long_session(db, n=10)
        view = db.get_anchored_view("s1", 999999, window=5, bookend=3)
        assert view["window"] == []
        assert view["bookend_start"] == []
        assert view["bookend_end"] == []
        assert view["messages_before"] == 0
        assert view["messages_after"] == 0


class TestSessionIsolation:
    """Bookends must not cross session boundaries."""

    def test_bookends_only_from_anchor_session(self, db):
        ids1 = _seed_long_session(db, sid="s1", n=20)
        _seed_long_session(db, sid="s2", n=20)
        view = db.get_anchored_view("s1", ids1[10], window=2, bookend=3)
        # All bookend messages should have session_id = s1 (or session_id col)
        for m in view["bookend_start"] + view["bookend_end"]:
            assert m.get("session_id") == "s1"
