"""
Unit tests for session_orchestration/feed.py (T007).

Coverage
--------
1. A transition into WAITING_USER pushes exactly one message to feed channel
   + task thread.
2. A repeat call with the same state (steady-state tick scenario) pushes
   nothing (debounce).
3. A transition back to RUNNING then back to WAITING_USER re-arms — next
   push fires again.
4. Missing feed_channel_id: only thread message is posted (not feed channel).
5. Missing discord_thread_id: only feed channel message is posted.
6. Both absent: no messages posted, returns False.

All tests use fakes — no live Discord calls.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import patch

import pytest

import session_orchestration.feed as feed_mod
from session_orchestration.feed import (
    clear_last_notified,
    push_turn_change,
)


# ---------------------------------------------------------------------------
# Fake Discord post helper
# ---------------------------------------------------------------------------


class _FakePostTracker:
    """Records calls to _post_discord_message and returns a fake message id."""

    def __init__(self):
        self.calls: List[Tuple[str, str]] = []  # [(channel_id, content), ...]
        self._call_count = 0

    def __call__(
        self,
        channel_id: str,
        content: str,
        *,
        token: Optional[str] = None,
    ) -> Optional[str]:
        self.calls.append((channel_id, content))
        self._call_count += 1
        return f"fake-msg-id-{self._call_count}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_row(
    task_id: str = "task-001",
    agent: str = "claude",
    thread_id: Optional[str] = "thread-999",
    project: str = "my-project",
) -> Dict[str, Any]:
    return {
        "task_id": task_id,
        "agent": agent,
        "discord_thread_id": thread_id,
        "project": project,
        "state": "RUNNING",
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_debounce():
    """Clear module-level debounce state before + after every test."""
    feed_mod._last_notified.clear()
    yield
    feed_mod._last_notified.clear()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPushTurnChange:
    """push_turn_change posts to feed + thread on a transition."""

    def test_transition_to_waiting_user_posts_to_feed_and_thread(self):
        tracker = _FakePostTracker()
        row = _make_row(task_id="t-001", thread_id="thread-111")

        with patch.object(feed_mod, "_post_discord_message", tracker):
            result = push_turn_change(
                "t-001",
                row,
                new_state="WAITING_USER",
                old_state="RUNNING",
                feed_channel_id="feed-ch-001",
            )

        assert result is True, "should report at least one message posted"
        # Expect two posts: feed channel + task thread
        channel_ids = [c for c, _ in tracker.calls]
        assert "feed-ch-001" in channel_ids, "must post to feed channel"
        assert "thread-111" in channel_ids, "must post to task thread"
        assert len(tracker.calls) == 2

    def test_transition_to_paused_handoff_posts_once(self):
        tracker = _FakePostTracker()
        row = _make_row(task_id="t-002", thread_id="thread-222")

        with patch.object(feed_mod, "_post_discord_message", tracker):
            result = push_turn_change(
                "t-002",
                row,
                new_state="PAUSED_HANDOFF",
                old_state="RUNNING",
                feed_channel_id="feed-ch-002",
            )

        assert result is True
        assert len(tracker.calls) == 2  # feed + thread

    def test_same_state_second_call_is_suppressed(self):
        """Repeat call with the same state pushes nothing (debounce)."""
        tracker = _FakePostTracker()
        row = _make_row(task_id="t-003", thread_id="thread-333")

        with patch.object(feed_mod, "_post_discord_message", tracker):
            # First call: state transition → should post
            r1 = push_turn_change(
                "t-003",
                row,
                new_state="WAITING_USER",
                old_state="RUNNING",
                feed_channel_id="feed-ch-003",
            )
            # Second call: same state → debounce suppresses
            r2 = push_turn_change(
                "t-003",
                row,
                new_state="WAITING_USER",
                old_state="RUNNING",
                feed_channel_id="feed-ch-003",
            )

        assert r1 is True, "first call should post"
        assert r2 is False, "second call (same state) must be suppressed"
        # Only 2 posts from the first call
        assert len(tracker.calls) == 2

    def test_rearm_after_transition_back(self):
        """Transition back to RUNNING then back to WAITING_USER re-arms."""
        tracker = _FakePostTracker()
        row = _make_row(task_id="t-004", thread_id="thread-444")

        with patch.object(feed_mod, "_post_discord_message", tracker):
            # First transition: RUNNING → WAITING_USER
            push_turn_change(
                "t-004",
                row,
                new_state="WAITING_USER",
                old_state="RUNNING",
                feed_channel_id="feed-ch-004",
            )
            count_after_first = len(tracker.calls)

            # Session goes back to RUNNING (clear debounce marker)
            clear_last_notified("t-004")

            # Second transition: RUNNING → WAITING_USER again
            r3 = push_turn_change(
                "t-004",
                row,
                new_state="WAITING_USER",
                old_state="RUNNING",
                feed_channel_id="feed-ch-004",
            )

        assert r3 is True, "re-armed notification should post"
        # 2 posts per transition × 2 transitions = 4 posts total
        assert len(tracker.calls) == count_after_first * 2

    def test_no_feed_channel_only_thread_posts(self):
        """When no feed_channel_id is configured, only the thread is posted."""
        tracker = _FakePostTracker()
        row = _make_row(task_id="t-005", thread_id="thread-555")

        # Patch config reader to return None
        with patch.object(feed_mod, "_get_feed_channel_id", return_value=None):
            with patch.object(feed_mod, "_post_discord_message", tracker):
                result = push_turn_change(
                    "t-005",
                    row,
                    new_state="WAITING_USER",
                    old_state="RUNNING",
                    feed_channel_id=None,  # not passed explicitly either
                )

        # Only thread post
        channel_ids = [c for c, _ in tracker.calls]
        assert "feed-ch-005" not in channel_ids
        assert "thread-555" in channel_ids
        # Returns True because thread post succeeded
        assert result is True

    def test_no_thread_id_only_feed_posts(self):
        """When row has no discord_thread_id, only the feed channel is posted."""
        tracker = _FakePostTracker()
        row = _make_row(task_id="t-006", thread_id=None)

        with patch.object(feed_mod, "_post_discord_message", tracker):
            result = push_turn_change(
                "t-006",
                row,
                new_state="WAITING_USER",
                old_state="RUNNING",
                feed_channel_id="feed-ch-006",
            )

        channel_ids = [c for c, _ in tracker.calls]
        assert "feed-ch-006" in channel_ids
        assert len(tracker.calls) == 1  # only feed
        assert result is True

    def test_no_feed_channel_no_thread_returns_false(self):
        """Both absent: no posts, returns False."""
        tracker = _FakePostTracker()
        row = _make_row(task_id="t-007", thread_id=None)

        with patch.object(feed_mod, "_get_feed_channel_id", return_value=None):
            with patch.object(feed_mod, "_post_discord_message", tracker):
                result = push_turn_change(
                    "t-007",
                    row,
                    new_state="WAITING_USER",
                    old_state="RUNNING",
                    feed_channel_id=None,
                )

        assert result is False
        assert tracker.calls == []


class TestClearLastNotified:
    """clear_last_notified re-arms the debounce for a task_id."""

    def test_clear_unknown_task_is_noop(self):
        """Clearing a task that was never notified should not raise."""
        clear_last_notified("task-never-seen")  # must not raise

    def test_clear_resets_marker(self):
        feed_mod._last_notified["task-x"] = "WAITING_USER"
        clear_last_notified("task-x")
        assert "task-x" not in feed_mod._last_notified
