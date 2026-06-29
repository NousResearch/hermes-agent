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


# ---------------------------------------------------------------------------
# T012 — summarize_fn / last_question tests
# ---------------------------------------------------------------------------


class TestPushTurnChangeSummarizeFn:
    """Tests for the summarize_fn / last_question feature (T012)."""

    def _row_with_question(self, question: str) -> Dict[str, Any]:
        row = _make_row(task_id="t-q01", thread_id="thread-q01")
        row["last_question"] = question
        return row

    def test_push_turn_change_waiting_user_includes_question(self):
        """WAITING_USER + last_question: message contains the question text."""
        tracker = _FakePostTracker()
        row = self._row_with_question("What should the output format be?")

        with patch.object(feed_mod, "_post_discord_message", tracker):
            result = push_turn_change(
                "t-q01",
                row,
                new_state="WAITING_USER",
                old_state="RUNNING",
                feed_channel_id="feed-ch-q01",
            )

        assert result is True
        # Both feed and thread messages should contain the question
        for _, content in tracker.calls:
            assert "What should the output format be?" in content, (
                f"question not found in message: {content!r}"
            )
        # Must be a readable blockquote, not a raw JSON dump
        for _, content in tracker.calls:
            assert content.count("{") == 0, "message must not contain raw JSON braces"

    def test_push_turn_change_summarize_fn_called(self):
        """When summarize_fn is provided it is called with last_question and its result is embedded."""
        tracker = _FakePostTracker()
        question_received: list = []

        def fake_summarize(q: str) -> str:
            question_received.append(q)
            return "SUMMARY"

        row = self._row_with_question("Should I use tabs or spaces?")

        with patch.object(feed_mod, "_post_discord_message", tracker):
            push_turn_change(
                "t-q01",
                row,
                new_state="WAITING_USER",
                old_state="RUNNING",
                feed_channel_id="feed-ch-q01",
                summarize_fn=fake_summarize,
            )

        # fn must have been called exactly once with the question string
        assert question_received == ["Should I use tabs or spaces?"], (
            f"summarize_fn not called correctly; got calls: {question_received}"
        )
        # Both posted messages must contain the fn's return value
        for _, content in tracker.calls:
            assert "SUMMARY" in content, f"summary not in message: {content!r}"

    def test_push_turn_change_no_question_no_summarize_fn_call(self):
        """When last_question is absent/empty, summarize_fn is NOT called."""
        tracker = _FakePostTracker()
        fn_calls: list = []

        def recording_fn(q: str) -> str:
            fn_calls.append(q)
            return "SHOULD_NOT_APPEAR"

        # Row with no last_question key
        row = _make_row(task_id="t-q02", thread_id="thread-q02")

        with patch.object(feed_mod, "_post_discord_message", tracker):
            result = push_turn_change(
                "t-q02",
                row,
                new_state="WAITING_USER",
                old_state="RUNNING",
                feed_channel_id="feed-ch-q02",
                summarize_fn=recording_fn,
            )

        assert fn_calls == [], f"summarize_fn must not be called when no question; got: {fn_calls}"
        # Post still succeeds (no question note appended, but message is valid)
        assert result is True
        for _, content in tracker.calls:
            assert "SHOULD_NOT_APPEAR" not in content

    def test_push_turn_change_non_waiting_state_no_summarize_fn_call(self):
        """When new_state != WAITING_USER, summarize_fn is NOT called even if last_question present."""
        tracker = _FakePostTracker()
        fn_calls: list = []

        def recording_fn(q: str) -> str:
            fn_calls.append(q)
            return "SHOULD_NOT_APPEAR"

        row = self._row_with_question("Is this a good stopping point?")

        with patch.object(feed_mod, "_post_discord_message", tracker):
            push_turn_change(
                "t-q01",
                row,
                new_state="PAUSED_HANDOFF",
                old_state="RUNNING",
                feed_channel_id="feed-ch-q01",
                summarize_fn=recording_fn,
            )

        assert fn_calls == [], (
            f"summarize_fn must not be called for non-WAITING_USER state; got: {fn_calls}"
        )
        for _, content in tracker.calls:
            assert "SHOULD_NOT_APPEAR" not in content


# ---------------------------------------------------------------------------
# T013 — DONE/RUNNING icons, WAITING_USER DM, hang stale DM
# ---------------------------------------------------------------------------


class TestPushTurnChangeIcons:
    """T013: DONE and RUNNING states render distinct icons in the posted message."""

    def test_done_icon_in_message(self):
        """new_state=DONE must produce a message containing '✅'."""
        tracker = _FakePostTracker()
        row = _make_row(task_id="t-done-icon", thread_id=None)

        with patch.object(feed_mod, "_post_discord_message", tracker):
            result = push_turn_change(
                "t-done-icon",
                row,
                new_state="DONE",
                old_state="RUNNING",
                feed_channel_id="feed-ch-done",
            )

        assert result is True
        assert tracker.calls, "at least one post expected"
        for _, content in tracker.calls:
            assert "✅" in content, f"DONE icon '✅' not in message: {content!r}"

    def test_running_icon_in_message(self):
        """new_state=RUNNING must produce a message containing '▶'."""
        tracker = _FakePostTracker()
        row = _make_row(task_id="t-running-icon", thread_id=None)

        with patch.object(feed_mod, "_post_discord_message", tracker):
            result = push_turn_change(
                "t-running-icon",
                row,
                new_state="RUNNING",
                old_state="WAITING_USER",
                feed_channel_id="feed-ch-running",
            )

        assert result is True
        assert tracker.calls, "at least one post expected"
        for _, content in tracker.calls:
            assert "▶" in content, f"RUNNING icon '▶' not in message: {content!r}"


class TestPushTurnChangeWaitingUserDM:
    """T013: push_turn_change sends a DM when new_state=WAITING_USER and user_id present."""

    def test_waiting_user_sends_dm(self, monkeypatch):
        """WAITING_USER + discord_user_id -> send_dm called with the message."""
        tracker = _FakePostTracker()
        dm_calls: list = []

        monkeypatch.setattr(
            "tools.discord_tool._get_bot_token",
            lambda: "fake-bot-token",
        )
        monkeypatch.setattr(
            "session_orchestration.dm_transport.send_dm",
            lambda user_id, message, token, **kw: dm_calls.append((user_id, message, token)),
        )

        row = _make_row(task_id="t-dm-wu-001", thread_id=None)
        row["discord_user_id"] = "user-dm-001"

        with patch.object(feed_mod, "_post_discord_message", tracker):
            push_turn_change(
                "t-dm-wu-001",
                row,
                new_state="WAITING_USER",
                old_state="RUNNING",
                feed_channel_id="feed-ch-dm",
            )

        assert len(dm_calls) == 1, "send_dm must be called exactly once"
        user_id_sent, message, token = dm_calls[0]
        assert user_id_sent == "user-dm-001"
        assert token == "fake-bot-token"

    def test_waiting_user_no_dm_when_no_user_id(self, monkeypatch):
        """WAITING_USER without discord_user_id: send_dm must NOT be called."""
        tracker = _FakePostTracker()
        dm_calls: list = []

        monkeypatch.setattr(
            "session_orchestration.dm_transport.send_dm",
            lambda *a, **kw: dm_calls.append(a),
        )

        row = _make_row(task_id="t-dm-wu-noid", thread_id=None)
        # No discord_user_id key in row

        with patch.object(feed_mod, "_post_discord_message", tracker):
            push_turn_change(
                "t-dm-wu-noid",
                row,
                new_state="WAITING_USER",
                old_state="RUNNING",
                feed_channel_id="feed-ch-dm-noid",
            )

        assert dm_calls == [], "send_dm must NOT be called when discord_user_id absent"


class TestPushHangNotificationDM:
    """T013: push_hang_notification sends a DM on the first stale rung only."""

    def test_hang_stale_sends_dm(self, monkeypatch):
        """escalate=False + discord_user_id -> send_dm called once."""
        from session_orchestration.feed import push_hang_notification

        dm_calls: list = []
        monkeypatch.setattr("tools.discord_tool._get_bot_token", lambda: "tok-hang")
        monkeypatch.setattr(
            "session_orchestration.dm_transport.send_dm",
            lambda user_id, message, token, **kw: dm_calls.append((user_id, message, token)),
        )

        tracker = _FakePostTracker()
        row = _make_row(task_id="t-hang-stale-dm", thread_id=None)
        row["discord_user_id"] = "user-hang-001"
        row["idle_ticks"] = 5

        with patch.object(feed_mod, "_post_discord_message", tracker):
            push_hang_notification(
                "t-hang-stale-dm",
                row,
                escalate=False,
                feed_channel_id="feed-ch-hang",
            )

        assert len(dm_calls) == 1, "send_dm must be called exactly once for stale rung"
        user_id_sent, _msg, token = dm_calls[0]
        assert user_id_sent == "user-hang-001"
        assert token == "tok-hang"

    def test_hang_escalate_no_stale_dm(self, monkeypatch):
        """escalate=True: push_hang_notification must NOT call send_dm (T010 handles it)."""
        from session_orchestration.feed import push_hang_notification

        dm_calls: list = []
        monkeypatch.setattr(
            "session_orchestration.dm_transport.send_dm",
            lambda *a, **kw: dm_calls.append(a),
        )

        tracker = _FakePostTracker()
        row = _make_row(task_id="t-hang-esc-nodm", thread_id=None)
        row["discord_user_id"] = "user-hang-esc"
        row["idle_ticks"] = 5

        with patch.object(feed_mod, "_post_discord_message", tracker):
            push_hang_notification(
                "t-hang-esc-nodm",
                row,
                escalate=True,
                feed_channel_id="feed-ch-hang-esc",
            )

        assert dm_calls == [], (
            "send_dm must NOT be called by push_hang_notification when escalate=True"
        )
