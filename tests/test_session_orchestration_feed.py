"""
Unit tests for session_orchestration/feed.py (T007 / T-FEED-003).

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
6. Both absent: no messages posted, returns None.

T-FEED-003 coverage (self-pruning feed-board):
(a) First non-terminal transition: POST to feed → returns new id.
(b) Second non-terminal transition with existing feed_message_id: PATCH same
    id → returns same id (no new POST).
(c) Terminal transition (DONE/ERROR) with existing feed_message_id: DELETE
    feed message → returns None.
(d) Per-session THREAD always receives a POST on every transition (append-only
    unchanged) even when feed uses PATCH or DELETE.
(e) feed_message_id column migration is idempotent (registry schema test).

All tests use fakes — no live Discord calls.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import patch

import pytest

import session_orchestration.feed as feed_mod
from session_orchestration.feed import (
    clear_last_notified,
    push_turn_change,
)


# ---------------------------------------------------------------------------
# Fake Discord transport helpers
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


class _FakeEditTracker:
    """Records calls to _edit_discord_message; always returns True."""

    def __init__(self):
        self.calls: List[Tuple[str, str, str]] = []  # [(channel_id, msg_id, content)]

    def __call__(
        self,
        channel_id: str,
        message_id: str,
        content: str,
        *,
        token: Optional[str] = None,
    ) -> bool:
        self.calls.append((channel_id, message_id, content))
        return True


class _FakeDeleteTracker:
    """Records calls to _delete_discord_message; always returns True."""

    def __init__(self):
        self.calls: List[Tuple[str, str]] = []  # [(channel_id, msg_id)]

    def __call__(
        self,
        channel_id: str,
        message_id: str,
        *,
        token: Optional[str] = None,
    ) -> bool:
        self.calls.append((channel_id, message_id))
        return True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_row(
    task_id: str = "task-001",
    agent: str = "claude",
    thread_id: Optional[str] = "thread-999",
    project: str = "my-project",
    feed_message_id: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "task_id": task_id,
        "agent": agent,
        "discord_thread_id": thread_id,
        "project": project,
        "state": "RUNNING",
        "feed_message_id": feed_message_id,
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

        # Returns the new feed message id (non-None = success)
        assert result is not None, "should return a feed message id when posted"
        assert isinstance(result, str), "feed message id must be a string"
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

        assert result is not None
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

        assert r1 is not None, "first call should return a feed message id"
        assert r2 is None, "second call (same state) must be suppressed → None"
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

        assert r3 is not None, "re-armed notification should return a feed message id"
        # 2 posts per transition × 2 transitions = 4 posts total
        assert len(tracker.calls) == count_after_first * 2

    def test_no_feed_channel_only_thread_posts(self):
        """When no feed_channel_id is configured, only the thread is posted.

        Return value is None (no feed message id) since no feed post was made.
        """
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
        # Returns None because no feed channel post was made
        assert result is None

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
        # Returns the feed message id
        assert result is not None
        assert isinstance(result, str)

    def test_no_feed_channel_no_thread_returns_none(self):
        """Both absent: no posts, returns None."""
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

        assert result is None
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

        assert result is not None, "should return a feed message id"
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
        assert result is not None, "must return a feed message id even when no question present"
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
        """new_state=DONE must produce a message containing '✅'.

        DONE is terminal so the feed channel receives a DELETE (not a POST).
        We verify icon content via the thread POST, which is always append-only.
        """
        post_tracker = _FakePostTracker()
        delete_tracker = _FakeDeleteTracker()
        # Give the row a thread so we can read the posted message content,
        # and an existing feed_message_id so the DELETE path fires.
        row = _make_row(
            task_id="t-done-icon",
            thread_id="thread-done-icon",
            feed_message_id="existing-feed-msg",
        )

        with (
            patch.object(feed_mod, "_post_discord_message", post_tracker),
            patch.object(feed_mod, "_delete_discord_message", delete_tracker),
        ):
            result = push_turn_change(
                "t-done-icon",
                row,
                new_state="DONE",
                old_state="RUNNING",
                feed_channel_id="feed-ch-done",
            )

        # Terminal state → feed message deleted, result is None
        assert result is None, "DONE is terminal; feed message id must be None"
        assert len(delete_tracker.calls) == 1, "feed message must be DELETEd"
        # Thread still receives a new POST (append-only)
        assert post_tracker.calls, "thread POST must fire even for DONE"
        for _, content in post_tracker.calls:
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

        assert result is not None, "RUNNING is non-terminal; must return a feed message id"
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


# ---------------------------------------------------------------------------
# T-FEED-003 — self-pruning feed board
# ---------------------------------------------------------------------------


class TestFeedBoardSingleMessage:
    """T-FEED-003: feed channel uses POST / PATCH / DELETE for a single board entry."""

    def test_first_transition_posts_to_feed_returns_new_id(self):
        """(a) First non-terminal transition: POST to feed → returns a new message id."""
        post_tracker = _FakePostTracker()
        edit_tracker = _FakeEditTracker()
        delete_tracker = _FakeDeleteTracker()
        # Row has no existing feed_message_id (fresh session)
        row = _make_row(task_id="t-fb-001", thread_id="thread-fb-001", feed_message_id=None)

        with (
            patch.object(feed_mod, "_post_discord_message", post_tracker),
            patch.object(feed_mod, "_edit_discord_message", edit_tracker),
            patch.object(feed_mod, "_delete_discord_message", delete_tracker),
        ):
            result = push_turn_change(
                "t-fb-001",
                row,
                new_state="WAITING_USER",
                old_state="RUNNING",
                feed_channel_id="feed-ch-fb",
            )

        # Must POST (not PATCH or DELETE) to the feed channel
        feed_posts = [ch for ch, _ in post_tracker.calls if ch == "feed-ch-fb"]
        assert len(feed_posts) == 1, "exactly one POST to feed channel on first transition"
        assert result is not None, "must return a non-None feed message id"
        assert isinstance(result, str), "feed message id must be a string"
        # No PATCH or DELETE should have been called
        assert edit_tracker.calls == [], "no PATCH on first POST"
        assert delete_tracker.calls == [], "no DELETE on first POST"

    def test_second_transition_patches_same_id(self):
        """(b) Second non-terminal transition with existing feed_message_id: PATCH, no new POST."""
        post_tracker = _FakePostTracker()
        edit_tracker = _FakeEditTracker()
        delete_tracker = _FakeDeleteTracker()
        existing_id = "existing-feed-msg-42"
        row = _make_row(
            task_id="t-fb-002",
            thread_id="thread-fb-002",
            feed_message_id=existing_id,
        )

        with (
            patch.object(feed_mod, "_post_discord_message", post_tracker),
            patch.object(feed_mod, "_edit_discord_message", edit_tracker),
            patch.object(feed_mod, "_delete_discord_message", delete_tracker),
        ):
            result = push_turn_change(
                "t-fb-002",
                row,
                new_state="PAUSED_HANDOFF",
                old_state="RUNNING",
                feed_channel_id="feed-ch-fb2",
            )

        # Feed channel must NOT receive a new POST (existing id retained)
        feed_posts = [ch for ch, _ in post_tracker.calls if ch == "feed-ch-fb2"]
        assert feed_posts == [], "no new POST to feed channel when existing id present"
        # Must PATCH the existing message
        assert len(edit_tracker.calls) == 1, "exactly one PATCH"
        patched_ch, patched_id, _ = edit_tracker.calls[0]
        assert patched_ch == "feed-ch-fb2"
        assert patched_id == existing_id, "PATCH must target the existing message id"
        # Return value must be the SAME id (retained)
        assert result == existing_id, "PATCH must return the same feed_message_id"
        # No DELETE
        assert delete_tracker.calls == [], "no DELETE on non-terminal PATCH"

    def test_terminal_transition_deletes_feed_message_returns_none(self):
        """(c) Terminal transition (DONE) with existing feed_message_id: DELETE → returns None."""
        post_tracker = _FakePostTracker()
        edit_tracker = _FakeEditTracker()
        delete_tracker = _FakeDeleteTracker()
        existing_id = "board-msg-to-reap"
        row = _make_row(
            task_id="t-fb-003",
            thread_id="thread-fb-003",
            feed_message_id=existing_id,
        )

        with (
            patch.object(feed_mod, "_post_discord_message", post_tracker),
            patch.object(feed_mod, "_edit_discord_message", edit_tracker),
            patch.object(feed_mod, "_delete_discord_message", delete_tracker),
        ):
            result = push_turn_change(
                "t-fb-003",
                row,
                new_state="DONE",
                old_state="RUNNING",
                feed_channel_id="feed-ch-fb3",
            )

        # Must DELETE the existing feed message
        assert len(delete_tracker.calls) == 1, "exactly one DELETE"
        del_ch, del_id = delete_tracker.calls[0]
        assert del_ch == "feed-ch-fb3"
        assert del_id == existing_id, "DELETE must target the existing message id"
        # Must return None (reaped)
        assert result is None, "terminal transition must return None"
        # No new POST or PATCH
        feed_posts = [ch for ch, _ in post_tracker.calls if ch == "feed-ch-fb3"]
        assert feed_posts == [], "no new POST on terminal transition"
        assert edit_tracker.calls == [], "no PATCH on terminal transition"

    def test_error_state_also_deletes_feed_message(self):
        """(c) ERROR is also terminal: DELETE existing feed entry → returns None."""
        post_tracker = _FakePostTracker()
        edit_tracker = _FakeEditTracker()
        delete_tracker = _FakeDeleteTracker()
        existing_id = "board-msg-error-reap"
        row = _make_row(
            task_id="t-fb-004",
            thread_id="thread-fb-004",
            feed_message_id=existing_id,
        )

        with (
            patch.object(feed_mod, "_post_discord_message", post_tracker),
            patch.object(feed_mod, "_edit_discord_message", edit_tracker),
            patch.object(feed_mod, "_delete_discord_message", delete_tracker),
        ):
            result = push_turn_change(
                "t-fb-004",
                row,
                new_state="ERROR",
                old_state="RUNNING",
                feed_channel_id="feed-ch-fb4",
            )

        assert len(delete_tracker.calls) == 1
        assert result is None

    def test_thread_always_receives_post_on_every_transition(self):
        """(d) Thread always receives a new POST regardless of PATCH/DELETE on feed."""
        post_tracker = _FakePostTracker()
        edit_tracker = _FakeEditTracker()
        delete_tracker = _FakeDeleteTracker()
        existing_id = "board-msg-patch-path"
        thread_id = "thread-fb-append-only"
        row = _make_row(
            task_id="t-fb-005",
            thread_id=thread_id,
            feed_message_id=existing_id,
        )

        with (
            patch.object(feed_mod, "_post_discord_message", post_tracker),
            patch.object(feed_mod, "_edit_discord_message", edit_tracker),
            patch.object(feed_mod, "_delete_discord_message", delete_tracker),
        ):
            # Non-terminal: PATCH on feed, POST on thread
            push_turn_change(
                "t-fb-005",
                row,
                new_state="WAITING_USER",
                old_state="RUNNING",
                feed_channel_id="feed-ch-fb5",
            )

        # Feed was PATCHed (no new post to feed channel)
        feed_posts = [ch for ch, _ in post_tracker.calls if ch == "feed-ch-fb5"]
        assert feed_posts == [], "feed channel must not receive a new POST on PATCH path"
        # Thread must still have received a new POST (append-only)
        thread_posts = [ch for ch, _ in post_tracker.calls if ch == thread_id]
        assert len(thread_posts) == 1, "thread must always receive a new POST"

        # Now verify DELETE path also keeps thread append-only
        feed_mod._last_notified.clear()
        row2 = _make_row(
            task_id="t-fb-005b",
            thread_id=thread_id,
            feed_message_id=existing_id,
        )
        post_tracker.calls.clear()
        edit_tracker.calls.clear()
        delete_tracker.calls.clear()

        with (
            patch.object(feed_mod, "_post_discord_message", post_tracker),
            patch.object(feed_mod, "_edit_discord_message", edit_tracker),
            patch.object(feed_mod, "_delete_discord_message", delete_tracker),
        ):
            push_turn_change(
                "t-fb-005b",
                row2,
                new_state="DONE",
                old_state="RUNNING",
                feed_channel_id="feed-ch-fb5",
            )

        # Feed was DELETEd
        assert len(delete_tracker.calls) == 1
        # Thread still received a new POST (append-only, even on terminal)
        thread_posts_terminal = [ch for ch, _ in post_tracker.calls if ch == thread_id]
        assert len(thread_posts_terminal) == 1, (
            "thread must receive a new POST even on terminal (DONE/ERROR) transitions"
        )


class TestFeedMessageIdMigration:
    """(e) feed_message_id column migration is idempotent."""

    def test_migration_is_idempotent(self):
        """Calling _migrate_schema twice on the same DB must not raise."""
        from session_orchestration.registry import SessionOrchestrationRegistry

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_state.db"
            # First instantiation: creates schema + runs migration
            reg1 = SessionOrchestrationRegistry(db_path=db_path)
            # Second instantiation: migration is idempotent (ALTER TABLE must
            # catch OperationalError when column already exists)
            reg2 = SessionOrchestrationRegistry(db_path=db_path)
            # Verify the column exists by inserting a row with feed_message_id
            reg1.upsert(
                "t-migrate-001",
                agent="claude",
                feed_message_id="test-msg-id",
            )
            row = reg1.get("t-migrate-001")
            assert row is not None
            assert row.get("feed_message_id") == "test-msg-id"

    def test_set_feed_message_id_updates_and_clears(self):
        """set_feed_message_id sets and clears the feed_message_id column."""
        from session_orchestration.registry import SessionOrchestrationRegistry

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_state2.db"
            reg = SessionOrchestrationRegistry(db_path=db_path)
            reg.upsert("t-sfmi-001", agent="claude")

            # Set a value
            reg.set_feed_message_id("t-sfmi-001", "msg-id-abc")
            row = reg.get("t-sfmi-001")
            assert row["feed_message_id"] == "msg-id-abc"

            # Clear it (None)
            reg.set_feed_message_id("t-sfmi-001", None)
            row = reg.get("t-sfmi-001")
            assert row["feed_message_id"] is None
