"""
Unit tests for session_orchestration/feed.py.

Coverage
--------
1. Optional task-thread notices for user attention and stale/frozen hooks never
   duplicate the channel-level feed digest.
2. The action digest renders deterministic checklist rows with reason,
   priority/staleness, opened age, last-output age, idle/nudge status, and
   thread links.
3. Digest reconciliation posts the first digest, skips unchanged hashes, edits
   existing messages, recreates missing messages, renders empty state, and
   avoids registry mutation on Discord failure.
4. Debounce/re-arm behavior keeps thread-local notices from repeating while the
   digest remains the sole feed_channel_id attention projection.

All tests use fakes — no live Discord calls.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import patch

import pytest

import session_orchestration.feed as feed_mod
from session_orchestration.feed import (
    clear_last_notified,
    push_turn_change,
    reconcile_attention_digest,
    render_attention_digest,
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


class _FakeRegistry:
    def __init__(
        self,
        *,
        attention_items: List[Dict[str, Any]],
        sessions: Optional[Dict[str, Dict[str, Any]]] = None,
        projection: Optional[Dict[str, Any]] = None,
    ):
        self.attention_items = attention_items
        self.sessions = sessions or {}
        self.projection = projection
        self.upserts: List[Dict[str, Any]] = []

    def list_unresolved_attention_items(self) -> List[Dict[str, Any]]:
        return [dict(item) for item in self.attention_items]

    def get(self, task_id: str) -> Optional[Dict[str, Any]]:
        row = self.sessions.get(task_id)
        return dict(row) if row else None

    def get_projection(
        self,
        channel_id: str,
        projection_name: str,
    ) -> Optional[Dict[str, Any]]:
        if self.projection is None:
            return None
        if (
            self.projection.get("channel_id") == channel_id
            and self.projection.get("projection_name") == projection_name
        ):
            return dict(self.projection)
        return None

    def upsert_projection(
        self,
        channel_id: str,
        projection_name: str,
        *,
        message_id: Optional[str] = None,
        content_hash: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self.projection = {
            "channel_id": channel_id,
            "projection_name": projection_name,
            "message_id": message_id,
            "content_hash": content_hash,
            "payload": payload or {},
        }
        self.upserts.append(dict(self.projection))
        return dict(self.projection)


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
    """push_turn_change posts only optional task-thread notices."""

    def test_transition_to_waiting_user_posts_only_to_thread(self):
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

        assert result is True, "should report the task-thread notice"
        assert tracker.calls == [
            (
                "thread-111",
                "🔔 **[claude] t-001** needs your input | my-project\n"
                "State: `RUNNING` → `WAITING_USER`",
            )
        ]

    def test_at_mention_ping_goes_to_feed_thread_gets_detail(self):
        """The @-ping lives in #feed (a fresh router post that links to the
        thread); the thread message carries the detail with NO @-mention (so
        the user isn't double-pinged)."""
        tracker = _FakePostTracker()
        row = _make_row(task_id="t-mention", thread_id="thread-333")
        row["discord_user_id"] = "555000111"
        row["last_question"] = "What should I do next?"

        with patch.object(feed_mod, "_post_discord_message", tracker):
            result = push_turn_change(
                "t-mention",
                row,
                new_state="WAITING_USER",
                old_state="RUNNING",
                feed_channel_id="feed-ch-001",
            )

        assert result is True
        channels = [c for c, _ in tracker.calls]
        assert channels == ["thread-333", "feed-ch-001"], "thread detail + feed ping"

        thread_body = dict(tracker.calls)["thread-333"]
        feed_body = dict(tracker.calls)["feed-ch-001"]
        # Thread: detail, NO @-mention.
        assert "<@555000111>" not in thread_body
        # Feed: the @-ping router linking to the thread, with a question snippet.
        assert feed_body.startswith("<@555000111> ")
        assert "<#thread-333>" in feed_body
        assert "What should I do next?" in feed_body

    def test_no_feed_ping_without_user_id(self):
        """No discord_user_id → no feed router ping (only the thread detail)."""
        tracker = _FakePostTracker()
        row = _make_row(task_id="t-nouid", thread_id="thread-777")

        with patch.object(feed_mod, "_post_discord_message", tracker):
            push_turn_change(
                "t-nouid", row, new_state="WAITING_USER", old_state="RUNNING",
                feed_channel_id="feed-ch-001",
            )

        assert [c for c, _ in tracker.calls] == ["thread-777"]

    def test_transition_to_paused_handoff_posts_once_to_thread(self):
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
        assert [channel_id for channel_id, _ in tracker.calls] == ["thread-222"]

    def test_same_state_second_call_is_suppressed(self):
        """Repeat call with the same state pushes nothing (debounce)."""
        tracker = _FakePostTracker()
        row = _make_row(task_id="t-003", thread_id="thread-333")

        with patch.object(feed_mod, "_post_discord_message", tracker):
            r1 = push_turn_change(
                "t-003",
                row,
                new_state="WAITING_USER",
                old_state="RUNNING",
                feed_channel_id="feed-ch-003",
            )
            r2 = push_turn_change(
                "t-003",
                row,
                new_state="WAITING_USER",
                old_state="RUNNING",
                feed_channel_id="feed-ch-003",
            )

        assert r1 is True, "first call should post to the thread"
        assert r2 is False, "second call (same state) must be suppressed"
        assert [channel_id for channel_id, _ in tracker.calls] == ["thread-333"]

    def test_rearm_after_transition_back(self):
        """Transition back to RUNNING then back to WAITING_USER re-arms."""
        tracker = _FakePostTracker()
        row = _make_row(task_id="t-004", thread_id="thread-444")

        with patch.object(feed_mod, "_post_discord_message", tracker):
            push_turn_change(
                "t-004",
                row,
                new_state="WAITING_USER",
                old_state="RUNNING",
                feed_channel_id="feed-ch-004",
            )
            clear_last_notified("t-004")
            r3 = push_turn_change(
                "t-004",
                row,
                new_state="WAITING_USER",
                old_state="RUNNING",
                feed_channel_id="feed-ch-004",
            )

        assert r3 is True, "re-armed thread notice should post"
        assert [channel_id for channel_id, _ in tracker.calls] == [
            "thread-444",
            "thread-444",
        ]

    def test_menu_renders_question_and_numbered_options(self):
        """A WAITING_USER menu row renders the question + numbered options."""
        tracker = _FakePostTracker()
        row = _make_row(task_id="t-menu", thread_id="thread-menu")
        row["last_question"] = "Apply the proposed edit?"
        row["last_input_kind"] = "menu"
        row["last_options"] = json.dumps(["Accept", "Defer", "Reject"])

        with patch.object(feed_mod, "_post_discord_message", tracker):
            push_turn_change(
                "t-menu", row, new_state="WAITING_USER", old_state="RUNNING",
                feed_channel_id="feed-ch",
            )

        assert len(tracker.calls) == 1
        _, body = tracker.calls[0]
        assert "Apply the proposed edit?" in body
        assert "1. Accept" in body
        assert "2. Defer" in body
        assert "3. Reject" in body
        assert "Reply with the option number" in body

    def test_new_question_within_waiting_user_renotifies(self):
        """A different question in the same WAITING_USER state re-notifies
        (debounce keys on state + question digest, not state alone)."""
        tracker = _FakePostTracker()
        row = _make_row(task_id="t-multi", thread_id="thread-multi")
        row["last_input_kind"] = "menu"

        with patch.object(feed_mod, "_post_discord_message", tracker):
            row["last_question"] = "First question?"
            row["last_options"] = json.dumps(["A", "B"])
            r1 = push_turn_change(
                "t-multi", row, new_state="WAITING_USER", old_state="RUNNING",
            )
            # Same state, SAME question — suppressed.
            r2 = push_turn_change(
                "t-multi", row, new_state="WAITING_USER", old_state="WAITING_USER",
            )
            # Same state, NEW question — must re-notify.
            row["last_question"] = "Second, different question?"
            row["last_options"] = json.dumps(["C", "D"])
            r3 = push_turn_change(
                "t-multi", row, new_state="WAITING_USER", old_state="WAITING_USER",
            )

        assert r1 is True
        assert r2 is False
        assert r3 is True

    def test_no_feed_channel_only_thread_posts(self):
        """When no feed_channel_id is configured, the task thread may still post."""
        tracker = _FakePostTracker()
        row = _make_row(task_id="t-005", thread_id="thread-555")

        with patch.object(feed_mod, "_get_feed_channel_id", return_value=None):
            with patch.object(feed_mod, "_post_discord_message", tracker):
                result = push_turn_change(
                    "t-005",
                    row,
                    new_state="WAITING_USER",
                    old_state="RUNNING",
                    feed_channel_id=None,
                )

        assert result is True
        assert [channel_id for channel_id, _ in tracker.calls] == ["thread-555"]

    def test_no_thread_id_does_not_post_feed_channel_message(self):
        """When row has no discord_thread_id, digest owns the feed channel."""
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

        assert result is False
        assert tracker.calls == []

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

    def test_thread_aliasing_feed_channel_is_not_posted_as_one_off(self):
        tracker = _FakePostTracker()
        row = _make_row(task_id="t-008", thread_id="feed-ch-008")

        with patch.object(feed_mod, "_post_discord_message", tracker):
            result = push_turn_change(
                "t-008",
                row,
                new_state="WAITING_USER",
                old_state="RUNNING",
                feed_channel_id="feed-ch-008",
            )

        assert result is False
        assert tracker.calls == []


class TestPushHangNotification:
    """push_hang_notification posts only optional task-thread notices."""

    def test_hang_notice_posts_only_to_thread(self):
        tracker = _FakePostTracker()
        row = _make_row(task_id="t-hang-001", thread_id="thread-hang")
        row["idle_ticks"] = 4

        with patch.object(feed_mod, "_post_discord_message", tracker):
            result = feed_mod.push_hang_notification(
                "t-hang-001",
                row,
                feed_channel_id="feed-hang",
            )

        assert result is True
        assert [channel_id for channel_id, _ in tracker.calls] == ["thread-hang"]

    def test_hang_notice_never_posts_to_feed_channel_alias(self):
        tracker = _FakePostTracker()
        row = _make_row(task_id="t-hang-002", thread_id="feed-hang")

        with patch.object(feed_mod, "_post_discord_message", tracker):
            result = feed_mod.push_hang_notification(
                "t-hang-002",
                row,
                feed_channel_id="feed-hang",
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


class TestAttentionDigestRenderer:
    def test_renderer_orders_deterministically(self):
        now = "2026-06-29T02:00:00+00:00"
        items = [
            {
                "id": 3,
                "task_id": "task-low",
                "reason": "WAITING_USER",
                "priority": 1,
                "opened_at": "2026-06-29T01:10:00+00:00",
            },
            {
                "id": 2,
                "task_id": "task-high-later",
                "reason": "STALE",
                "priority": 5,
                "opened_at": "2026-06-29T01:30:00+00:00",
            },
            {
                "id": 1,
                "task_id": "task-high-earlier",
                "reason": "STALE",
                "priority": 5,
                "opened_at": "2026-06-29T01:00:00+00:00",
            },
        ]

        content = render_attention_digest(items, now=now)

        assert content.index("task-high-earlier") < content.index("task-high-later")
        assert content.index("task-high-later") < content.index("task-low")

    def test_renderer_is_concise_at_mention_router(self):
        """Each row is a one-line router: thread link + @-mention + identity +
        reason + detail + a truncated question snippet. Verbose telemetry
        (age / idle / nudges / priority prose) lives in the thread, not here."""
        now = "2026-06-29T02:00:00+00:00"
        items = [
            {
                "id": 1,
                "task_id": "task-1",
                "reason": "FROZEN_STALE",
                "priority": 9,
                "detail": "pane hash unchanged",
                "opened_at": "2026-06-29T01:30:00+00:00",
            }
        ]
        sessions = {
            "task-1": {
                "agent": "claude",
                "project": "hermes-agent",
                "tmux_session": "sess-1",
                "discord_thread_id": "thread-1",
                "discord_user_id": "555000",
                "last_question": "Which migration strategy should I use?",
                "last_output_ts": "2026-06-29T01:45:00+00:00",
                "idle_ticks": 4,
                "nudge_count": 2,
            }
        }

        content = render_attention_digest(items, sessions, now=now)

        # Router essentials present.
        assert "<#thread-1>" in content
        assert "<@555000>" in content
        assert "tmux `sess-1`" in content
        assert "claude/hermes-agent" in content
        assert "reason `FROZEN_STALE`" in content
        assert "pane hash unchanged" in content
        assert "Which migration strategy should I use?" in content
        # Verbose telemetry moved to the thread — kept off the board.
        assert "opened 30m ago" not in content
        assert "idle 4 tick(s)" not in content
        assert "P9/frozen/stuck" not in content

    def test_renderer_truncates_long_question(self):
        now = "2026-06-29T02:00:00+00:00"
        items = [{"id": 1, "task_id": "task-1", "reason": "WAITING_USER"}]
        long_q = "word " * 60
        sessions = {"task-1": {"agent": "omp", "discord_thread_id": "t-9",
                               "last_question": long_q}}
        content = render_attention_digest(items, sessions, now=now)
        assert "…" in content
        # No single rendered line should carry the full 300-char question.
        assert long_q.strip() not in content


class TestAttentionDigestReconciler:
    def test_reconciler_first_post_records_projection(self):
        registry = _FakeRegistry(attention_items=[])
        calls: List[Tuple[str, str]] = []

        def fake_post(
            channel_id: str,
            content: str,
            *,
            token: Optional[str] = None,
        ) -> Optional[str]:
            calls.append((channel_id, content))
            return "msg-1"

        with patch.object(feed_mod, "_post_discord_message", fake_post):
            result = reconcile_attention_digest(
                registry,
                feed_channel_id="feed-1",
                now="2026-06-29T02:00:00+00:00",
            )

        assert result["status"] == "posted"
        assert result["message_id"] == "msg-1"
        assert calls == [
            ("feed-1", "**Hermes action feed**\nNo unresolved attention items.")
        ]
        assert registry.upserts[0]["message_id"] == "msg-1"
        assert registry.upserts[0]["payload"] == {"item_count": 0, "task_ids": []}

    def test_reconciler_skips_unchanged_hash(self):
        now = "2026-06-29T02:00:00+00:00"
        content = render_attention_digest([], now=now)
        registry = _FakeRegistry(
            attention_items=[],
            projection={
                "channel_id": "feed-1",
                "projection_name": feed_mod._ATTENTION_DIGEST_PROJECTION_NAME,
                "message_id": "msg-1",
                "content_hash": feed_mod._content_hash(content),
                "payload": {"item_count": 0, "task_ids": []},
            },
        )

        with patch.object(feed_mod, "_post_discord_message") as post:
            with patch.object(feed_mod, "_edit_discord_message") as edit:
                result = reconcile_attention_digest(
                    registry,
                    feed_channel_id="feed-1",
                    now=now,
                )

        assert result["status"] == "unchanged"
        post.assert_not_called()
        edit.assert_not_called()
        assert registry.upserts == []

    def test_reconciler_edits_existing_message(self):
        registry = _FakeRegistry(
            attention_items=[
                {
                    "id": 1,
                    "task_id": "task-1",
                    "reason": "WAITING_USER",
                    "priority": 3,
                    "opened_at": "2026-06-29T01:55:00+00:00",
                }
            ],
            projection={
                "channel_id": "feed-1",
                "projection_name": feed_mod._ATTENTION_DIGEST_PROJECTION_NAME,
                "message_id": "msg-1",
                "content_hash": "old-hash",
                "payload": {},
            },
        )

        with patch.object(
            feed_mod,
            "_edit_discord_message",
            return_value=feed_mod._EDIT_OK,
        ) as edit:
            with patch.object(feed_mod, "_post_discord_message") as post:
                result = reconcile_attention_digest(
                    registry,
                    feed_channel_id="feed-1",
                    now="2026-06-29T02:00:00+00:00",
                )

        assert result["status"] == "edited"
        edit.assert_called_once()
        post.assert_not_called()
        assert registry.upserts[0]["message_id"] == "msg-1"
        assert registry.upserts[0]["payload"] == {
            "item_count": 1,
            "task_ids": ["task-1"],
        }

    def test_reconciler_recreates_missing_message(self):
        registry = _FakeRegistry(
            attention_items=[
                {
                    "id": 1,
                    "task_id": "task-1",
                    "reason": "WAITING_USER",
                    "priority": 100,
                    "opened_at": "2026-06-29T01:55:00+00:00",
                }
            ],
            projection={
                "channel_id": "feed-1",
                "projection_name": feed_mod._ATTENTION_DIGEST_PROJECTION_NAME,
                "message_id": "missing-msg",
                "content_hash": "old-hash",
                "payload": {},
            },
        )

        with patch.object(
            feed_mod,
            "_edit_discord_message",
            return_value=feed_mod._EDIT_MISSING,
        ) as edit:
            with patch.object(
                feed_mod,
                "_post_discord_message",
                return_value="msg-2",
            ) as post:
                result = reconcile_attention_digest(
                    registry,
                    feed_channel_id="feed-1",
                    now="2026-06-29T02:00:00+00:00",
                )

        assert result["status"] == "recreated"
        edit.assert_called_once()
        post.assert_called_once()
        assert registry.upserts[0]["message_id"] == "msg-2"
        assert "task-1" in post.call_args.args[1]
        assert "reason `WAITING_USER`" in post.call_args.args[1]
        assert registry.upserts[0]["payload"] == {
            "item_count": 1,
            "task_ids": ["task-1"],
        }

    def test_reconciler_posts_empty_state_render(self):
        registry = _FakeRegistry(attention_items=[])
        posted: List[str] = []

        def fake_post(
            channel_id: str,
            content: str,
            *,
            token: Optional[str] = None,
        ) -> Optional[str]:
            posted.append(content)
            return "msg-empty"

        with patch.object(feed_mod, "_post_discord_message", fake_post):
            result = reconcile_attention_digest(
                registry,
                feed_channel_id="feed-1",
                now="2026-06-29T02:00:00+00:00",
            )

        assert result["status"] == "posted"
        assert posted == ["**Hermes action feed**\nNo unresolved attention items."]
        assert registry.projection is not None

    def test_reconciler_discord_failure_does_not_mutate_registry(self):
        original_projection = {
            "channel_id": "feed-1",
            "projection_name": feed_mod._ATTENTION_DIGEST_PROJECTION_NAME,
            "message_id": "msg-1",
            "content_hash": "old-hash",
            "payload": {"item_count": 99},
        }
        registry = _FakeRegistry(
            attention_items=[
                {
                    "id": 1,
                    "task_id": "task-1",
                    "reason": "WAITING_USER",
                    "priority": 3,
                    "opened_at": "2026-06-29T01:55:00+00:00",
                }
            ],
            projection=dict(original_projection),
        )
        original_attention_items = [dict(item) for item in registry.attention_items]

        with patch.object(
            feed_mod,
            "_edit_discord_message",
            return_value=feed_mod._EDIT_FAILED,
        ):
            result = reconcile_attention_digest(
                registry,
                feed_channel_id="feed-1",
                now="2026-06-29T02:00:00+00:00",
            )

        assert result["status"] == "discord_failed"
        assert registry.upserts == []
        assert registry.projection == original_projection
        assert registry.attention_items == original_attention_items
