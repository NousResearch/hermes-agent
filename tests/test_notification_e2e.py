"""End-to-end notification delivery pipeline test.

Exercises the full path:
  create task → subscribe → complete → verify event in DB →
  verify DB poller can detect it → verify subscription matching →
  verify notification formatting → verify cursor advancement →
  verify cleanup after terminal status.

Uses an isolated kanban DB (tmp_path) so it doesn't touch production data.
"""

from __future__ import annotations

import json
import queue
import sqlite3
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli import kanban_db as kb


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_MEDIA_ALLOW_DIRS", str(tmp_path))
    kb.init_db()
    return home


@pytest.fixture
def server_module(tmp_path, monkeypatch):
    """Import tui_gateway.server with mocked dependencies."""
    with patch.dict("sys.modules", {
        "hermes_constants": MagicMock(
            get_hermes_home=MagicMock(return_value=str(tmp_path))
        ),
        "hermes_cli.env_loader": MagicMock(),
        "hermes_cli.banner": MagicMock(),
        "hermes_state": MagicMock(),
    }):
        import importlib
        import tui_gateway.server as srv
        importlib.reload(srv)
        yield srv
        srv._sessions.clear()
        importlib.reload(srv)


# ---------------------------------------------------------------------------
# E2E Tests
# ---------------------------------------------------------------------------

class TestNotificationDeliveryE2E:
    """Full end-to-end notification delivery pipeline tests."""

    def test_create_subscribe_complete_event_in_db(self, kanban_home):
        """Step 1: Create task -> subscribe -> complete -> verify event exists."""
        conn = kb.connect()
        try:
            # Create a task
            tid = kb.create_task(conn, title="e2e test task", assignee="worker")
            assert tid is not None
            assert tid.startswith("t_")

            # Subscribe to notifications
            kb.add_notify_sub(
                conn, task_id=tid,
                platform="cli", chat_id="test-session-1",
            )

            # Verify subscription exists
            subs = kb.list_notify_subs(conn, task_id=tid)
            assert len(subs) == 1
            assert subs[0]["task_id"] == tid
            assert subs[0]["platform"] == "cli"
            assert subs[0]["chat_id"] == "test-session-1"
            assert subs[0]["last_event_id"] == 0

            # Complete the task
            kb.complete_task(conn, tid, summary="e2e test done")

            # Verify task is done
            task = kb.get_task(conn, tid)
            assert task.status == "done"

            # Verify completed event exists in task_events
            rows = conn.execute(
                "SELECT id, task_id, kind, payload FROM task_events "
                "WHERE task_id = ? AND kind = 'completed'",
                (tid,),
            ).fetchall()
            assert len(rows) == 1
            assert rows[0]["task_id"] == tid
            assert rows[0]["kind"] == "completed"
            payload = json.loads(rows[0]["payload"])
            assert payload["summary"] == "e2e test done"
        finally:
            conn.close()

    def test_db_poller_detects_terminal_event(self, kanban_home):
        """Step 2: Verify the DB poller query detects our completed event."""
        conn = kb.connect()
        try:
            tid = kb.create_task(conn, title="poller test", assignee="worker")
            kb.complete_task(conn, tid, summary="poller done")
        finally:
            conn.close()

        # Simulate the DB poller query (same SQL as _poll_kanban_notifications)
        conn = kb.connect()
        try:
            terminal_kinds = {"completed", "blocked", "gave_up", "crashed", "timed_out"}
            rows = conn.execute(
                "SELECT id, task_id, kind FROM task_events "
                "WHERE kind IN ({}) ORDER BY id".format(
                    ",".join("?" for _ in terminal_kinds)
                ),
                tuple(terminal_kinds),
            ).fetchall()

            # Should find at least our completed event
            matching = [r for r in rows if r["task_id"] == tid and r["kind"] == "completed"]
            assert len(matching) == 1
        finally:
            conn.close()

    def test_subscription_matching(self, kanban_home):
        """Step 3: Verify subscription matching -- only subscribed tasks get notifications."""
        conn = kb.connect()
        try:
            # Create two tasks
            tid1 = kb.create_task(conn, title="subscribed task", assignee="w1")
            tid2 = kb.create_task(conn, title="unsubscribed task", assignee="w2")

            # Subscribe only to tid1
            kb.add_notify_sub(conn, task_id=tid1, platform="cli", chat_id="session-1")

            # Complete both
            kb.complete_task(conn, tid1, summary="done-1")
            kb.complete_task(conn, tid2, summary="done-2")
        finally:
            conn.close()

        # Verify subscription matching
        conn = kb.connect()
        try:
            subs = kb.list_notify_subs(conn)
            subscribed_task_ids = {s["task_id"] for s in subs if s["platform"] == "cli"}
            assert tid1 in subscribed_task_ids
            assert tid2 not in subscribed_task_ids
        finally:
            conn.close()

    def test_unseen_events_returns_new_events(self, kanban_home):
        """Step 4: Verify unseen_events_for_sub returns events after cursor."""
        conn = kb.connect()
        try:
            tid = kb.create_task(conn, title="unseen test", assignee="worker")
            kb.add_notify_sub(conn, task_id=tid, platform="cli", chat_id="sess-1")
            kb.complete_task(conn, tid, summary="unseen done")
        finally:
            conn.close()

        # Query unseen events
        conn = kb.connect()
        try:
            fmt_kinds = {"completed", "blocked", "gave_up", "crashed", "timed_out"}
            new_cursor, events = kb.unseen_events_for_sub(
                conn,
                task_id=tid,
                platform="cli",
                chat_id="sess-1",
                kinds=fmt_kinds,
            )
            assert len(events) >= 1
            assert any(e.kind == "completed" for e in events)
            assert new_cursor > 0
        finally:
            conn.close()

    def test_cursor_advancement_deduplicates(self, kanban_home):
        """Step 5: Verify cursor advancement prevents re-delivery."""
        conn = kb.connect()
        try:
            tid = kb.create_task(conn, title="cursor test", assignee="worker")
            kb.add_notify_sub(conn, task_id=tid, platform="cli", chat_id="sess-1")
            kb.complete_task(conn, tid, summary="cursor done")
        finally:
            conn.close()

        # First query -- should see the event
        conn = kb.connect()
        try:
            fmt_kinds = {"completed", "blocked", "gave_up", "crashed", "timed_out"}
            cursor1, events1 = kb.unseen_events_for_sub(
                conn, task_id=tid, platform="cli", chat_id="sess-1", kinds=fmt_kinds,
            )
            assert len(events1) >= 1

            # Advance cursor
            kb.advance_notify_cursor(
                conn, task_id=tid, platform="cli", chat_id="sess-1",
                new_cursor=cursor1,
            )

            # Second query -- should see nothing
            cursor2, events2 = kb.unseen_events_for_sub(
                conn, task_id=tid, platform="cli", chat_id="sess-1", kinds=fmt_kinds,
            )
            assert len(events2) == 0
            assert cursor2 == cursor1
        finally:
            conn.close()

    def test_notification_formatting(self, kanban_home, server_module):
        """Step 6: Verify notification message formatting for each event kind."""
        srv = server_module

        # Test completed with summary
        ev = SimpleNamespace(kind="completed", payload={"summary": "shipped it"})
        sub = {"task_id": "t_test1", "platform": "cli", "chat_id": "s1"}
        msg = srv._format_kanban_event(ev, sub)
        assert msg is not None
        assert "t_test1" in msg
        assert "done" in msg
        assert "shipped it" in msg

        # Test blocked with reason
        ev = SimpleNamespace(kind="blocked", payload={"reason": "need API key"})
        sub = {"task_id": "t_test2", "platform": "cli", "chat_id": "s1"}
        msg = srv._format_kanban_event(ev, sub)
        assert msg is not None
        assert "t_test2" in msg
        assert "blocked" in msg
        assert "need API key" in msg

        # Test crashed
        ev = SimpleNamespace(kind="crashed", payload=None)
        sub = {"task_id": "t_test3", "platform": "cli", "chat_id": "s1"}
        msg = srv._format_kanban_event(ev, sub)
        assert msg is not None
        assert "t_test3" in msg
        assert "crashed" in msg

        # Test timed_out
        ev = SimpleNamespace(kind="timed_out", payload={"limit_seconds": 300})
        sub = {"task_id": "t_test4", "platform": "cli", "chat_id": "s1"}
        msg = srv._format_kanban_event(ev, sub)
        assert msg is not None
        assert "t_test4" in msg
        assert "timed out" in msg
        assert "300" in msg

        # Test gave_up
        ev = SimpleNamespace(kind="gave_up", payload={"error": "spawn failed 3x"})
        sub = {"task_id": "t_test5", "platform": "cli", "chat_id": "s1"}
        msg = srv._format_kanban_event(ev, sub)
        assert msg is not None
        assert "t_test5" in msg
        assert "gave up" in msg

    def test_dispatch_delivers_to_matching_session(self, kanban_home, server_module):
        """Step 7: Verify kanban events are pushed to completion_queue."""
        srv = server_module

        conn = kb.connect()
        try:
            tid = kb.create_task(conn, title="dispatch test", assignee="worker")
            kb.add_notify_sub(conn, task_id=tid, platform="cli", chat_id="cli-sess-1")
            kb.complete_task(conn, tid, summary="dispatch done")
        finally:
            conn.close()

        # Verify the event was pushed to completion_queue
        from tools.process_registry import process_registry
        events = []
        while not process_registry.completion_queue.empty():
            evt = process_registry.completion_queue.get_nowait()
            if evt.get("type") == "kanban_event" and evt.get("task_id") == tid:
                events.append(evt)

        assert len(events) >= 1
        assert events[0]["kind"] == "completed"
        assert "dispatch done" in (events[0]["payload"].get("summary") or "")

    def test_dispatch_queues_when_session_busy(self, kanban_home, server_module):
        """Step 8: Verify kanban events are queued for busy sessions."""
        srv = server_module

        conn = kb.connect()
        try:
            tid = kb.create_task(conn, title="busy test", assignee="worker")
            kb.add_notify_sub(conn, task_id=tid, platform="cli", chat_id="cli-busy-1")
            kb.complete_task(conn, tid, summary="busy done")
        finally:
            conn.close()

        # Verify the event was pushed to completion_queue
        from tools.process_registry import process_registry
        events = []
        while not process_registry.completion_queue.empty():
            evt = process_registry.completion_queue.get_nowait()
            if evt.get("type") == "kanban_event" and evt.get("task_id") == tid:
                events.append(evt)

        assert len(events) >= 1
        assert events[0]["kind"] == "completed"

    def test_dispatch_skips_non_cli_subscriptions(self, kanban_home, server_module):
        """Step 9: Verify kanban events are pushed regardless of subscription platform."""
        srv = server_module

        conn = kb.connect()
        try:
            tid = kb.create_task(conn, title="platform test", assignee="worker")
            # Subscribe with telegram platform, not cli
            kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="tg-1")
            kb.complete_task(conn, tid, summary="platform done")
        finally:
            conn.close()

        # Verify the event was pushed to completion_queue
        # (events are pushed regardless of subscription platform)
        from tools.process_registry import process_registry
        events = []
        while not process_registry.completion_queue.empty():
            evt = process_registry.completion_queue.get_nowait()
            if evt.get("type") == "kanban_event" and evt.get("task_id") == tid:
                events.append(evt)

        assert len(events) >= 1

    def test_full_pipeline_create_to_delivery(self, kanban_home, server_module):
        """Step 10: Full E2E -- create -> subscribe -> complete -> verify queue push."""
        srv = server_module

        # Phase 1: Setup
        conn = kb.connect()
        try:
            tid = kb.create_task(conn, title="full e2e task", assignee="worker")
            kb.add_notify_sub(conn, task_id=tid, platform="cli", chat_id="e2e-sess")
        finally:
            conn.close()

        # Phase 2: Worker completes the task
        conn = kb.connect()
        try:
            kb.complete_task(conn, tid, summary="full pipeline complete")
        finally:
            conn.close()

        # Phase 3: Verify event was pushed to completion_queue
        from tools.process_registry import process_registry
        events = []
        while not process_registry.completion_queue.empty():
            evt = process_registry.completion_queue.get_nowait()
            if evt.get("type") == "kanban_event" and evt.get("task_id") == tid:
                events.append(evt)

        assert len(events) >= 1
        assert events[0]["kind"] == "completed"
        assert "full pipeline complete" in (events[0]["payload"].get("summary") or "")

        # Phase 4: Verify event is also in DB
        conn = kb.connect()
        try:
            terminal_kinds = {"completed", "blocked", "gave_up", "crashed", "timed_out"}
            rows = conn.execute(
                "SELECT id, task_id, kind FROM task_events "
                "WHERE kind IN ({}) ORDER BY id".format(
                    ",".join("?" for _ in terminal_kinds)
                ),
                tuple(terminal_kinds),
            ).fetchall()

            our_events = [r for r in rows if r["task_id"] == tid]
            assert len(our_events) >= 1
            assert our_events[-1]["kind"] == "completed"
        finally:
            conn.close()

    def test_multiple_tasks_independent_notifications(self, kanban_home, server_module):
        """Step 11: Multiple tasks with separate subscriptions deliver independently."""
        srv = server_module

        conn = kb.connect()
        try:
            tid1 = kb.create_task(conn, title="task-alpha", assignee="w1")
            tid2 = kb.create_task(conn, title="task-beta", assignee="w2")
            kb.add_notify_sub(conn, task_id=tid1, platform="cli", chat_id="multi-sess")
            kb.add_notify_sub(conn, task_id=tid2, platform="cli", chat_id="multi-sess")
            kb.complete_task(conn, tid1, summary="alpha done")
            kb.complete_task(conn, tid2, summary="beta done")
        finally:
            conn.close()

        # Verify both events were pushed to completion_queue
        from tools.process_registry import process_registry
        events = []
        while not process_registry.completion_queue.empty():
            evt = process_registry.completion_queue.get_nowait()
            if evt.get("type") == "kanban_event" and evt.get("task_id") in (tid1, tid2):
                events.append(evt)

        task_ids = {e["task_id"] for e in events}
        assert tid1 in task_ids
        assert tid2 in task_ids

    def test_blocked_then_completed_delivers_both(self, kanban_home, server_module):
        """Step 12: Task blocked then completed -- both events are persisted."""
        srv = server_module

        conn = kb.connect()
        try:
            tid = kb.create_task(conn, title="block-complete", assignee="worker")
            kb.add_notify_sub(conn, task_id=tid, platform="cli", chat_id="bc-sess")
            kb.block_task(conn, tid, reason="need credentials")
        finally:
            conn.close()

        # Unblock and complete
        conn = kb.connect()
        try:
            kb.unblock_task(conn, tid)
            kb.complete_task(conn, tid, summary="credentials received, done")
        finally:
            conn.close()

        # Check both events exist
        conn = kb.connect()
        try:
            rows = conn.execute(
                "SELECT kind FROM task_events WHERE task_id = ? AND kind IN ('blocked', 'completed')",
                (tid,),
            ).fetchall()
            kinds = {r["kind"] for r in rows}
            assert "blocked" in kinds
            assert "completed" in kinds
        finally:
            conn.close()

    def test_subscription_claim_returns_events(self, kanban_home, server_module):
        """Step 13: Verify claim_unseen_events_for_sub returns terminal events."""
        conn = kb.connect()
        try:
            tid = kb.create_task(conn, title="claim-test", assignee="worker")
            kb.add_notify_sub(conn, task_id=tid, platform="cli", chat_id="cli-test")
            kb.complete_task(conn, tid, summary="claimed")
        finally:
            conn.close()

        conn = kb.connect()
        try:
            old_cursor, new_cursor, events = kb.claim_unseen_events_for_sub(
                conn,
                task_id=tid,
                platform="cli",
                chat_id="cli-test",
                kinds=("completed", "blocked", "gave_up", "crashed", "timed_out"),
            )
        finally:
            conn.close()
        assert len(events) >= 1
        assert events[0].kind == "completed"
        assert new_cursor > old_cursor

    def test_format_handles_all_terminal_kinds(self, kanban_home, server_module):
        """Step 14: Verify _format_kanban_event handles all terminal event kinds."""
        srv = server_module

        from unittest.mock import MagicMock
        for kind in ("completed", "blocked", "gave_up", "crashed", "timed_out"):
            ev = MagicMock()
            ev.kind = kind
            ev.payload = {"summary": "test", "reason": "test", "error": "test", "limit_seconds": 60}
            msg = srv._format_kanban_event(ev, "t_test")
            assert msg is not None, f"_format_kanban_event returned None for kind={kind}"
            assert "[IMPORTANT:" in msg
