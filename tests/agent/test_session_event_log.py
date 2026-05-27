"""Tests for SessionEventLog — subagent lifecycle events (Phase A)."""

import os
import tempfile
import threading
import time
from pathlib import Path

from agent.session_event_log import (
    EventLog,
    EVENT_SUBAGENT_STARTED,
    EVENT_SUBAGENT_COMPLETED,
    EVENT_SUBAGENT_FAILED,
    EVENT_SUBAGENT_INTERRUPTED,
    EVENT_SUBAGENT_BACKGROUNDED,
    EVENT_SUBAGENT_SEND_MESSAGE,
    EVENT_SWARM_TASK_CLAIMED,
    EVENT_SWARM_TASK_REASSIGNED,
    EVENT_COORDINATOR_NOTIFICATION,
)


class TestSubagentEventConstants:
    """Phase A event constants are defined."""

    def test_event_constants_defined(self):
        assert EVENT_SUBAGENT_STARTED == "subagent.started"
        assert EVENT_SUBAGENT_COMPLETED == "subagent.completed"
        assert EVENT_SUBAGENT_FAILED == "subagent.failed"
        assert EVENT_SUBAGENT_INTERRUPTED == "subagent.interrupted"

    def test_reserved_constants_defined(self):
        """Phase B/C reserved constants exist but are not part of MVP acceptance."""
        assert EVENT_SUBAGENT_BACKGROUNDED == "subagent.backgrounded"
        assert EVENT_SUBAGENT_SEND_MESSAGE == "subagent.send_message"
        assert EVENT_SWARM_TASK_CLAIMED == "swarm.task_claimed"
        assert EVENT_SWARM_TASK_REASSIGNED == "swarm.task_reassigned"
        assert EVENT_COORDINATOR_NOTIFICATION == "coordinator.notification_received"


class TestSubagentEvents:
    """Subagent lifecycle events write and read correctly."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = Path(self.tmpdir) / "test_events.db"
        self.elog = EventLog(db_path=self.db_path)

    def teardown_method(self):
        self.elog.close()
        if self.db_path.exists():
            self.db_path.unlink(missing_ok=True)

    def test_log_subagent_started(self):
        """log_subagent_started writes a valid event."""
        event = self.elog.log_subagent_started(
            task_id="task-1",
            session_id="sess-1",
            subagent_id="sa-0-abc123",
            goal_preview="Research project X",
            parent_id=None,
            agent_id="kimi",
            role="leaf",
            effective_toolsets=["file", "web"],
            blocked_tools=["delegate_task", "terminal"],
            isolation="readonly",
        )
        assert event.type == EVENT_SUBAGENT_STARTED
        assert event.payload["subagent_id"] == "sa-0-abc123"
        assert event.payload["agent_id"] == "kimi"
        assert event.payload["goal_preview"] == "Research project X"

    def test_log_subagent_completed(self):
        """log_subagent_completed writes a valid event."""
        event = self.elog.log_subagent_completed(
            task_id="task-1",
            session_id="sess-1",
            subagent_id="sa-0-abc123",
            status="completed",
            agent_id="kimi",
            role="leaf",
            duration_seconds=12.5,
            api_calls=3,
            tokens={"input": 1000, "output": 500},
            fallback_activations=[
                {
                    "from_model": "deepseek-v4-pro",
                    "to_model": "qwen3.7-max",
                    "reason": "rate_limit",
                }
            ],
            fallback_continuation={"risk": "side_effects_may_have_retried"},
        )
        assert event.type == EVENT_SUBAGENT_COMPLETED
        assert event.payload["status"] == "completed"
        assert event.payload["duration_seconds"] == 12.5
        assert event.payload["api_calls"] == 3
        assert event.payload["fallback_activations"][0]["to_model"] == "qwen3.7-max"
        assert event.payload["fallback_continuation"]["risk"] == "side_effects_may_have_retried"

    def test_log_subagent_failed(self):
        """log_subagent_failed writes a valid event."""
        event = self.elog.log_subagent_failed(
            task_id="task-1",
            session_id="sess-1",
            subagent_id="sa-1-def456",
            error="API timeout after 600s",
            agent_id="claude",
            role="leaf",
        )
        assert event.type == EVENT_SUBAGENT_FAILED
        assert "API timeout" in event.payload["error"]

    def test_log_subagent_interrupted(self):
        """log_subagent_interrupted writes a valid event."""
        event = self.elog.log_subagent_interrupted(
            task_id="task-1",
            session_id="sess-1",
            subagent_id="sa-2-ghi789",
            reason="User pressed /stop",
            agent_id="claude",
            role="leaf",
        )
        assert event.type == EVENT_SUBAGENT_INTERRUPTED
        assert "stop" in event.payload["reason"]

    def test_started_event_has_required_payload(self):
        """subagent.started payload must include subagent_id and goal_preview."""
        event = self.elog.log_subagent_started(
            task_id="t1", session_id="s1",
            subagent_id="sa-x", goal_preview="test",
        )
        # Validation should pass (no ValueError for missing required keys)
        assert event.validate() == []

    def test_events_persisted_and_readable(self):
        """Events are persisted in events.db and can be read back."""
        self.elog.log_subagent_started(
            task_id="task-rw", session_id="sess-rw",
            subagent_id="sa-rw", goal_preview="Read/write test",
        )
        self.elog.log_subagent_completed(
            task_id="task-rw", session_id="sess-rw",
            subagent_id="sa-rw", status="completed",
        )
        events = self.elog.get_events_for_task("task-rw")
        assert len(events) == 2
        types = [e["type"] for e in events]
        assert EVENT_SUBAGENT_STARTED in types
        assert EVENT_SUBAGENT_COMPLETED in types

    def test_event_log_reopens_connection_across_threads(self):
        """Gateway sessions can reuse EventLog from different worker threads."""
        self.elog.log_subagent_started(
            task_id="task-thread", session_id="sess-thread",
            subagent_id="sa-main", goal_preview="main thread",
        )

        error = []

        def worker():
            try:
                self.elog.log_subagent_completed(
                    task_id="task-thread", session_id="sess-thread",
                    subagent_id="sa-main", status="completed",
                )
            except Exception as exc:
                error.append(exc)

        t = threading.Thread(target=worker)
        t.start()
        t.join(timeout=5)

        assert not t.is_alive()
        assert error == []
        assert len(self.elog.get_events_for_task("task-thread")) == 2

    def test_event_log_handles_concurrent_thread_writes(self):
        """Concurrent gateway workers should not trip SQLite thread affinity."""
        errors = []

        def worker(index):
            try:
                self.elog.log_subagent_started(
                    task_id="task-concurrent",
                    session_id="sess-concurrent",
                    subagent_id=f"sa-{index}",
                    goal_preview=f"worker {index}",
                )
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)

        assert errors == []
        assert len(self.elog.get_events_for_task("task-concurrent")) == 8
