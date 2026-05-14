import pytest
from gateway.run import (
    _safe_int,
    _safe_bool,
    _heartbeat_eligibility,
    _format_heartbeat_message,
    _blocked_reminder_should_send,
    _blocked_notified_at,
    _reset_blocked_tracker,
    _load_kanban_board_state,
    _BLOCKED_REMINDER_INTERVAL_DEFAULT,
)

class TestSafeInt:
    def test_none_returns_default(self):
        assert _safe_int(None) == 0
        assert _safe_int(None, 5) == 5

    def test_valid_int(self):
        assert _safe_int(5) == 5
        assert _safe_int(10, 99) == 10

    def test_invalid_type(self):
        assert _safe_int("abc") == 0
        assert _safe_int("abc", 7) == 7
        assert _safe_int([1,2]) == 0

    def test_float_coercion(self):
        assert _safe_int(3.7) == 3


class TestSafeBool:
    def test_none_returns_default(self):
        assert _safe_bool(None) is False
        assert _safe_bool(None, True) is True

    def test_valid_bool(self):
        assert _safe_bool(True) is True
        assert _safe_bool(False) is False

    def test_truthy_values(self):
        assert _safe_bool(1) is True
        assert _safe_bool("yes") is True
        assert _safe_bool([1]) is True

    def test_falsey_values(self):
        assert _safe_bool(0) is False
        assert _safe_bool("") is False
        assert _safe_bool([]) is False


class TestHeartbeatEligibility:
    def _eligibility(self, session, board, process):
        return _heartbeat_eligibility(session, board, process)

    def test_terminal_session_never_sends(self):
        result, label = self._eligibility({"status": "terminal", "is_kanban_worker": False}, {}, {})
        assert result is False
        assert label == "terminal"

    def test_completed_kanban_task(self):
        for status in ["done", "completed", "cancelled", "archived"]:
            result, label = self._eligibility(
                {"is_kanban_worker": True, "kanban_task_id": "t_123"},
                {"status": status},
                {},
            )
            assert result is False
            assert label == "completed"

    def test_blocked_kanban_sends(self):
        result, label = self._eligibility(
            {"is_kanban_worker": True, "kanban_task_id": "t_123"},
            {"status": "blocked"},
            {},
        )
        assert result is True
        assert label == "blocked"

    def test_idle_kanban_no_agent(self):
        result, label = self._eligibility(
            {"is_kanban_worker": True, "kanban_task_id": "t_123", "agent_active": False},
            {},
            {},
        )
        assert result is False
        assert label == "idle"

    def test_active_kanban_with_agent(self):
        result, label = self._eligibility(
            {"is_kanban_worker": True, "kanban_task_id": "t_123", "agent_active": True},
            {},
            {},
        )
        assert result is True
        assert label == "active"

    def test_blocked_board_non_kanban(self):
        result, label = self._eligibility(
            {"is_kanban_worker": False},
            {"status": "blocked"},
            {},
        )
        assert result is True
        assert label == "blocked"

    def test_empty_process_idle(self):
        result, label = self._eligibility({}, {}, {})
        assert result is False
        assert label == "idle"

    def test_active_process_sends(self):
        result, label = self._eligibility({}, {}, {"active_process_count": 1})
        assert result is True
        assert label == "active"

    def test_active_process_sends_even_if_agent_active_false(self):
        result, label = self._eligibility(
            {"agent_active": False, "is_kanban_worker": False},
            {},
            {"active_process_count": 1, "active_kanban_count": 0},
        )
        assert result is True
        assert label == "active"

    def test_active_kanban_count_sends(self):
        result, label = self._eligibility({}, {}, {"active_process_count": 0, "active_kanban_count": 1})
        assert result is True
        assert label == "active"

    def test_idle_non_kanban_no_active_process(self):
        result, label = self._eligibility(
            {"is_kanban_worker": False, "agent_active": False},
            {},
            {"active_process_count": 0, "active_kanban_count": 0},
        )
        assert result is False
        assert label == "idle"

    def test_active_non_kanban_with_process(self):
        result, label = self._eligibility(
            {"is_kanban_worker": False, "agent_active": True},
            {},
            {"active_process_count": 1},
        )
        assert result is True
        assert label == "active"

    def test_no_kanban_no_board_idle(self):
        result, label = self._eligibility(
            {"is_kanban_worker": False},
            None,
            {"active_process_count": 0, "active_kanban_count": 0},
        )
        assert result is False
        assert label == "idle"


class TestHeartbeatNoneState:
    def _eligibility(self, session, board, process):
        return _heartbeat_eligibility(session, board, process)

    def test_process_count_none(self):
        result, label = self._eligibility({}, {}, {"active_process_count": None})
        assert result is False
        assert label == "idle"

    def test_kanban_count_none(self):
        result, label = self._eligibility({}, {}, {"active_process_count": 0, "active_kanban_count": None})
        assert result is False
        assert label == "idle"

    def test_active_kanban_count_none_with_active_process(self):
        result, label = self._eligibility(
            {"is_kanban_worker": False},
            {},
            {"active_process_count": 1, "active_kanban_count": None},
        )
        assert result is True
        assert label == "active"

    def test_agent_active_none_with_active_process(self):
        result, label = self._eligibility(
            {"agent_active": None, "is_kanban_worker": False},
            {},
            {"active_process_count": 1},
        )
        assert result is True
        assert label == "active"


class TestBlockedRateLimit:
    """Tests for blocked-card reminder rate-limiting."""

    def setup_method(self):
        # Save and restore so parallel workers with shared module state
        # don't corrupt each other's tracker.
        self._saved = dict(_blocked_notified_at)
        _reset_blocked_tracker()

    def teardown_method(self):
        _reset_blocked_tracker()
        _blocked_notified_at.update(self._saved)

    def test_first_blocked_notification_sent(self):
        # First time a card is blocked — should always send.
        assert _blocked_reminder_should_send("t_first", "needs approval", 1000.0) is True
        # Tracker was updated.
        assert _blocked_notified_at["t_first"] == (1000.0, "needs approval")

    def test_same_card_within_interval_skipped(self):
        # Notify at t=1000.
        assert _blocked_reminder_should_send("t_same", "needs approval", 1000.0) is True
        # Same card, same reason, 60 s later — well within 60-min default -> blocked.
        assert _blocked_reminder_should_send("t_same", "needs approval", 1060.0) is False

    def test_same_card_after_long_interval_sent(self):
        # Notify at t=1000.
        assert _blocked_reminder_should_send("t_long", "needs approval", 1000.0) is True
        # 61 minutes later — past the 60-min (3600 s) default -> allowed.
        assert _blocked_reminder_should_send("t_long", "needs approval", 4660.0) is True
        # Tracker updated to new timestamp.
        assert _blocked_notified_at["t_long"][0] == 4660.0

    def test_changed_blocked_reason_triggers_immediately(self):
        # Notify at t=1000.
        assert _blocked_reminder_should_send("t_reason", "needs approval", 1000.0) is True
        # Same card but different reason — should send immediately (fp mismatch).
        assert _blocked_reminder_should_send("t_reason", "user replied", 1060.0) is True
        # Tracker updated with new fingerprint.
        assert _blocked_notified_at["t_reason"] == (1060.0, "user replied")

    def test_empty_reason_tracked_correctly(self):
        # Empty string and None both normalise to "" fingerprint.
        # First call with empty reason on a new card — sent.
        assert _blocked_reminder_should_send("t_empty", "", 1000.0) is True
        # Second call with None on same card, 60 s later — same fp, blocked.
        assert _blocked_reminder_should_send("t_empty", None, 1060.0) is False

    def test_custom_reminder_interval_short(self):
        # When interval is set very short (e.g. 10 s), next tick within that
        # interval is still blocked.
        assert _blocked_reminder_should_send("t_short", "a", 1000.0, reminder_interval=10.0) is True
        assert _blocked_reminder_should_send("t_short", "a", 1005.0, reminder_interval=10.0) is False
        assert _blocked_reminder_should_send("t_short", "a", 1011.0, reminder_interval=10.0) is True

    def test_custom_reminder_interval_zero_disables(self):
        # Zero interval means always send (no rate-limiting).
        assert _blocked_reminder_should_send("t_zero", "a", 1000.0, reminder_interval=0.0) is True
        assert _blocked_reminder_should_send("t_zero", "a", 1001.0, reminder_interval=0.0) is True

    def test_different_cards_independent(self):
        # Each card has its own entry in the tracker.
        assert _blocked_reminder_should_send("t_card_A", "reason1", 1000.0) is True
        assert _blocked_reminder_should_send("t_card_B", "reason2", 1000.0) is True
        # t_card_A checked again 61 min later — interval expired -> allowed.
        assert _blocked_reminder_should_send("t_card_A", "reason1", 4660.0) is True
        # t_card_B checked again just 60 s later — still within 60-min -> blocked.
        assert _blocked_reminder_should_send("t_card_B", "reason2", 1060.0) is False

    def test_default_interval_is_60_minutes(self):
        assert _BLOCKED_REMINDER_INTERVAL_DEFAULT == 3600

    def test_custom_tracker_is_isolated(self):
        tracker = {}
        assert _blocked_reminder_should_send("t_local", "a", 1000.0, tracker=tracker) is True
        assert tracker == {"t_local": (1000.0, "a")}
        assert "t_local" not in _blocked_notified_at
        _reset_blocked_tracker(tracker)
        assert tracker == {}


class TestKanbanBoardState:
    def test_load_state_uses_canonical_db_path_and_block_reason(self, tmp_path):
        from hermes_cli import kanban_db as kb

        db_path = tmp_path / "kanban.db"
        with kb.connect(db_path=db_path) as conn:
            task_id = kb.create_task(conn, title="Review deployment", body="check", assignee=None)
            assert kb.block_task(conn, task_id, reason="needs human approval") is True

        state = _load_kanban_board_state(task_id, db_path=str(db_path))

        assert state == {
            "status": "blocked",
            "title": "Review deployment",
            "blocked_reason": "needs human approval",
        }

    def test_load_state_handles_done_task(self, tmp_path):
        from hermes_cli import kanban_db as kb

        db_path = tmp_path / "kanban.db"
        with kb.connect(db_path=db_path) as conn:
            task_id = kb.create_task(conn, title="Ship patch", body="done", assignee=None)
            assert kb.complete_task(conn, task_id, result="merged") is True

        state = _load_kanban_board_state(task_id, db_path=str(db_path))

        assert state == {
            "status": "done",
            "title": "Ship patch",
        }


class TestFormatHeartbeat:
    def _format(self, state_label, session, board, process, elapsed=5):
        return _format_heartbeat_message(state_label, session, board, process, elapsed)

    def test_blocked_message(self):
        msg = self._format(
            "blocked",
            {"is_kanban_worker": True, "kanban_task_id": "t_abc", "session_id": "s_1"},
            {"title": "My Card", "blocked_reason": "needs approval"},
            {},
            10,
        )
        assert "10 min" in msg
        assert "My Card" in msg
        assert "t_abc" in msg
        assert "needs approval" in msg

    def test_terminal_message(self):
        msg = self._format(
            "terminal",
            {"session_id": "s_xyz"},
            None,
            {},
            1,
        )
        assert "s_xyz" in msg

    def test_non_kanban_working(self):
        msg = self._format(
            "active",
            {"is_kanban_worker": False},
            None,
            {},
            7,
        )
        assert "7 min" in msg

    def test_non_kanban_with_iteration(self):
        msg = self._format(
            "active",
            {"is_kanban_worker": False},
            None,
            {"iteration": "3/10"},
            7,
        )
        assert "iteration 3/10" in msg

    def test_non_kanban_with_tool(self):
        msg = self._format(
            "active",
            {"is_kanban_worker": False},
            None,
            {"current_tool": "bash"},
            7,
        )
        assert "running: bash" in msg

    def test_kanban_card_message(self):
        msg = self._format(
            "active",
            {"is_kanban_worker": True, "kanban_task_id": "t_xyz"},
            {"title": "Implement login"},
            {},
            15,
        )
        assert "15 min" in msg
        assert "Implement login" in msg or "t_xyz" in msg

    def test_kanban_blocked_no_heartbeat(self):
        msg = self._format(
            "blocked",
            {"is_kanban_worker": True, "kanban_task_id": "t_xyz"},
            {"title": "My Card"},
            {},
            5,
        )
        assert "5 min" in msg


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
