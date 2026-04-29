"""Tests for ScopeGuard — runaway loop detection and progress checkpoints."""

import pytest

from agent.scope_guard import ScopeGuard, RUNAWAY_THRESHOLD, CHECKPOINT_INTERVAL


class TestScopeGuardRunawayDetection:
    """Test runaway loop detection."""

    def test_no_warning_on_different_calls(self):
        guard = ScopeGuard(runaway_threshold=3)
        r1 = guard.record_tool_call("terminal", {"command": "ls"}, True)
        r2 = guard.record_tool_call("terminal", {"command": "pwd"}, True)
        assert r1 is None
        assert r2 is None

    def test_no_warning_on_success(self):
        guard = ScopeGuard(runaway_threshold=2)
        guard.record_tool_call("terminal", {"command": "ls"}, False)
        guard.record_tool_call("terminal", {"command": "ls"}, False)
        # Success calls should not trigger runaway
        assert guard._consecutive.get("terminal", 0) == 0

    def test_runaway_warning_at_threshold(self):
        guard = ScopeGuard(runaway_threshold=3)
        guard.record_tool_call("terminal", {"command": "bad"}, True)
        guard.record_tool_call("terminal", {"command": "bad"}, True)
        hint = guard.record_tool_call("terminal", {"command": "bad"}, True)
        assert hint is not None
        assert "SCOPE GUARD" in hint
        assert "runaway" in hint.lower()
        assert "terminal" in hint

    def test_runaway_resets_after_warning(self):
        guard = ScopeGuard(runaway_threshold=2)
        guard.record_tool_call("terminal", {"command": "bad"}, True)
        hint1 = guard.record_tool_call("terminal", {"command": "bad"}, True)
        assert hint1 is not None
        # Should not warn again immediately (counter reset)
        hint2 = guard.record_tool_call("terminal", {"command": "bad"}, True)
        # After reset, need threshold more calls
        assert hint2 is None

    def test_different_tools_tracked_independently(self):
        guard = ScopeGuard(runaway_threshold=2)
        guard.record_tool_call("terminal", {"command": "bad"}, True)
        guard.record_tool_call("execute_code", {"code": "bad"}, True)
        # Each tool only failed once
        hint_t = guard.record_tool_call("terminal", {"command": "bad"}, True)
        hint_e = guard.record_tool_call("execute_code", {"code": "bad"}, True)
        assert hint_t is not None
        assert hint_e is not None

    def test_different_args_reset_consecutive(self):
        guard = ScopeGuard(runaway_threshold=3)
        guard.record_tool_call("terminal", {"command": "ls"}, True)
        guard.record_tool_call("terminal", {"command": "pwd"}, True)
        guard.record_tool_call("terminal", {"command": "ls"}, True)
        # Different args in between reset the consecutive count
        assert guard._consecutive["terminal"] == 1


class TestScopeGuardCheckpoints:
    """Test progress checkpoint injection."""

    def test_checkpoint_at_interval(self):
        guard = ScopeGuard(checkpoint_interval=5)
        # Run 4 calls - no checkpoint
        for i in range(4):
            hint = guard.record_tool_call(f"tool_{i}", {"i": i}, False)
            assert hint is None
        # 5th call should trigger checkpoint
        hint = guard.record_tool_call("tool_5", {"i": 5}, False)
        assert hint is not None
        assert "PROGRESS CHECKPOINT" in hint
        assert "iteration 5" in hint

    def test_checkpoint_respects_max(self):
        guard = ScopeGuard(checkpoint_interval=2, runaway_threshold=100)
        # Each checkpoint triggers at every 2nd iteration
        checkpoints = 0
        for i in range(20):
            hint = guard.record_tool_call("tool", {"i": i}, False)
            if hint and "PROGRESS CHECKPOINT" in hint:
                checkpoints += 1
        # Should be capped at MAX_CHECKPOINTS (4)
        assert checkpoints <= 4

    def test_checkpoint_mentions_iterations(self):
        guard = ScopeGuard(checkpoint_interval=10)
        for i in range(9):
            guard.record_tool_call("tool", {"i": i}, False)
        hint = guard.record_tool_call("tool", {"i": 10}, False)
        assert hint is not None
        assert "10" in hint


class TestScopeGuardStats:
    """Test statistics tracking."""

    def test_stats_tracking(self):
        guard = ScopeGuard()
        guard.record_tool_call("terminal", {"command": "ls"}, False)
        guard.record_tool_call("terminal", {"command": "pwd"}, True)
        stats = guard.get_stats()
        assert stats["iteration_count"] == 2
        assert stats["checkpoints_emitted"] >= 0

    def test_reset(self):
        guard = ScopeGuard()
        guard.record_tool_call("terminal", {"command": "ls"}, True)
        guard.reset()
        stats = guard.get_stats()
        assert stats["iteration_count"] == 0
        assert stats["consecutive_failures"] == {}
