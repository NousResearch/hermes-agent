"""Tests for tool guardrail same-result detection with varying args (#60084)."""

import pytest
from agent.tool_guardrails import (
    ToolCallGuardrailConfig,
    ToolCallGuardrailController,
    ToolGuardrailDecision,
)


@pytest.fixture
def config():
    return ToolCallGuardrailConfig(
        warnings_enabled=True,
        no_progress_warn_after=3,
        hard_stop_enabled=False,
        idempotent_tools=["read_file", "skill_view"],
    )


@pytest.fixture
def ctl(config):
    return ToolCallGuardrailController(config)


class TestSameResultDetection:
    def test_different_args_same_result_triggers_warning(self, ctl):
        """Same result from different args should still be caught (#60084)."""
        # First call — args A, result X
        ctl.after_call("execute_code", {"code": "print(1)"}, "output: 42")
        # Second call — args B, result X (same!)
        r = ctl.after_call("execute_code", {"code": "print(2)"}, "output: 42")
        assert r.action is None  # still below threshold (warn_after=3)

        # Third call — args C, result X
        r = ctl.after_call("execute_code", {"code": "print(3)"}, "output: 42")
        assert r.action == "warn"
        assert r.code == "same_result_warning"

    def test_same_args_same_result_idempotent(self, ctl):
        """Idempotent tools: same-result detection fires first."""
        for i in range(3):
            r = ctl.after_call("read_file", {"path": "/x"}, "content")
        # Same-result detection fires before idempotent path for same
        # (tool_name, result_hash) pairs.
        assert r.action == "warn"
        assert r.code == "same_result_warning"

    def test_different_results_reset_counter(self, ctl):
        """Different results should not accumulate."""
        ctl.after_call("execute_code", {"code": "a"}, "result 1")
        ctl.after_call("execute_code", {"code": "b"}, "result 2")
        r = ctl.after_call("execute_code", {"code": "c"}, "result 3")
        assert r.action is None  # all different

    def test_mixed_same_and_different(self, ctl):
        """Counter should only count truly identical results."""
        ctl.after_call("execute_code", {"code": "a"}, "result X")
        ctl.after_call("execute_code", {"code": "b"}, "result Y")
        ctl.after_call("execute_code", {"code": "c"}, "result X")  # same as first
        r = ctl.after_call("execute_code", {"code": "d"}, "result X")  # third X
        assert r.action == "warn"
        assert r.code == "same_result_warning"

    def test_reset_for_turn_clears_same_result(self, ctl):
        """reset_for_turn should clear _same_result tracking."""
        ctl.after_call("execute_code", {"code": "a"}, "result X")
        ctl.after_call("execute_code", {"code": "b"}, "result X")
        ctl.reset_for_turn()
        r = ctl.after_call("execute_code", {"code": "c"}, "result X")
        assert r.action is None  # should be first count again

    def test_hard_stop_warns_not_blocks(self):
        """hard_stop: same-result only warns, does not block."""
        c = ToolCallGuardrailController(ToolCallGuardrailConfig(
            warnings_enabled=True,
            no_progress_warn_after=2,
            hard_stop_enabled=True,
            no_progress_block_after=4,
        ))
        for i in range(5):
            r = c.after_call("execute_code", {"code": str(i)}, "same")
        assert r.action == "warn"
        assert r.code == "same_result_warning"
        # Same-result detection does not set halt_decision (no blocking).
