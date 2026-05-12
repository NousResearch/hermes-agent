"""Tests for built-in AgentLoop middleware implementations.

RED phase: define the contract for concrete middleware that extracts
concerns from the 3128-line while-loop body in run_conversation().

Phase 3 middleware targets:
  - SteerDrainMiddleware: pre-API-call /steer injection
  - StepCallbackMiddleware: gateway step event emission
  - SkillNudgeMiddleware: skill nudge counter tracking
"""

import pytest
from unittest.mock import MagicMock, patch
from agent.loop import AgentLoop, LoopContext, MiddlewareBase


# ---------------------------------------------------------------------------
# SteerDrainMiddleware
# ---------------------------------------------------------------------------

class TestSteerDrainMiddleware:
    def test_import(self):
        from agent.middleware import SteerDrainMiddleware
        assert issubclass(SteerDrainMiddleware, MiddlewareBase)

    def test_before_iteration_calls_drain(self):
        """before_iteration should call the agent's _drain_pending_steer."""
        from agent.middleware import SteerDrainMiddleware

        agent = MagicMock()
        agent._drain_pending_steer.return_value = "user guidance text"

        mw = SteerDrainMiddleware(agent)
        ctx = LoopContext(max_iterations=10)
        mw.before_iteration(ctx, 1, messages=[])

        agent._drain_pending_steer.assert_called_once()

    def test_before_iteration_injects_into_last_tool_msg(self):
        """Steer text should be appended to the last tool-role message."""
        from agent.middleware import SteerDrainMiddleware

        agent = MagicMock()
        agent._drain_pending_steer.return_value = "steer text"

        mw = SteerDrainMiddleware(agent)
        ctx = LoopContext(max_iterations=10)
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "tool_calls": [{"id": "1", "function": {"name": "test"}}]},
            {"role": "tool", "tool_call_id": "1", "content": "result"},
        ]
        mw.before_iteration(ctx, 1, messages=messages)

        assert "steer text" in messages[2]["content"]

    def test_no_steer_text_no_mutation(self):
        """When drain returns empty/None, messages should not be touched."""
        from agent.middleware import SteerDrainMiddleware

        agent = MagicMock()
        agent._drain_pending_steer.return_value = None

        mw = SteerDrainMiddleware(agent)
        ctx = LoopContext(max_iterations=10)
        messages = [{"role": "tool", "content": "original"}]
        mw.before_iteration(ctx, 1, messages=messages)

        assert messages[0]["content"] == "original"


# ---------------------------------------------------------------------------
# StepCallbackMiddleware
# ---------------------------------------------------------------------------

class TestStepCallbackMiddleware:
    def test_import(self):
        from agent.middleware import StepCallbackMiddleware
        assert issubclass(StepCallbackMiddleware, MiddlewareBase)

    def test_before_iteration_fires_callback(self):
        """before_iteration should invoke the step_callback with iteration count."""
        from agent.middleware import StepCallbackMiddleware

        callback = MagicMock()
        mw = StepCallbackMiddleware(callback)
        ctx = LoopContext(max_iterations=10)

        messages = [{"role": "assistant", "tool_calls": [
            {"id": "1", "function": {"name": "read_file", "arguments": "{}"}}
        ]}]
        tool_msgs = [{"role": "tool", "tool_call_id": "1", "content": "data"}]
        mw.before_iteration(ctx, 3, messages=messages + tool_msgs)

        callback.assert_called_once()
        args = callback.call_args[0]
        assert args[0] == 3  # iteration number

    def test_no_callback_no_error(self):
        """When callback is None, before_iteration should be a no-op."""
        from agent.middleware import StepCallbackMiddleware

        mw = StepCallbackMiddleware(None)
        ctx = LoopContext(max_iterations=10)
        # Should not raise
        mw.before_iteration(ctx, 1, messages=[])


# ---------------------------------------------------------------------------
# SkillNudgeMiddleware
# ---------------------------------------------------------------------------

class TestSkillNudgeMiddleware:
    def test_import(self):
        from agent.middleware import SkillNudgeMiddleware
        assert issubclass(SkillNudgeMiddleware, MiddlewareBase)

    def test_counter_increments_on_before_iteration(self):
        """Counter should increment each iteration when skill_manage is valid."""
        from agent.middleware import SkillNudgeMiddleware

        mw = SkillNudgeMiddleware(nudge_interval=5, has_skill_manage=True)
        ctx = LoopContext(max_iterations=10)

        mw.before_iteration(ctx, 1)
        mw.before_iteration(ctx, 2)
        mw.before_iteration(ctx, 3)

        assert mw.iters_since_skill == 3

    def test_counter_resets(self):
        """reset() should set counter back to 0."""
        from agent.middleware import SkillNudgeMiddleware

        mw = SkillNudgeMiddleware(nudge_interval=5, has_skill_manage=True)
        ctx = LoopContext(max_iterations=10)

        mw.before_iteration(ctx, 1)
        mw.before_iteration(ctx, 2)
        mw.reset()
        assert mw.iters_since_skill == 0

    def test_no_increment_when_skill_manage_absent(self):
        """When has_skill_manage is False, counter stays at 0."""
        from agent.middleware import SkillNudgeMiddleware

        mw = SkillNudgeMiddleware(nudge_interval=5, has_skill_manage=False)
        ctx = LoopContext(max_iterations=10)

        mw.before_iteration(ctx, 1)
        mw.before_iteration(ctx, 2)

        assert mw.iters_since_skill == 0

    def test_no_increment_when_interval_zero(self):
        """When nudge_interval is 0, counter stays at 0."""
        from agent.middleware import SkillNudgeMiddleware

        mw = SkillNudgeMiddleware(nudge_interval=0, has_skill_manage=True)
        ctx = LoopContext(max_iterations=10)

        mw.before_iteration(ctx, 1)
        assert mw.iters_since_skill == 0
