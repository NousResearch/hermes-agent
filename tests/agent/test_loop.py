"""Tests for agent/loop.py — extracted AgentLoop class.

RED phase: These tests define the contract for the AgentLoop that will
eventually replace the 3770-line while-loop inside run_conversation().

Phase 2 strategy: extract the interface and skeleton with middleware
hook points, not a full rewrite. The loop delegates back to AIAgent for
the actual API call and tool dispatch (for now), creating a clean seam
for future middleware without a risky big-bang migration.

SOLID:
  S — AgentLoop owns iteration control, budget, interrupts
  O — extensible via middleware hooks without modifying the loop
  D — depends on abstract LoopContext, not AIAgent directly
"""

import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# AgentLoop creation
# ---------------------------------------------------------------------------

class TestAgentLoopCreation:
    def test_create_with_defaults(self):
        from agent.loop import AgentLoop, LoopContext
        ctx = LoopContext(max_iterations=10)
        loop = AgentLoop(ctx)
        assert loop.context.max_iterations == 10
        assert loop.iteration == 0
        assert loop.interrupted is False

    def test_create_with_middleware(self):
        from agent.loop import AgentLoop, LoopContext, MiddlewareBase

        class LoggingMiddleware(MiddlewareBase):
            def __init__(self):
                self.calls = []
            def before_iteration(self, ctx, iteration, **kw):
                self.calls.append(("before", iteration))
            def after_iteration(self, ctx, iteration, result):
                self.calls.append(("after", iteration))

        ctx = LoopContext(max_iterations=5)
        mw = LoggingMiddleware()
        loop = AgentLoop(ctx, middlewares=[mw])
        assert len(loop.middlewares) == 1


# ---------------------------------------------------------------------------
# LoopContext
# ---------------------------------------------------------------------------

class TestLoopContext:
    def test_create_with_budget(self):
        from agent.loop import LoopContext
        ctx = LoopContext(max_iterations=30)
        assert ctx.max_iterations == 30
        assert ctx.remaining > 0

    def test_exhausted(self):
        from agent.loop import LoopContext
        ctx = LoopContext(max_iterations=2)
        assert ctx.should_continue() is True
        ctx.consume()
        assert ctx.should_continue() is True
        ctx.consume()
        assert ctx.should_continue() is False

    def test_interrupt_sets_stopped(self):
        from agent.loop import LoopContext
        ctx = LoopContext(max_iterations=100)
        ctx.request_interrupt()
        assert ctx.should_continue() is False

    def test_grace_call(self):
        from agent.loop import LoopContext
        ctx = LoopContext(max_iterations=1)
        ctx.consume()
        assert ctx.should_continue() is False
        ctx.enable_grace_call()
        assert ctx.should_continue() is True  # one more chance
        ctx.consume()
        assert ctx.should_continue() is False


# ---------------------------------------------------------------------------
# MiddlewareBase
# ---------------------------------------------------------------------------

class TestMiddlewareBase:
    def test_default_hooks_are_noops(self):
        from agent.loop import MiddlewareBase, LoopContext
        mw = MiddlewareBase()
        ctx = LoopContext(max_iterations=10)
        # All hooks should be callable without error (default no-ops)
        mw.before_iteration(ctx, 1)
        mw.after_iteration(ctx, 1, {"has_tool_calls": False})
        mw.on_interrupt(ctx, 1)
        mw.on_budget_exhausted(ctx, 1)
        mw.on_tool_call(ctx, 1, {"name": "test", "arguments": "{}"})
        mw.on_tool_result(ctx, 1, "result")

    def test_custom_middleware_receives_events(self):
        from agent.loop import AgentLoop, LoopContext, MiddlewareBase

        events = []

        class Tracker(MiddlewareBase):
            def before_iteration(self, ctx, i, **kw):
                events.append(f"before:{i}")
            def after_iteration(self, ctx, i, result):
                events.append(f"after:{i}")

        ctx = LoopContext(max_iterations=3)
        mw = Tracker()
        loop = AgentLoop(ctx, middlewares=[mw])

        # Simulate a run with a simple callable
        def fake_api_call(messages, tools):
            ctx.consume()
            return {"has_tool_calls": False, "content": "done"}

        loop.run([], [], api_call_fn=fake_api_call)
        assert "before:1" in events
        assert "after:1" in events


# ---------------------------------------------------------------------------
# AgentLoop.run — core iteration control
# ---------------------------------------------------------------------------

class TestAgentLoopRun:
    def test_single_call_no_tools(self):
        """Model returns final response immediately — loop runs once."""
        from agent.loop import AgentLoop, LoopContext

        ctx = LoopContext(max_iterations=10)
        loop = AgentLoop(ctx)

        def fake_api(messages, tools):
            ctx.consume()
            return {"has_tool_calls": False, "content": "Hello!"}

        result = loop.run([], [], api_call_fn=fake_api)
        assert result["content"] == "Hello!"
        assert loop.iteration == 1

    def test_tool_call_then_final(self):
        """Model calls a tool, then returns final response — loop runs twice."""
        from agent.loop import AgentLoop, LoopContext

        ctx = LoopContext(max_iterations=10)
        loop = AgentLoop(ctx)
        call_count = 0

        def fake_api(messages, tools):
            nonlocal call_count
            call_count += 1
            ctx.consume()
            if call_count == 1:
                return {
                    "has_tool_calls": True,
                    "tool_calls": [{"id": "tc1", "name": "read_file", "arguments": "{}"}],
                }
            return {"has_tool_calls": False, "content": "Here's the file content."}

        def fake_tool_dispatch(tool_calls):
            return [{"tool_call_id": tc["id"], "content": "file contents here"} for tc in tool_calls]

        result = loop.run([], [], api_call_fn=fake_api, tool_dispatch_fn=fake_tool_dispatch)
        assert result["content"] == "Here's the file content."
        assert loop.iteration == 2

    def test_budget_exhausted(self):
        """Loop stops when budget is exhausted."""
        from agent.loop import AgentLoop, LoopContext

        ctx = LoopContext(max_iterations=2)
        loop = AgentLoop(ctx)

        def always_tools(messages, tools):
            ctx.consume()
            return {
                "has_tool_calls": True,
                "tool_calls": [{"id": "tc", "name": "test", "arguments": "{}"}],
            }

        def noop_dispatch(tool_calls):
            return [{"tool_call_id": tc["id"], "content": "ok"} for tc in tool_calls]

        result = loop.run([], [], api_call_fn=always_tools, tool_dispatch_fn=noop_dispatch)
        assert loop.iteration == 2
        assert result.get("budget_exhausted") is True

    def test_interrupt_stops_loop(self):
        """Loop stops when interrupt is requested."""
        from agent.loop import AgentLoop, LoopContext

        ctx = LoopContext(max_iterations=100)
        loop = AgentLoop(ctx)

        def api_then_interrupt(messages, tools):
            ctx.consume()
            ctx.request_interrupt()
            return {"has_tool_calls": False, "content": "partial"}

        result = loop.run([], [], api_call_fn=api_then_interrupt)
        assert loop.iteration == 1
        assert loop.interrupted is True

    def test_middleware_can_modify_messages(self):
        """Middleware can inject context before each iteration."""
        from agent.loop import AgentLoop, LoopContext, MiddlewareBase

        injected = []

        class InjectMiddleware(MiddlewareBase):
            def before_iteration(self, ctx, i, **kw):
                injected.append(f"turn-{i}")

        ctx = LoopContext(max_iterations=3)
        loop = AgentLoop(ctx, middlewares=[InjectMiddleware()])

        def tools_always(messages, tools):
            ctx.consume()
            return {
                "has_tool_calls": True,
                "tool_calls": [{"id": "tc", "name": "test", "arguments": "{}"}],
            }

        def noop_dispatch(tool_calls):
            return [{"tool_call_id": tc["id"], "content": "ok"} for tc in tool_calls]

        loop.run([], [], api_call_fn=tools_always, tool_dispatch_fn=noop_dispatch)
        assert injected == ["turn-1", "turn-2", "turn-3"]

    def test_middleware_on_budget_exhausted_event(self):
        from agent.loop import AgentLoop, LoopContext, MiddlewareBase

        exhausted_at = []

        class BudgetWatcher(MiddlewareBase):
            def on_budget_exhausted(self, ctx, i):
                exhausted_at.append(i)

        ctx = LoopContext(max_iterations=1)
        loop = AgentLoop(ctx, middlewares=[BudgetWatcher()])

        def always_tools(messages, tools):
            ctx.consume()
            return {
                "has_tool_calls": True,
                "tool_calls": [{"id": "tc", "name": "test", "arguments": "{}"}],
            }

        def noop_dispatch(tool_calls):
            return [{"tool_call_id": tc["id"], "content": "ok"} for tc in tool_calls]

        loop.run([], [], api_call_fn=always_tools, tool_dispatch_fn=noop_dispatch)
        assert len(exhausted_at) == 1
        assert exhausted_at[0] == 1
