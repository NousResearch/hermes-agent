"""Tests for S2 middlewares: DanglingToolCall + LoopDetection.

Spec: ~/wiki/concepts/hermes-openclaw-deerflow-integration-plan.md §S2
These run on top of the S3 middleware chain framework.
"""

from __future__ import annotations

import pytest

from agent_bus.middleware import (
    MiddlewareChain,
    MiddlewareContext,
    clear_registry,
)


@pytest.fixture(autouse=True)
def reset_registry():
    clear_registry()
    yield
    clear_registry()


# -------- DanglingToolCall --------
class TestDanglingToolCall:
    def _import(self):
        from agent_bus.middlewares import register_defaults
        register_defaults()

    def test_happy_path_no_injection(self):
        self._import()
        chain = MiddlewareChain.build()
        ctx = MiddlewareContext(messages=[
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "name": "wiki_search"}]},
            {"role": "tool", "tool_call_id": "c1", "content": "result"},
            {"role": "assistant", "content": "done"},
        ])
        ctx = chain.run("before_model", ctx)
        # No orphan → messages unchanged, no decision recorded
        assert len(ctx.messages) == 4
        assert not any(d["action"] == "injected-placeholders" for d in ctx.decisions)

    def test_orphan_tool_call_gets_placeholder(self):
        self._import()
        chain = MiddlewareChain.build()
        ctx = MiddlewareContext(messages=[
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "name": "wiki_search"},
                {"id": "c2", "name": "bash"},
            ]},
            {"role": "tool", "tool_call_id": "c1", "content": "result"},
            # c2 missing! (user interrupted before tool ran)
            {"role": "user", "content": "stop"},
        ])
        ctx = chain.run("before_model", ctx)
        # A new tool message should exist for c2 (order-independent check)
        tool_msgs = [m for m in ctx.messages if m.get("role") == "tool"]
        assert len(tool_msgs) == 2
        placeholders = [m for m in tool_msgs if m.get("_placeholder")]
        assert len(placeholders) == 1
        assert placeholders[0]["tool_call_id"] == "c2"
        assert "c2" in placeholders[0]["content"]
        # Decision recorded
        assert any(d["action"] == "injected-placeholders" for d in ctx.decisions)

    def test_multiple_orphans_fixed(self):
        self._import()
        chain = MiddlewareChain.build()
        ctx = MiddlewareContext(messages=[
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "a", "name": "t1"},
                {"id": "b", "name": "t2"},
                {"id": "c", "name": "t3"},
            ]},
        ])
        ctx = chain.run("before_model", ctx)
        tool_msgs = [m for m in ctx.messages if m.get("role") == "tool"]
        assert {m["tool_call_id"] for m in tool_msgs} == {"a", "b", "c"}

    def test_assistant_without_tool_calls_untouched(self):
        self._import()
        chain = MiddlewareChain.build()
        original = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        ctx = MiddlewareContext(messages=list(original))
        ctx = chain.run("before_model", ctx)
        assert ctx.messages == original


# -------- LoopDetection --------
class TestLoopDetection:
    def _import(self):
        from agent_bus.middlewares import register_defaults
        register_defaults()

    def _make_ai_tool_call(self, call_id: str, name: str, args: dict):
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": call_id, "name": name, "args": args}],
        }

    def test_no_loop_when_different_args(self):
        self._import()
        chain = MiddlewareChain.build()
        ctx = MiddlewareContext(messages=[
            self._make_ai_tool_call("c1", "wiki_search", {"q": "a"}),
            self._make_ai_tool_call("c2", "wiki_search", {"q": "b"}),
            self._make_ai_tool_call("c3", "wiki_search", {"q": "c"}),
        ])
        ctx = chain.run("after_model", ctx)
        assert not any(d["action"] == "loop-stopped" for d in ctx.decisions)
        assert ctx.messages[-1]["tool_calls"]  # not cleared

    def test_loop_detected_clears_tool_calls(self):
        self._import()
        chain = MiddlewareChain.build()
        same_args = {"q": "same"}
        ctx = MiddlewareContext(messages=[
            self._make_ai_tool_call("c1", "wiki_search", same_args),
            self._make_ai_tool_call("c2", "wiki_search", same_args),
            self._make_ai_tool_call("c3", "wiki_search", same_args),
        ])
        ctx = chain.run("after_model", ctx)
        # Loop tripped
        assert any(d["action"] == "loop-stopped" for d in ctx.decisions)
        # Latest AIMessage tool_calls cleared
        assert ctx.messages[-1]["tool_calls"] == []
        # Reminder text injected
        assert "Stopping tool loop" in ctx.messages[-1]["content"]
        assert ctx.messages[-1]["_loop_stopped"] is True

    def test_threshold_respected(self):
        self._import()
        chain = MiddlewareChain.build()
        same_args = {"q": "same"}
        # Only 2 repetitions — below default threshold of 3
        ctx = MiddlewareContext(messages=[
            {"role": "assistant", "content": "early text"},  # break
            self._make_ai_tool_call("c1", "wiki_search", same_args),
            self._make_ai_tool_call("c2", "wiki_search", same_args),
        ])
        ctx = chain.run("after_model", ctx)
        assert not any(d["action"] == "loop-stopped" for d in ctx.decisions)

    def test_custom_threshold_via_metadata(self):
        self._import()
        chain = MiddlewareChain.build()
        same_args = {"q": "same"}
        ctx = MiddlewareContext(
            messages=[
                self._make_ai_tool_call("c1", "wiki_search", same_args),
                self._make_ai_tool_call("c2", "wiki_search", same_args),
            ],
            metadata={"loop_threshold": 2},
        )
        ctx = chain.run("after_model", ctx)
        assert any(d["action"] == "loop-stopped" for d in ctx.decisions)

    def test_latest_no_tool_calls_no_trip(self):
        self._import()
        chain = MiddlewareChain.build()
        ctx = MiddlewareContext(messages=[
            self._make_ai_tool_call("c1", "wiki_search", {"q": "same"}),
            self._make_ai_tool_call("c2", "wiki_search", {"q": "same"}),
            self._make_ai_tool_call("c3", "wiki_search", {"q": "same"}),
            {"role": "assistant", "content": "ok done"},  # latest = text, no tool_calls
        ])
        ctx = chain.run("after_model", ctx)
        assert not any(d["action"] == "loop-stopped" for d in ctx.decisions)


# -------- Integration: both middlewares together --------
class TestBothMiddlewares:
    def test_order_dangling_before_loop(self):
        """before_model (dangling) precedes after_model (loop) in lifecycle.

        Both middlewares can coexist and each fires at its own hook without
        interfering.
        """
        from agent_bus.middlewares import register_defaults
        register_defaults()

        same_args = {"q": "x"}
        chain = MiddlewareChain.build()
        ctx = MiddlewareContext(messages=[
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "orphan", "name": "t1"},
                {"id": "c1", "name": "wiki_search", "args": same_args},
            ]},
            # only c1 gets a response
            {"role": "tool", "tool_call_id": "c1", "content": "r1"},
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": "c2", "name": "wiki_search", "args": same_args}]},
            {"role": "tool", "tool_call_id": "c2", "content": "r2"},
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": "c3", "name": "wiki_search", "args": same_args}]},
        ])

        # before_model: dangling injects placeholder for "orphan"
        ctx = chain.run("before_model", ctx)
        assert any(d["middleware"] == "dangling-tool-call" for d in ctx.decisions)

        # after_model: loop detect on the 3rd wiki_search w/ same args
        ctx = chain.run("after_model", ctx)
        assert any(d["middleware"] == "loop-detection" and d["action"] == "loop-stopped"
                   for d in ctx.decisions)
