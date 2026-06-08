"""Tests for observability/langfuse usage accumulation (#42306).

When post_api_request fires per-API-call with usage data, the values must
be accumulated in TraceState so that a subsequent post_llm_call (which
does not receive response/usage from turn_finalizer.py) can still attach
usage to the generation span and root trace.
"""
from __future__ import annotations

import importlib
import sys

import pytest


def _fresh_mod():
    sys.modules.pop("plugins.observability.langfuse", None)
    return importlib.import_module("plugins.observability.langfuse")


class _FakeClient:
    """Minimal Langfuse client stand-in."""
    def flush(self):
        pass


class TestUsageAccumulation:
    """post_api_request usage must accumulate in TraceState."""

    def test_usage_accumulated_from_post_api_request(self, monkeypatch):
        mod = _fresh_mod()
        monkeypatch.setattr(mod, "_get_langfuse", lambda: _FakeClient())

        gen = object()
        state = mod.TraceState(trace_id="t", root_ctx=None, root_span=None)
        state.generations["0"] = gen

        task_key = mod._trace_key("task-1", "sess-1")
        monkeypatch.setitem(mod._TRACE_STATE, task_key, state)

        ended = {}

        def fake_end(obs, *, output=None, usage_details=None, cost_details=None, metadata=None):
            ended["usage"] = usage_details
            ended["cost"] = cost_details

        monkeypatch.setattr(mod, "_end_observation", fake_end)

        # Simulate post_api_request firing with usage dict
        mod.on_post_llm_call(
            task_id="task-1",
            session_id="sess-1",
            api_call_count=0,
            usage={"input_tokens": 1000, "output_tokens": 500},
            assistant_content_chars=100,
            assistant_tool_call_count=0,
        )

        assert ended["usage"]["input"] == 1000
        assert ended["usage"]["output"] == 500
        # Accumulated in state
        assert state.accumulated_usage_details["input"] == 1000
        assert state.accumulated_usage_details["output"] == 500

    def test_usage_accumulates_across_multiple_api_calls(self, monkeypatch):
        mod = _fresh_mod()
        monkeypatch.setattr(mod, "_get_langfuse", lambda: _FakeClient())

        state = mod.TraceState(trace_id="t", root_ctx=None, root_span=None)
        # Use a no-op root_span to prevent _finish_trace from crashing
        state.root_span = type("S", (), {
            "update": lambda self, **kw: None,
            "end": lambda self: None,
            "set_trace_io": lambda self, **kw: None,
        })()

        task_key = mod._trace_key("task-1", "sess-1")
        monkeypatch.setitem(mod._TRACE_STATE, task_key, state)

        ended_calls = []

        def fake_end(obs, *, output=None, usage_details=None, cost_details=None, metadata=None):
            ended_calls.append({"usage": usage_details, "cost": cost_details})

        monkeypatch.setattr(mod, "_end_observation", fake_end)

        # First API call — has tool calls so _finish_trace is NOT called
        state.generations["0"] = object()
        mod.on_post_llm_call(
            task_id="task-1", session_id="sess-1", api_call_count=0,
            usage={"input_tokens": 1000, "output_tokens": 500},
            assistant_content_chars=100, assistant_tool_call_count=2,
        )

        # Re-inject state since _finish_trace may have popped it
        # (it won't because has_tools=True, but be safe)
        monkeypatch.setitem(mod._TRACE_STATE, task_key, state)

        # Second API call — also has tool calls
        state.generations["1"] = object()
        mod.on_post_llm_call(
            task_id="task-1", session_id="sess-1", api_call_count=1,
            usage={"input_tokens": 2000, "output_tokens": 800},
            assistant_content_chars=200, assistant_tool_call_count=1,
        )

        # Accumulated totals
        assert state.accumulated_usage_details["input"] == 3000
        assert state.accumulated_usage_details["output"] == 1300

    def test_fallback_uses_accumulated_usage(self, monkeypatch):
        """When post_llm_call fires without response/usage, accumulated
        values from earlier post_api_request calls must be used."""
        mod = _fresh_mod()
        monkeypatch.setattr(mod, "_get_langfuse", lambda: _FakeClient())

        gen = object()
        state = mod.TraceState(trace_id="t", root_ctx=None, root_span=None)
        state.generations["0"] = gen
        # Pre-populate accumulated usage (as if post_api_request already ran)
        state.accumulated_usage_details = {"input": 1500, "output": 700}
        state.accumulated_cost_details = {"input": 0.003, "output": 0.007}

        task_key = mod._trace_key("task-1", "sess-1")
        monkeypatch.setitem(mod._TRACE_STATE, task_key, state)

        ended = {}

        def fake_end(obs, *, output=None, usage_details=None, cost_details=None, metadata=None):
            ended["usage"] = usage_details
            ended["cost"] = cost_details

        monkeypatch.setattr(mod, "_end_observation", fake_end)

        # Simulate post_llm_call from turn_finalizer.py (no response, no usage)
        mod.on_post_llm_call(
            task_id="task-1",
            session_id="sess-1",
            assistant_response="Hello world",
        )

        # Should use accumulated values as fallback
        assert ended["usage"]["input"] == 1500
        assert ended["usage"]["output"] == 700
        assert ended["cost"]["input"] == 0.003
        assert ended["cost"]["output"] == 0.007

    def test_fallback_empty_when_no_prior_accumulation(self, monkeypatch):
        """When post_llm_call fires without response/usage AND no prior
        post_api_request ran, usage should be empty (no crash)."""
        mod = _fresh_mod()
        monkeypatch.setattr(mod, "_get_langfuse", lambda: _FakeClient())

        gen = object()
        state = mod.TraceState(trace_id="t", root_ctx=None, root_span=None)
        state.generations["0"] = gen

        task_key = mod._trace_key("task-1", "sess-1")
        monkeypatch.setitem(mod._TRACE_STATE, task_key, state)

        ended = {}

        def fake_end(obs, *, output=None, usage_details=None, cost_details=None, metadata=None):
            ended["usage"] = usage_details
            ended["cost"] = cost_details

        monkeypatch.setattr(mod, "_end_observation", fake_end)

        mod.on_post_llm_call(
            task_id="task-1",
            session_id="sess-1",
            assistant_response="Hello world",
        )

        # No prior accumulation — should be empty, not crash
        assert ended["usage"] == {}
        assert ended["cost"] == {}


class TestFinishTraceUsageAttachment:
    """_finish_trace must attach accumulated usage to the root span."""

    def test_finish_trace_attaches_usage_to_root_span(self, monkeypatch):
        mod = _fresh_mod()

        root_span = type("FakeSpan", (), {
            "update": lambda self, **kw: setattr(self, "_update_kwargs", kw),
            "end": lambda self: None,
            "set_trace_io": lambda self, **kw: None,
        })()

        state = mod.TraceState(trace_id="t", root_ctx=None, root_span=root_span)
        state.accumulated_usage_details = {"input": 5000, "output": 2000}
        state.accumulated_cost_details = {"input": 0.01, "output": 0.02}

        task_key = mod._trace_key("task-1", "sess-1")
        monkeypatch.setitem(mod._TRACE_STATE, task_key, state)

        monkeypatch.setattr(mod, "_get_langfuse", lambda: _FakeClient())

        mod._finish_trace(task_key, output="done")

        # Verify usage was attached to root span
        assert hasattr(root_span, "_update_kwargs")
        kwargs = root_span._update_kwargs
        assert kwargs["usage_details"]["input"] == 5000
        assert kwargs["usage_details"]["output"] == 2000
        assert kwargs["cost_details"]["input"] == 0.01
        assert kwargs["cost_details"]["output"] == 0.02

    def test_finish_trace_no_usage_when_accumulated_empty(self, monkeypatch):
        mod = _fresh_mod()

        update_calls = []
        root_span = type("FakeSpan", (), {
            "update": lambda self, **kw: update_calls.append(kw),
            "end": lambda self: None,
            "set_trace_io": lambda self, **kw: None,
        })()

        state = mod.TraceState(trace_id="t", root_ctx=None, root_span=root_span)
        # No accumulated usage

        task_key = mod._trace_key("task-1", "sess-1")
        monkeypatch.setitem(mod._TRACE_STATE, task_key, state)

        monkeypatch.setattr(mod, "_get_langfuse", lambda: _FakeClient())

        mod._finish_trace(task_key, output="done")

        # Only one update call (for output), no usage_details
        assert len(update_calls) == 1
        assert "usage_details" not in update_calls[0]
