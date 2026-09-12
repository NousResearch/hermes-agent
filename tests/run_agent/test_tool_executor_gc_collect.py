"""Regression tests for gc.collect after large tool results (#70684)."""

import gc
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import agent.tool_executor as _te
from agent.tool_executor import (
    _ManagedToolResult,
    execute_tool_calls_sequential,
)


@pytest.fixture(autouse=True)
def _isolate_hermes(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    (tmp_path / ".hermes").mkdir(exist_ok=True)


def _make_agent():
    """Minimal agent stub with the attributes the sequential executor needs."""
    agent = SimpleNamespace()
    agent._interrupt_requested = False
    agent._incremental_persistence_failed = False
    agent.quiet_mode = True
    agent.verbose_logging = False
    agent.log_prefix = ""
    agent.log_prefix_chars = 200
    agent.session_id = ""
    agent._current_turn_id = ""
    agent._current_api_request_id = ""
    agent._current_tool = None
    agent._touch_activity = lambda desc: None
    agent._vprint = lambda msg, force=False: None
    agent._safe_print = lambda msg: None
    agent._should_emit_quiet_tool_messages = lambda: False
    agent._should_start_quiet_spinner = lambda: False
    agent.tool_progress_callback = None
    agent.tool_start_callback = None
    agent.tool_complete_callback = None
    agent._subdirectory_hints = MagicMock()
    agent._subdirectory_hints.check_tool_call = lambda *a, **kw: None
    agent._append_guardrail_observation = lambda name, args, result, failed, **kwargs: result
    agent._record_file_mutation_result = lambda *a, **kw: None
    agent._tool_result_content_for_active_model = lambda name, result: result
    agent._apply_pending_steer_to_tool_results = lambda *a, **kw: None
    agent._flush_messages_to_session_db = MagicMock(return_value=True)
    agent._tool_guardrails = MagicMock()
    agent._tool_guardrails.before_call = lambda name, args: MagicMock(allows_execution=True)
    agent._context_engine_tool_names = set()
    agent._memory_manager = None
    return agent


def _make_assistant(tool_name, args="{}"):
    tc = SimpleNamespace(
        id="tc_1",
        type="function",
        function=SimpleNamespace(name=tool_name, arguments=args),
    )
    return SimpleNamespace(content="", tool_calls=[tc])


def test_maybe_collect_gc_triggers_for_large_content(monkeypatch):
    """_maybe_collect_gc_after_tool_result calls gc.collect for >=1MB content."""
    helper = getattr(_te, "_maybe_collect_gc_after_tool_result", None)
    assert helper is not None
    mock_collect = MagicMock()
    monkeypatch.setattr(gc, "collect", mock_collect)

    large = "x" * 1_100_000
    helper(large)
    assert mock_collect.call_count == 1, "gc.collect() should fire for large content"


def test_maybe_collect_gc_ignores_small_content(monkeypatch):
    """_maybe_collect_gc_after_tool_result does not call gc.collect for small content."""
    helper = getattr(_te, "_maybe_collect_gc_after_tool_result", None)
    assert helper is not None
    mock_collect = MagicMock()
    monkeypatch.setattr(gc, "collect", mock_collect)

    helper("small result")
    assert mock_collect.call_count == 0, "gc.collect() should not fire for small content"


def test_sequential_tool_gc_collect_for_persisted_large_result(monkeypatch):
    """Spillover replacement cannot hide the raw result from the GC threshold."""
    mock_collect = MagicMock()
    monkeypatch.setattr(gc, "collect", mock_collect)

    large_result = "x" * 1_100_000
    agent = _make_agent()
    assistant = _make_assistant("read_file")
    messages = []

    def _fake_middleware(*args, **kwargs):
        return _ManagedToolResult(
            result=large_result,
            args=kwargs.get("function_args", {}),
            middleware_trace=[],
            blocked=False,
            dispatched=True,
        )

    with (
        patch(
            "agent.tool_executor._run_sequential_tool_execution_middleware",
            side_effect=_fake_middleware,
        ),
        patch(
            "agent.tool_executor.maybe_persist_tool_result",
            return_value="<persisted-output>small stub</persisted-output>",
        ),
    ):
        execute_tool_calls_sequential(agent, assistant, messages, "task-1")

    assert len(messages) == 1
    assert "small stub" in messages[0]["content"]
    assert mock_collect.call_count == 1, "gc.collect() should fire after a large tool result"


def test_sequential_tool_no_gc_collect_for_small_result(monkeypatch):
    """A small tool result appended by the sequential executor does not trigger gc.collect."""
    mock_collect = MagicMock()
    monkeypatch.setattr(gc, "collect", mock_collect)

    agent = _make_agent()
    assistant = _make_assistant("read_file")
    messages = []

    def _fake_middleware(*args, **kwargs):
        return _ManagedToolResult(
            result="ok",
            args=kwargs.get("function_args", {}),
            middleware_trace=[],
            blocked=False,
            dispatched=True,
        )

    with (
        patch(
            "agent.tool_executor._run_sequential_tool_execution_middleware",
            side_effect=_fake_middleware,
        ),
        patch(
            "agent.tool_executor.maybe_persist_tool_result",
            side_effect=lambda **kwargs: kwargs["content"],
        ),
    ):
        execute_tool_calls_sequential(agent, assistant, messages, "task-1")

    assert len(messages) == 1
    assert mock_collect.call_count == 0, "gc.collect() should not fire after a small tool result"
