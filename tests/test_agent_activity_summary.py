"""Regression tests for AIAgent activity summary observability fields."""

import threading
import time
from types import SimpleNamespace

from run_agent import AIAgent


def test_get_activity_summary_includes_active_child_metadata():
    parent = object.__new__(AIAgent)
    parent._last_activity_ts = time.time() - 3
    parent._last_activity_desc = "delegating work"
    parent._current_tool = "delegate_task"
    parent._api_call_count = 4
    parent.max_iterations = 90
    parent.iteration_budget = SimpleNamespace(used=6, max_total=120)
    parent._current_task_id = "parent-task"
    parent.session_id = "session-123"
    parent.model = "openrouter/owl-alpha"
    parent.provider = "openrouter"
    parent._active_children_lock = threading.Lock()
    parent._active_children = [
        SimpleNamespace(
            _subagent_id="sa-1",
            model="qwen/qwen3-coder:free",
            provider="openrouter",
            session_id="child-session",
            _current_task_id="child-task",
            _last_activity_ts=time.time() - 1,
            _last_activity_desc="running tests",
            _current_tool="terminal",
            _api_call_count=2,
            max_iterations=50,
            iteration_budget=SimpleNamespace(used=3, max_total=50),
        )
    ]

    summary = parent.get_activity_summary()

    assert summary["last_activity_desc"] == "delegating work"
    assert summary["current_tool"] == "delegate_task"
    assert summary["task_id"] == "parent-task"
    assert summary["session_id"] == "session-123"
    assert summary["model"] == "openrouter/owl-alpha"
    assert summary["provider"] == "openrouter"
    assert len(summary["active_children"]) == 1
    child = summary["active_children"][0]
    assert child["index"] == 1
    assert child["subagent_id"] == "sa-1"
    assert child["model"] == "qwen/qwen3-coder:free"
    assert child["provider"] == "openrouter"
    assert child["session_id"] == "child-session"
    assert child["task_id"] == "child-task"
    assert child["last_activity_desc"] == "running tests"
    assert child["current_tool"] == "terminal"
    assert child["api_call_count"] == 2
    assert child["max_iterations"] == 50
    assert child["budget_used"] == 3
    assert child["budget_max"] == 50
    assert child["seconds_since_activity"] >= 0


def test_get_activity_summary_defensively_handles_child_snapshot_errors():
    parent = object.__new__(AIAgent)
    parent._last_activity_ts = time.time()
    parent._last_activity_desc = "working"
    parent._current_tool = None
    parent._api_call_count = 0
    parent.max_iterations = 90
    parent.iteration_budget = SimpleNamespace(used=0, max_total=90)
    parent._active_children_lock = None
    parent._active_children = []

    summary = parent.get_activity_summary()

    assert summary["active_children"] == []
    assert summary["task_id"] is None
    assert summary["session_id"] is None
