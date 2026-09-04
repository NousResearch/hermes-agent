"""Tests for todo-checklist rendering and dispatch in gateway/run_turn_runner.py."""

import json
import logging
import queue
from unittest.mock import MagicMock

import pytest

from gateway.run_turn_runner import _dispatch_todo_progress, _render_todo_checklist


# ── _render_todo_checklist ─────────────────────────────────────────────────────


class TestRenderTodoChecklist:
    """Unit tests for the checklist rendering function."""

    def test_all_statuses(self):
        """All four statuses render distinct glyphs."""
        data = {"todos": [
            {"content": "Write tests", "status": "pending", "id": "t1"},
            {"content": "Run tests",   "status": "in_progress", "id": "t2"},
            {"content": "Debug CI",    "status": "completed", "id": "t3"},
            {"content": "Drop task",   "status": "cancelled", "id": "t4"},
        ]}
        result = _render_todo_checklist(json.dumps(data))
        assert "[ ] Write tests (t1)" in result
        assert "[>] Run tests (t2)" in result
        assert "[x] Debug CI (t3)" in result
        assert "[-] Drop task (t4)" in result
        assert result.startswith("📋 Task List")

    def test_long_content_truncated(self):
        """Content exceeding 80 chars is truncated."""
        long = "A" * 120
        data = {"todos": [{"content": long, "status": "pending", "id": "long1"}]}
        result = _render_todo_checklist(json.dumps(data))
        assert "A" * 80 in result
        assert "A" * 81 not in result

    def test_empty_input(self):
        """Empty string returns empty string."""
        assert _render_todo_checklist("") == ""

    def test_malformed_json(self):
        """Invalid JSON returns empty string."""
        assert _render_todo_checklist("not json") == ""

    def test_no_todos_key(self):
        """Dict without 'todos' key returns empty string."""
        assert _render_todo_checklist(json.dumps({"other": "data"})) == ""

    def test_empty_todos_list(self):
        """Empty list of todos returns empty string."""
        assert _render_todo_checklist(json.dumps({"todos": []})) == ""

    def test_none_input(self):
        """None input returns empty string."""
        assert _render_todo_checklist(None) == ""

    def test_dict_input_direct(self):
        """Direct dict input (not JSON string) is handled."""
        data = {"todos": [{"content": "Direct dict", "status": "completed", "id": "d1"}]}
        result = _render_todo_checklist(data)
        assert "[x] Direct dict (d1)" in result

    def test_missing_item_id(self):
        """Item without id renders without the id suffix."""
        data = {"todos": [{"content": "No ID item", "status": "pending"}]}
        result = _render_todo_checklist(json.dumps(data))
        assert "[ ] No ID item" in result
        assert "()" not in result

    def test_unknown_status(self):
        """Unknown status falls back to the default [ ] glyph."""
        data = {"todos": [{"content": "Odd status", "status": "unknown_xyz", "id": "u1"}]}
        result = _render_todo_checklist(json.dumps(data))
        assert "[ ] Odd status (u1)" in result


# ── _dispatch_todo_progress ────────────────────────────────────────────────────


class TestDispatchTodoProgress:
    """Tests for the progress-callback dispatch helper."""

    def test_todo_completed_dispatched(self):
        """todo + tool.completed pushes checklist onto progress_queue."""
        q = queue.Queue()
        data = json.dumps({"todos": [{"content": "Hello", "status": "completed", "id": "t1"}]})
        consumed = _dispatch_todo_progress(q, "tool.completed", "todo", data, logging.getLogger())
        assert consumed is True
        assert not q.empty()
        assert "Hello" in q.get()

    def test_todo_list_completed_dispatched(self):
        """todo_list alias also dispatches."""
        q = queue.Queue()
        data = json.dumps({"todos": [{"content": "Via alias", "status": "pending", "id": "t2"}]})
        consumed = _dispatch_todo_progress(q, "tool.completed", "todo_list", data, logging.getLogger())
        assert consumed is True
        assert not q.empty()

    def test_non_todo_tool_fallthrough(self):
        """Non-todo tool name returns False."""
        q = queue.Queue()
        consumed = _dispatch_todo_progress(q, "tool.completed", "web_search", "", logging.getLogger())
        assert consumed is False
        assert q.empty()

    def test_non_completed_event_fallthrough(self):
        """Non-completed event type returns False."""
        q = queue.Queue()
        consumed = _dispatch_todo_progress(q, "tool.started", "todo", "", logging.getLogger())
        assert consumed is False
        assert q.empty()

    def test_empty_result_no_queue_push(self):
        """Empty result string does not push onto queue."""
        q = queue.Queue()
        consumed = _dispatch_todo_progress(q, "tool.completed", "todo", "", logging.getLogger())
        assert consumed is True  # still consumed (it's a todo tool completed)
        assert q.empty()

    def test_none_result_no_queue_push(self):
        """None result does not push onto queue."""
        q = queue.Queue()
        consumed = _dispatch_todo_progress(q, "tool.completed", "todo", None, logging.getLogger())
        assert consumed is True
        assert q.empty()

    def test_none_queue_does_not_crash(self):
        """None progress_queue does not raise."""
        consumed = _dispatch_todo_progress(None, "tool.completed", "todo", "{}", logging.getLogger())
        assert consumed is True

    def test_logger_exception_handling(self):
        """Broken data in result doesn't propagate — logger catches it."""
        q = queue.Queue()
        # This string will crash json.loads inside _render_todo_checklist
        bad_data = '{"todos": [{"content": \x00binary"}]}'  # invalid escape
        consumed = _dispatch_todo_progress(q, "tool.completed", "todo", bad_data, logging.getLogger())
        assert consumed is True
        assert q.empty()