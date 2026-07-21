"""Tests for Telegram todo checklist rendering in gateway progress messages.

When the todo tool completes during a Telegram session, the gateway should
render a compact editable checklist in the progress bubble. Subsequent todo
calls edit the same message in-place.
"""

import json


def _make_todo_result(todos):
    """Build the JSON string that the todo tool returns for a given item list."""
    pending = sum(1 for i in todos if i.get("status", "pending") == "pending")
    in_progress = sum(1 for i in todos if i.get("status") == "in_progress")
    completed = sum(1 for i in todos if i.get("status") == "completed")
    cancelled = sum(1 for i in todos if i.get("status") == "cancelled")
    return json.dumps({
        "todos": todos,
        "summary": {
            "total": len(todos),
            "pending": pending,
            "in_progress": in_progress,
            "completed": completed,
            "cancelled": cancelled,
        },
    }, ensure_ascii=False)


class TestRenderTodoChecklist:
    """Unit tests for the _render_todo_checklist helper in gateway/run.py."""

    def _render(self, result_str):
        from gateway.run import _render_todo_checklist
        return _render_todo_checklist(result_str)

    def test_renders_checklist_from_todo_result(self):
        """Should render a full checklist with all four status types."""
        result = _make_todo_result([
            {"id": "1", "content": "Set up CI/CD", "status": "pending"},
            {"id": "2", "content": "Write tests", "status": "in_progress"},
            {"id": "3", "content": "Deploy", "status": "completed"},
            {"id": "4", "content": "Cancel this", "status": "cancelled"},
        ])
        checklist = self._render(result)
        assert "📋 Task List" in checklist
        assert "[ ] Set up CI/CD" in checklist
        assert "[>] Write tests" in checklist
        assert "[x] Deploy" in checklist
        assert "[-] Cancel this" in checklist

    def test_empty_todos_returns_empty_string(self):
        """Empty todo list should return empty string (not crash)."""
        result = _make_todo_result([])
        assert self._render(result) == ""

    def test_invalid_json_returns_empty_string(self):
        """Invalid JSON result should return empty string (fail closed)."""
        assert self._render("not valid json") == ""
        assert self._render("") == ""
        assert self._render(None) == ""

    def test_truncates_long_content(self):
        """Items longer than 80 chars should be truncated."""
        long_content = "A" * 200
        result = _make_todo_result([
            {"id": "1", "content": long_content, "status": "pending"},
        ])
        checklist = self._render(result)
        assert len(long_content) > 80
        assert "A" * 80 in checklist
        assert "A" * 81 not in checklist

    def test_default_status_is_pending(self):
        """Items without a status should render as pending."""
        result = _make_todo_result([
            {"id": "1", "content": "No status", "status": ""},
        ])
        checklist = self._render(result)
        assert "[ ] No status" in checklist