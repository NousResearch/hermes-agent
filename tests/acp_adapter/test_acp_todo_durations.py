"""ACP todo result rendering shows per-item wall-clock durations.

This is the Zed/ACP surface from the screenshot — the tool-call result body
that lists each task. Durations come from elapsed_seconds in the todo result.
"""

import json

from acp_adapter.tools import _format_todo_result


def test_todo_result_renders_per_item_durations():
    result = json.dumps({
        "todos": [
            {"id": "a", "content": "build core", "status": "completed", "elapsed_seconds": 134.0},
            {"id": "b", "content": "write tests", "status": "in_progress", "elapsed_seconds": 5.0},
            {"id": "c", "content": "review", "status": "pending", "elapsed_seconds": None},
        ],
        "summary": {
            "total": 3, "completed": 1, "in_progress": 1, "pending": 1,
            "cancelled": 0, "total_elapsed_seconds": 139.0,
        },
    })
    out = _format_todo_result(result)
    assert out is not None
    assert "✅ (2m 14s) build core" in out
    assert "🔄 (5.0s) write tests" in out
    # Pending (unmeasured) task gets no duration paren.
    review_line = next(line for line in out.splitlines() if "review" in line)
    assert "(" not in review_line
    # Total tracked time appended to the progress summary.
    assert "tracked" in out


def test_todo_result_without_timing_is_backward_compatible():
    # Old-shape result (no elapsed_seconds / total_elapsed_seconds) renders the
    # same list it always did, just with no duration parens.
    result = json.dumps({
        "todos": [
            {"id": "a", "content": "do thing", "status": "completed"},
            {"id": "b", "content": "next", "status": "pending"},
        ],
        "summary": {"total": 2, "completed": 1, "in_progress": 0, "pending": 1, "cancelled": 0},
    })
    out = _format_todo_result(result)
    assert out is not None
    assert "✅ do thing" in out
    assert "⏳ next" in out
    assert "(" not in out  # no durations anywhere
    assert "tracked" not in out


def test_todo_result_invalid_returns_none():
    assert _format_todo_result("not json") is None
    assert _format_todo_result(json.dumps({"nope": 1})) is None
