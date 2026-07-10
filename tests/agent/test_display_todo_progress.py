"""Tests for get_cute_tool_message todo progress display.

Verifies the completion status rendering (done/total ✓) on all three
todo tool call paths: read, create (merge=False), update (merge=True).
"""

import json
from agent.display import get_cute_tool_message


def _todo_result(total: int, completed: int) -> str:
    """Build a fake todo_tool return value."""
    return json.dumps({
        "todos": [],
        "summary": {
            "total": total,
            "pending": total - completed,
            "in_progress": 0,
            "completed": completed,
            "cancelled": 0,
        },
    })


class TestTodoRead:
    """get_cute_tool_message(…, result=…) when todos_arg is None (read path)."""

    def test_read_no_result(self):
        msg = get_cute_tool_message("todo", {}, 0.5)
        assert "reading tasks" in msg
        assert "0.5s" in msg

    def test_read_with_progress(self):
        msg = get_cute_tool_message("todo", {}, 0.5,
                                    result=_todo_result(4, 2))
        assert "2/4" in msg
        assert "task(s)" in msg

    def test_read_all_done(self):
        msg = get_cute_tool_message("todo", {}, 0.5,
                                    result=_todo_result(4, 4))
        assert "4/4" in msg
        assert "task(s)" in msg

    def test_read_zero_total(self):
        """Edge case: empty todo list returns summary with total=0."""
        msg = get_cute_tool_message("todo", {}, 0.5,
                                    result=_todo_result(0, 0))
        assert "reading tasks" in msg

    def test_read_invalid_result_fallback(self):
        """Garbage result should not crash; fall back to reading tasks."""
        msg = get_cute_tool_message("todo", {}, 0.5, result="not json")
        assert "reading tasks" in msg

    def test_read_result_missing_summary(self):
        msg = get_cute_tool_message("todo", {}, 0.5,
                                    result='{"todos": []}')
        assert "reading tasks" in msg


class TestTodoCreate:
    """get_cute_tool_message when merge=False (new plan creation)."""

    def test_create_default(self):
        """Brand-new plan: all pending, no result — plain count."""
        msg = get_cute_tool_message("todo",
                                    {"todos": [
                                        {"id": "a", "content": "x", "status": "pending"},
                                    ]}, 0.3)
        assert "1 task(s)" in msg
        assert "0.3s" in msg
        assert "/" not in msg  # no progress fraction

    def test_create_multiple(self):
        msg = get_cute_tool_message("todo",
                                    {"todos": [
                                        {"id": "a", "content": "x", "status": "pending"},
                                        {"id": "b", "content": "y", "status": "pending"},
                                        {"id": "c", "content": "z", "status": "pending"},
                                    ]}, 0.2)
        assert "3 task(s)" in msg

    def test_create_with_result_shows_progress_when_done(self):
        """Even on create, if result has completed tasks show it."""
        msg = get_cute_tool_message("todo",
                                    {"todos": [{"id": "a", "content": "x", "status": "completed"}]},
                                    0.4,
                                    result=_todo_result(1, 1))
        assert "1/1" in msg
        assert "task(s)" in msg

    def test_create_with_result_zero_done(self):
        """New plan with 0 done — plain count, no progress fraction."""
        msg = get_cute_tool_message("todo",
                                    {"todos": [
                                        {"id": "a", "content": "x", "status": "pending"},
                                        {"id": "b", "content": "y", "status": "pending"},
                                    ]},
                                    0.3,
                                    result=_todo_result(2, 0))
        assert "2 task(s)" in msg
        assert "/" not in msg


class TestTodoUpdate:
    """get_cute_tool_message when merge=True (incremental update)."""

    def test_update_no_result(self):
        """No result available — plain update N task(s)."""
        msg = get_cute_tool_message("todo",
                                    {"todos": [{"id": "a", "status": "completed"}],
                                     "merge": True}, 0.5)
        assert "update 1 task(s)" in msg

    def test_update_partial_progress(self):
        """1/4 tasks completed — show fraction with checkmark."""
        msg = get_cute_tool_message("todo",
                                    {"todos": [{"id": "a", "status": "completed"}],
                                     "merge": True},
                                    0.5,
                                    result=_todo_result(4, 1))
        assert "update" in msg
        assert "1/4" in msg
        assert "✓" in msg

    def test_update_halfway(self):
        """2/4 — midpoint progress."""
        msg = get_cute_tool_message("todo",
                                    {"todos": [{"id": "b", "status": "in_progress"}],
                                     "merge": True},
                                    0.7,
                                    result=_todo_result(4, 2))
        assert "2/4" in msg
        assert "✓" in msg

    def test_update_all_completed(self):
        """4/4 — full checkmark."""
        msg = get_cute_tool_message("todo",
                                    {"todos": [{"id": "d", "status": "completed"}],
                                     "merge": True},
                                    0.2,
                                    result=_todo_result(4, 4))
        assert "4/4" in msg
        assert "✓" in msg

    def test_update_zero_done(self):
        """No completed tasks yet — plain update N task(s)."""
        msg = get_cute_tool_message("todo",
                                    {"todos": [{"id": "a", "status": "pending"}],
                                     "merge": True},
                                    0.3,
                                    result=_todo_result(3, 0))
        assert "update 1 task(s)" in msg
        assert "✓" not in msg
        assert "/" not in msg  # no progress fraction when done=0

    def test_update_invalid_result_fallback(self):
        """Bad JSON result — fall back to plain update N task(s)."""
        msg = get_cute_tool_message("todo",
                                    {"todos": [{"id": "a", "status": "completed"}],
                                     "merge": True},
                                    0.6,
                                    result="{broken")
        assert "update 1 task(s)" in msg
        assert "✓" not in msg

    def test_update_result_missing_summary(self):
        """Result no summary key — fall back to plain update."""
        msg = get_cute_tool_message("todo",
                                    {"todos": [{"id": "a", "status": "completed"}],
                                     "merge": True},
                                    0.4,
                                    result='{"todos": []}')
        assert "update 1 task(s)" in msg
        assert "✓" not in msg

    def test_update_total_not_in_summary(self):
        """Result summary missing total key."""
        msg = get_cute_tool_message("todo",
                                    {"todos": [{"id": "a", "status": "completed"}],
                                     "merge": True},
                                    0.3,
                                    result=json.dumps({"summary": {"completed": 2}}))
        assert "update 1 task(s)" in msg
        assert "✓" not in msg

    def test_update_multiple_tasks_in_line(self):
        """Update line with several tasks in the update request."""
        msg = get_cute_tool_message("todo",
                                    {"todos": [
                                        {"id": "a", "status": "completed"},
                                        {"id": "b", "status": "in_progress"},
                                    ], "merge": True},
                                    0.5,
                                    result=_todo_result(5, 3))
        assert "update" in msg
        assert "3/5" in msg
        assert "✓" in msg


class TestTodoEdgeCases:
    """Boundary cases that should not crash."""

    def test_merge_default_value(self):
        """merge defaults to False in function signature, should be False when absent."""
        msg = get_cute_tool_message("todo",
                                    {"todos": [{"id": "a", "content": "x", "status": "pending"}]},
                                    1.0)
        assert "1 task(s)" in msg

    def test_duration_formatting(self):
        """Duration formatting works correctly."""
        msg = get_cute_tool_message("todo", {}, 0.123)
        assert "0.1s" in msg

        msg = get_cute_tool_message("todo", {}, 1.0)
        assert "1.0s" in msg

        msg = get_cute_tool_message("todo", {}, 123.456)
        assert "123.5s" in msg

    def test_large_task_count(self):
        """Many tasks should not break formatting."""
        many = [{"id": str(i), "content": "x", "status": "pending"} for i in range(50)]
        msg = get_cute_tool_message("todo", {"todos": many}, 0.5)
        assert "50 task(s)" in msg

    def test_read_with_no_args_and_no_result(self):
        """Completely empty call."""
        msg = get_cute_tool_message("todo", {}, 0.0)
        assert "reading tasks" in msg


class TestTodoSkinIntegration:
    """Verify the skin prefix is applied to todo messages too.
    This uses the same pattern as test_skin_engine test_tool_message_uses_skin_prefix.
    """

    def test_default_skin_prefix(self):
        msg = get_cute_tool_message("todo", {}, 0.5)
        assert msg.startswith("┊")


class TestWebExtractDisplay:
    """get_cute_tool_message for web_extract handles dict objects from web_search results.

    Reproduces and verifies fix for #61693 where web_search result dicts
    caused AttributeError when web_extract tried to extract domain names.
    """

    def test_web_extract_with_dict_url_field(self):
        """Dict with 'url' field (standard web_search result shape)."""
        args = {
            "urls": [
                {"url": "https://example.com/path", "title": "Example", "snippet": "..."}
            ]
        }
        msg = get_cute_tool_message("web_extract", args, 0.5)
        assert "example.com" in msg
        assert "📄" in msg
        assert "0.5s" in msg
        assert "+" not in msg  # only 1 URL

    def test_web_extract_with_dict_href_field(self):
        """Dict with 'href' field (alternate key)."""
        args = {
            "urls": [
                {"href": "http://test.org/page", "title": "Test", "snippet": "..."}
            ]
        }
        msg = get_cute_tool_message("web_extract", args, 0.3)
        assert "test.org" in msg

    def test_web_extract_with_dict_no_url_field(self):
        """Dict without url/href fields - should not crash."""
        args = {
            "urls": [
                {"title": "No URL", "snippet": "Missing url field"}
            ]
        }
        msg = get_cute_tool_message("web_extract", args, 0.2)
        assert "📄" in msg
        assert "pages" in msg
        assert "0.2s" in msg

    def test_web_extract_with_non_string_item_uses_generic_label(self):
        msg = get_cute_tool_message("web_extract", {"urls": [123]}, 0.2)
        assert "pages" in msg

    def test_web_extract_with_multiple_dicts(self):
        """Multiple dict URLs show '+N' suffix."""
        args = {
            "urls": [
                {"url": "https://example.com/page1"},
                {"url": "https://example.com/page2"},
                {"url": "https://example.com/page3"},
            ]
        }
        msg = get_cute_tool_message("web_extract", args, 0.6)
        assert "example.com" in msg
        assert "+2" in msg  # 3 URLs total, +2 beyond first

    def test_web_extract_with_mixed_types(self):
        """Mix of string URLs and dict objects."""
        args = {
            "urls": [
                "https://direct.com/page",
                {"url": "https://dict.com/page", "title": "Dict URL"},
            ]
        }
        msg = get_cute_tool_message("web_extract", args, 0.4)
        # First item is a string, so domain should come from it
        assert "direct.com" in msg

    def test_web_extract_empty_urls(self):
        """Empty urls list - shows 'pages' placeholder."""
        args = {"urls": []}
        msg = get_cute_tool_message("web_extract", args, 0.1)
        assert "pages" in msg
        assert "📄" in msg
