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



    def test_read_zero_total(self):
        """Edge case: empty todo list returns summary with total=0."""
        msg = get_cute_tool_message("todo", {}, 0.5,
                                    result=_todo_result(0, 0))
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


    def test_update_halfway(self):
        """2/4 — midpoint progress."""
        msg = get_cute_tool_message("todo",
                                    {"todos": [{"id": "b", "status": "in_progress"}],
                                     "merge": True},
                                    0.7,
                                    result=_todo_result(4, 2))
        assert "2/4" in msg
        assert "✓" in msg





    def test_update_total_not_in_summary(self):
        """Result summary missing total key."""
        msg = get_cute_tool_message("todo",
                                    {"todos": [{"id": "a", "status": "completed"}],
                                     "merge": True},
                                    0.3,
                                    result=json.dumps({"summary": {"completed": 2}}))
        assert "update 1 task(s)" in msg
        assert "✓" not in msg



class TestTodoEdgeCases:
    """Boundary cases that should not crash."""

    def test_merge_default_value(self):
        """merge defaults to False in function signature, should be False when absent."""
        msg = get_cute_tool_message("todo",
                                    {"todos": [{"id": "a", "content": "x", "status": "pending"}]},
                                    1.0)
        assert "1 task(s)" in msg


    def test_large_task_count(self):
        """Many tasks should not break formatting."""
        many = [{"id": str(i), "content": "x", "status": "pending"} for i in range(50)]
        msg = get_cute_tool_message("todo", {"todos": many}, 0.5)
        assert "50 task(s)" in msg



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


    def test_web_extract_with_dict_href_field(self):
        """Dict with 'href' field (alternate key)."""
        args = {
            "urls": [
                {"href": "http://test.org/page", "title": "Test", "snippet": "..."}
            ]
        }
        msg = get_cute_tool_message("web_extract", args, 0.3)
        assert "test.org" in msg




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
