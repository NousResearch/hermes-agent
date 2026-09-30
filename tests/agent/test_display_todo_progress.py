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

class TestTodoCreate:
    """get_cute_tool_message when merge=False (new plan creation)."""

    def test_create_with_result_zero_done(self):
        """New plan with 0 done — plain count, no progress fraction."""
        msg = get_cute_tool_message("todo_list",
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

    def test_update_halfway(self):
        """2/4 — midpoint progress."""
        msg = get_cute_tool_message("todo_list",
                                    {"todos": [{"id": "b", "status": "in_progress"}],
                                     "merge": True},
                                    0.7,
                                    result=_todo_result(4, 2))
        assert "2/4" in msg
        assert "✓" in msg

    def test_update_total_not_in_summary(self):
        """Result summary missing total key."""
        msg = get_cute_tool_message("todo_list",
                                    {"todos": [{"id": "a", "status": "completed"}],
                                     "merge": True},
                                    0.3,
                                    result=json.dumps({"summary": {"completed": 2}}))
        assert "update 1 task(s)" in msg
        assert "✓" not in msg

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
