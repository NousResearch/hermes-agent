import pytest
import json
from unittest.mock import patch
from agent.display import get_cute_tool_message, _detect_tool_failure

def test_terminal_error_with_message():
    data = {"output": "some output", "exit_code": 1, "error": "This is a very specific and long error message that should be truncated"}
    is_failure, suffix = _detect_tool_failure("terminal", json.dumps(data))
    assert is_failure is True
    # err_msg[:32] is "This is a very specific and long"
    assert suffix == " [This is a very specific and long]"

def test_terminal_error_no_message():
    data = {"output": "some output", "exit_code": 1}
    is_failure, suffix = _detect_tool_failure("terminal", json.dumps(data))
    assert is_failure is True
    assert suffix == " [exit 1]"

def test_file_not_found_path_shortening():
    data = {"error": "File not found: /very/long/path/name.txt"}
    is_failure, suffix = _detect_tool_failure("some_tool", json.dumps(data))
    assert is_failure is True
    assert suffix == " [File not found: name.txt]"

def test_generic_error_in_json():
    data = {"error": "Specific generic error"}
    is_failure, suffix = _detect_tool_failure("some_tool", json.dumps(data))
    assert is_failure is True
    assert suffix == " [Specific generic error]"

    data = {"message": "Another generic error", "success": False}
    is_failure, suffix = _detect_tool_failure("some_tool", json.dumps(data))
    assert is_failure is True
    assert suffix == " [Another generic error]"

@patch("agent.display._tool_preview_max_len", 256)
def test_trunc_large_default():
    long_string = "A" * 300
    msg = get_cute_tool_message("terminal", {"command": long_string}, 0.5)
    assert len(msg) > 0
    assert "..." in msg
    assert len(msg) < 300

@patch("agent.display._tool_preview_max_len", 128)
def test_path_large_default():
    long_path = "/a" * 100
    msg = get_cute_tool_message("read_file", {"path": long_path}, 0.5)
    assert len(msg) > 0
    assert "..." in msg
    assert len(msg) < 300

@patch("agent.display._tool_preview_max_len", 64)
def test_web_search_long_query():
    long_query = "Q" * 100
    msg = get_cute_tool_message("web_search", {"query": long_query}, 0.5)
    assert len(msg) > 0
    assert "..." in msg
    assert len(msg) < 150

def test_success_result_no_suffix():
    result_data = {"exit_code": 0, "output": "hello"}
    msg = get_cute_tool_message("terminal", {"command": "echo 'hello'"}, 0.5, json.dumps(result_data))
    assert "exit 0" not in msg
    assert "error" not in msg
    assert "]" not in msg


# ── web_search empty result detection ──

def test_web_search_empty_results_detected_as_failure():
    """Empty web_search data.web should be detected as failure with [0 results]."""
    data = {"success": True, "data": {"web": []}}
    is_failure, suffix = _detect_tool_failure("web_search", json.dumps(data))
    assert is_failure is True
    assert suffix == " [0 results]"


def test_web_search_with_results_not_failure():
    """web_search with non-empty results should not be detected as failure."""
    data = {"success": True, "data": {"web": [{"title": "X", "url": "http://x.com", "description": "desc"}]}}
    is_failure, suffix = _detect_tool_failure("web_search", json.dumps(data))
    assert is_failure is False


def test_web_search_error_message_detected():
    """web_search with error+success=false should still be detected."""
    data = {"error": "0 results after 2 attempts", "success": False}
    is_failure, suffix = _detect_tool_failure("web_search", json.dumps(data))
    assert is_failure is True
    assert suffix == " [0 results after 2 attempts]"


# ── web_search result count in display ──

def test_web_search_display_shows_result_count():
    """get_cute_tool_message should show result count in web_search line."""
    data = {"success": True, "data": {"web": [
        {"title": "A", "url": "http://a.com", "description": "d"},
        {"title": "B", "url": "http://b.com", "description": "d"},
    ]}}
    msg = get_cute_tool_message("web_search", {"query": "test"}, 1.5, json.dumps(data))
    assert "(2 results)" in msg


def test_web_search_display_singular_result():
    """Single result should show '(1 result)' not '(1 results)'."""
    data = {"success": True, "data": {"web": [
        {"title": "A", "url": "http://a.com", "description": "d"},
    ]}}
    msg = get_cute_tool_message("web_search", {"query": "test"}, 0.5, json.dumps(data))
    assert "(1 result)" in msg


def test_web_search_display_zero_results_with_failure_suffix():
    """Zero results should show both count and [0 results] suffix."""
    data = {"success": True, "data": {"web": []}}
    msg = get_cute_tool_message("web_search", {"query": "test"}, 9.0, json.dumps(data))
    assert "(0 results)" in msg
    assert "[0 results]" in msg


def test_web_search_display_no_result_param():
    """When result is not provided, no count is shown (backward compat)."""
    msg = get_cute_tool_message("web_search", {"query": "test"}, 0.5)
    assert "(0 results)" not in msg
    assert "results" not in msg
    assert "🔍 search" in msg

