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

