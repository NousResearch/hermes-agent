#!/usr/bin/env python3
"""E2E test: aliased tool call succeeds end-to-end."""

import json
import sys

# Add paths for imports
sys.path.insert(0, "/home/hermes/hermes-agent")
sys.path.insert(0, "/home/hermes/hermes-agent/.worktrees/tool-input-repair")

# Initialize tool registry
import tools.registry

tools.registry.discover_builtin_tools()

# Import the function under test
from model_tools import handle_function_call


def _json_from_result(result_text: str) -> dict:
    if result_text.startswith("[Tool input repair applied:"):
        result_text = result_text[result_text.find("\n") + 1 :]
    return json.loads(result_text)


def test_e2e_aliased_field_succeeds():
    """E2E: a tool call with {"file_path": "/etc/hosts"} to read_file succeeds end-to-end."""

    result_json = handle_function_call(
        function_name="read_file",
        function_args={"file_path": "/etc/hosts", "offset": 1, "limit": 10},
        task_id="test_e2e_task",
        session_id="test_e2e_session",
    )

    result = _json_from_result(result_json)

    assert "error" not in result, f"Expected success, got error: {result}"
    assert "content" in result, f"Expected 'content' field in result: {result}"
    assert "127.0.0.1" in result["content"], f"Expected /etc/hosts content, got: {result['content'][:100]}"


def test_e2e_json_stringified_array_succeeds():
    result_json = handle_function_call(
        function_name="read_file",
        function_args={"path": "/etc/hosts", "offset": 1, "limit": "[10]"},
        task_id="test_e2e_task",
        session_id="test_e2e_session",
    )

    result = _json_from_result(result_json)
    assert "error" not in result
    assert "content" in result


if __name__ == "__main__":
    test_e2e_aliased_field_succeeds()
    test_e2e_json_stringified_array_succeeds()
    print("All E2E tests passed!")
