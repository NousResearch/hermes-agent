#!/usr/bin/env python3
"""E2E test: aliased tool call succeeds end-to-end."""

import os
import sys
import tempfile
import json

# Add paths for imports
sys.path.insert(0, "/home/hermes/hermes-agent")
sys.path.insert(0, "/home/hermes/hermes-agent/.worktrees/tool-input-repair")

# Initialize tool registry
import tools.registry
tools.registry.discover_builtin_tools()

# Import the function under test
from model_tools import handle_function_call


def test_e2e_aliased_field_succeeds():
    """E2E: a tool call with {"file_path": "/etc/hosts"} to read_file succeeds end-to-end."""

    # Call read_file with aliased field "file_path" instead of "path"
    result_json = handle_function_call(
        function_name="read_file",
        function_args={"file_path": "/etc/hosts", "offset": 1, "limit": 10},
        task_id="test_e2e_task",
        session_id="test_e2e_session",
    )

    # Result has repair note prepended, extract the JSON part
    # Format: "[Tool input repair applied: file_path → path]\n{...json...}"
    if result_json.startswith("[Tool input repair applied:"):
        # Find the newline after the note
        json_start = result_json.find("\n") + 1
        result_json = result_json[json_start:]

    result = json.loads(result_json)

    # The call should succeed (not error)
    assert "error" not in result, f"Expected success, got error: {result}"

    # Should have file content
    assert "content" in result, f"Expected 'content' field in result: {result}"

    # /etc/hosts should contain "127.0.0.1"
    assert "127.0.0.1" in result["content"], f"Expected /etc/hosts content, got: {result['content'][:100]}"

    print("✓ E2E test passed: aliased field 'file_path' was repaired to 'path' and read_file succeeded")


if __name__ == "__main__":
    test_e2e_aliased_field_succeeds()
    print("\nAll E2E tests passed!")