"""Tests for post-execution tool result validation.

The validator catches malformed tool results before they reach the LLM.
It is intentionally lenient — unknown tools always pass — so that new
tools added anywhere in the codebase are never silently broken by a
validation rule that was never written for them.
"""

import pytest

from agent.tool_result_validator import validate_tool_result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def assert_valid(tool: str, result: object) -> None:
    is_valid, error = validate_tool_result(tool, result)
    assert is_valid, f"Expected valid for {tool!r} → {result!r}, got error: {error}"


def assert_invalid(tool: str, result: object) -> None:
    is_valid, error = validate_tool_result(tool, result)
    assert not is_valid, f"Expected invalid for {tool!r} → {result!r}"
    assert error, "Expected a non-empty error message"


# ---------------------------------------------------------------------------
# File-class tools: read_file, write_file, patch, search_files
# ---------------------------------------------------------------------------


class TestFileTools:
    def test_read_file_string_is_valid(self):
        assert_valid("read_file", "file content here")

    def test_read_file_empty_string_is_valid(self):
        # Empty files are legal
        assert_valid("read_file", "")

    def test_read_file_none_is_invalid(self):
        assert_invalid("read_file", None)

    def test_read_file_dict_is_invalid(self):
        assert_invalid("read_file", {"content": "oops"})

    def test_write_file_string_is_valid(self):
        assert_valid("write_file", "42 bytes written")

    def test_patch_string_is_valid(self):
        assert_valid("patch", "patched successfully")

    def test_search_files_string_is_valid(self):
        assert_valid("search_files", "src/foo.py:12: def bar()")

    def test_search_files_list_is_valid(self):
        # Some implementations return a list of match objects
        assert_valid("search_files", [{"file": "src/foo.py", "line": 12}])


# ---------------------------------------------------------------------------
# Terminal tool
# ---------------------------------------------------------------------------


class TestTerminalTool:
    def test_string_output_is_valid(self):
        assert_valid("terminal", "exit code 0\nsome output")

    def test_empty_output_is_valid(self):
        assert_valid("terminal", "")

    def test_none_is_invalid(self):
        assert_invalid("terminal", None)

    def test_dict_is_invalid(self):
        # terminal must return text, not a structured object
        assert_invalid("terminal", {"stdout": "ok", "exit_code": 0})


# ---------------------------------------------------------------------------
# Web / API tools: web_search, web_extract
# ---------------------------------------------------------------------------


class TestWebTools:
    def test_web_search_list_of_dicts_is_valid(self):
        assert_valid("web_search", [{"title": "T", "url": "https://x.com"}])

    def test_web_search_empty_list_is_valid(self):
        assert_valid("web_search", [])

    def test_web_search_string_is_valid(self):
        # Some providers return plain text summaries
        assert_valid("web_search", "No results found")

    def test_web_search_none_is_invalid(self):
        assert_invalid("web_search", None)

    def test_web_extract_string_is_valid(self):
        assert_valid("web_extract", "# Page Title\n\nBody text.")

    def test_web_extract_dict_is_valid(self):
        assert_valid("web_extract", {"url": "https://x.com", "content": "text"})


# ---------------------------------------------------------------------------
# Unknown / unregistered tools — must always pass
# ---------------------------------------------------------------------------


class TestUnknownTools:
    """Unknown tools must never be rejected — we have no schema for them."""

    def test_unknown_tool_string_passes(self):
        assert_valid("some_future_tool", "any string")

    def test_unknown_tool_dict_passes(self):
        assert_valid("my_custom_mcp_tool", {"key": "value"})

    def test_unknown_tool_list_passes(self):
        assert_valid("plugin_xyz_action", [1, 2, 3])

    def test_unknown_tool_none_passes(self):
        # Unknown tools: we can't know if None is wrong, so allow it
        assert_valid("mystery_tool", None)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_tool_name_case_sensitivity(self):
        # Tool names are case-sensitive; READ_FILE is unknown → always valid
        assert_valid("READ_FILE", None)

    def test_very_large_string_result(self):
        assert_valid("read_file", "x" * 100_000)

    def test_nested_dict_in_valid_list(self):
        assert_valid(
            "web_search",
            [{"title": "A", "url": "https://a.com", "nested": {"k": "v"}}],
        )


# ---------------------------------------------------------------------------
# Regression: error-keyed dicts must NOT be flagged as invalid
# ---------------------------------------------------------------------------


class TestErrorKeyedDictsAreData:
    """An {'error': ...} response from web_search/web_extract is valid data
    that the model should reason about — not a malformed result."""

    def test_web_search_error_dict_is_valid(self):
        assert_valid("web_search", {"error": "rate limited"})

    def test_web_extract_error_dict_is_valid(self):
        assert_valid("web_extract", {"error": "page not found", "url": "https://x.com"})

    def test_web_search_error_with_extra_keys_is_valid(self):
        assert_valid("web_search", {"error": "timeout", "query": "foo", "results": []})


# ---------------------------------------------------------------------------
# Wiring contract: validate_tool_result return values drive middleware_trace
# ---------------------------------------------------------------------------


class TestWiringContract:
    """The executor wiring appends to middleware_trace on invalid results.
    Simulate that logic here to confirm the contract without needing a full
    agent fixture.
    """

    def _simulate_executor_wiring(self, tool_name: str, result: object) -> list:
        """Mirrors the post-exec block in tool_executor.py."""
        from agent.tool_result_validator import validate_tool_result

        trace: list = []
        is_valid, validation_error = validate_tool_result(tool_name, result)
        if not is_valid:
            trace.append({
                "type": "tool_result_validation_error",
                "error": validation_error,
                "preview": repr(result)[:200],
            })
        return trace

    def test_none_result_appends_trace_entry(self):
        trace = self._simulate_executor_wiring("read_file", None)
        assert len(trace) == 1
        assert trace[0]["type"] == "tool_result_validation_error"
        assert "None" in trace[0]["error"]

    def test_valid_result_leaves_trace_empty(self):
        trace = self._simulate_executor_wiring("read_file", "file contents")
        assert trace == []

    def test_unknown_tool_none_leaves_trace_empty(self):
        # Unknown tools always pass — trace must stay empty
        trace = self._simulate_executor_wiring("future_tool", None)
        assert trace == []

    def test_trace_entry_includes_preview(self):
        trace = self._simulate_executor_wiring("terminal", {"exit_code": 0})
        assert len(trace) == 1
        assert "preview" in trace[0]
        assert trace[0]["preview"]  # non-empty

