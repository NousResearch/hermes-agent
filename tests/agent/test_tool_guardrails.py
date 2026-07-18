"""Regression tests for agent.tool_guardrails — issue #19814.

Before the fix, ``classify_tool_failure()`` crashed with
``KeyError: slice(None, 500, None)`` when a custom-tools plugin returned a
dict result, because the function assumed all tool results are strings.
"""

from __future__ import annotations

import pytest
from agent.tool_guardrails import classify_tool_failure


class TestClassifyToolFailureDictResult:
    """``classify_tool_failure`` must handle non-string results gracefully."""

    def test_dict_result_returns_no_error(self) -> None:
        """A dict-typed tool result is not a failure string — no crash."""
        is_error, label = classify_tool_failure("custom_read", {"key": "value"})
        assert is_error is False
        assert label == ""

    def test_list_result_returns_no_error(self) -> None:
        """List-typed content from multimodal tools must not crash."""
        is_error, label = classify_tool_failure(
            "browser_vision", [{"type": "text", "text": "OK"}]
        )
        assert is_error is False
        assert label == ""

    def test_plain_string_error_still_detected(self) -> None:
        """String results with error keywords are still flagged."""
        is_error, label = classify_tool_failure("unknown_tool", '{"error": "failed"}')
        assert is_error is True
        assert "[error]" in label

    def test_plain_string_success_passes(self) -> None:
        """Normal string results without error markers pass."""
        is_error, label = classify_tool_failure("unknown_tool", '{"result": "done"}')
        assert is_error is False
        assert label == ""

    def test_none_result_returns_no_error(self) -> None:
        """None results (cancelled/timeout tools) must not crash."""
        is_error, label = classify_tool_failure("terminal", None)
        assert is_error is False
        assert label == ""

    def test_int_result_returns_no_error(self) -> None:
        """Numeric results (rare but possible from custom tools) must not crash."""
        is_error, label = classify_tool_failure("custom_counter", 42)
        assert is_error is False
        assert label == ""

    def test_bool_result_returns_no_error(self) -> None:
        """Boolean results must not crash the classifier."""
        is_error, label = classify_tool_failure("custom_check", True)
        assert is_error is False
        assert label == ""
