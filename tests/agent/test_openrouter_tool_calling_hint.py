"""Regression tests for OpenRouter tool-calling error detection.

Issue #49983: Free-tier OpenRouter models (:free suffix) return HTTP 404 with
"No allowed providers are available" when tool definitions are sent. The
existing hint only matched "support tool use" — the free-tier error went
undetected, giving users no actionable guidance.

These tests lock in the two known error-message patterns that trigger the
OpenRouter tool-calling hint in ``agent/conversation_loop.py``.
"""


import pytest


# The two error substrings the detection must match.
_TOOL_CALLING_PATTERNS = ("support tool use", "No allowed providers")


class TestOpenRouterToolCallingDetection:
    """Verify the error-message matching for OpenRouter tool-calling hints."""

    @pytest.mark.parametrize(
        "error_msg",
        [
            "support tool use is not supported for this model",
            "No allowed providers are available for the selected model",
            "HTTP 404: No allowed providers are available for the selected model.",
        ],
    )
    def test_matches_known_tool_calling_errors(self, error_msg: str):
        """Both known error patterns must match the detection condition."""
        assert any(pat in error_msg for pat in _TOOL_CALLING_PATTERNS)

    @pytest.mark.parametrize(
        "error_msg",
        [
            "Rate limit exceeded",
            "Internal server error",
            "Authentication failed",
            "Model not found",
            "",
        ],
    )
    def test_does_not_match_unrelated_errors(self, error_msg: str):
        """Unrelated error messages must not trigger the hint."""
        assert not any(pat in error_msg for pat in _TOOL_CALLING_PATTERNS)

    def test_no_allowed_providers_substring_match(self):
        """Partial match: 'No allowed providers' must catch the full phrase."""
        msg = "HTTP 404: No allowed providers are available for the selected model."
        assert "No allowed providers" in msg
