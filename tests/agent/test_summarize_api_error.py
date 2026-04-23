"""Tests for AIAgent._summarize_api_error — the error-line formatter
used by the retry loop's `📝 Error:` display.

Regression tests for a gap where exceptions with empty string payloads
(bare `raise`, bare `assert`, or third-party SDK assertions that don't
carry the original error message) would surface as `📝 Error:` with no
content, forcing users to re-run under a debugger to see what failed.
"""
from run_agent import AIAgent


def _raise_and_catch(exc: Exception) -> Exception:
    """Raise and immediately re-catch so the exception has a real traceback."""
    try:
        raise exc
    except type(exc) as caught:
        return caught


class TestSummarizeApiErrorEmptyMessage:
    """Empty-message errors still surface actionable context."""

    def test_bare_assertion_error_shows_traceback_frame(self):
        # `raise AssertionError()` produces a truly empty-message error
        # (unlike `assert False` which Python annotates with source).
        def _trigger():
            raise AssertionError()

        try:
            _trigger()
        except AssertionError as e:
            result = AIAgent._summarize_api_error(e)

        assert "AssertionError" in result
        assert "_trigger" in result
        assert ".py:" in result

    def test_empty_raise_shows_type_only_when_no_traceback(self):
        # Construct an exception without raising it — no __traceback__
        err = RuntimeError("")
        result = AIAgent._summarize_api_error(err)

        assert "RuntimeError" in result
        assert "no message" in result

    def test_whitespace_only_message_treated_as_empty(self):
        err = _raise_and_catch(RuntimeError("   \n\t  "))
        result = AIAgent._summarize_api_error(err)

        assert "RuntimeError" in result
        # Must include a frame locator — not just the empty payload
        assert ".py:" in result or "no message" in result

    def test_non_empty_message_falls_through_to_normal_path(self):
        """Non-empty errors must not be captured by the new branch."""
        err = _raise_and_catch(RuntimeError("some real error"))
        result = AIAgent._summarize_api_error(err)

        # The normal path returns the raw string (possibly truncated),
        # NOT the traceback-frame format introduced by this fix.
        assert "some real error" in result
        assert ".py:" not in result
