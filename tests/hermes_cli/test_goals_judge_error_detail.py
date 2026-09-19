"""Tests for goal-judge error detail surfacing (fix/goal-judge-error-body).

When the goal judge's LLM call fails, the continuation-prompt reason must
surface the provider's actual error body (truncated), not just the SDK
exception class name — a bare "PermissionDeniedError" hides the real cause
(e.g. OpenRouter 403 "model not available in your region").
"""

from unittest.mock import MagicMock, patch

from hermes_cli import goals


class _EmptyStrError(Exception):
    """An SDK-style exception whose str() is empty — the detail lives only
    on .body/.status attributes (OpenRouter/Gemini error shapes)."""

    def __init__(self, body=None, status=None):
        super().__init__()
        self.body = body
        self.status = status

    def __str__(self) -> str:
        return ""


class TestJudgeErrorDetail:
    def test_error_includes_exception_detail(self):
        with patch(
            "agent.auxiliary_client.call_llm",
            side_effect=RuntimeError("403 model not available in your region"),
        ):
            verdict, reason, _, _wd, _tf = goals.judge_goal("goal", "response")
        assert verdict == "continue"
        assert "judge error" in reason.lower()
        assert "403" in reason
        assert "model not available" in reason

    def test_error_detail_truncated(self):
        """A pathologically long provider error is bounded to a fixed window
        so the continuation prompt doesn't balloon."""
        with patch(
            "agent.auxiliary_client.call_llm",
            side_effect=RuntimeError("x" * 500),
        ):
            verdict, reason, _, _wd, _tf = goals.judge_goal("goal", "response")
        assert verdict == "continue"
        assert len(reason) < 500

    def test_original_fail_open_semantics_unchanged(self):
        """The pre-existing fail-open behaviour (continue on any judge error)
        is preserved — the patch only enriches the reason string."""
        with patch(
            "agent.auxiliary_client.call_llm",
            side_effect=Exception("any failure"),
        ):
            verdict, reason, _, _wd, _tf = goals.judge_goal("goal", "response")
        assert verdict == "continue"
        assert "judge error" in reason.lower()

    def test_fallback_to_sdk_body_when_str_is_empty(self):
        """Some SDK exceptions stringify to '' — the detail must be picked up
        from .body/.status so the reason still carries the real cause."""
        err = _EmptyStrError(body="403 model not available in your region")
        with patch("agent.auxiliary_client.call_llm", side_effect=err):
            verdict, reason, _, _wd, _tf = goals.judge_goal("goal", "response")
        assert verdict == "continue"
        assert "judge error" in reason.lower()
        assert "403" in reason
        assert "model not available" in reason

    def test_fully_empty_exception_has_no_dangling_colon(self):
        """str(exc), .body and .status are all empty → reason is just
        \"judge error: <TypeName>\" and never ends in a dangling \": \"."""
        err = _EmptyStrError()
        with patch("agent.auxiliary_client.call_llm", side_effect=err):
            verdict, reason, _, _wd, _tf = goals.judge_goal("goal", "response")
        assert verdict == "continue"
        assert reason == f"judge error: {type(err).__name__}"
        assert not reason.endswith(": ")
