"""Tests for goal-judge error detail surfacing (fix/goal-judge-error-body).

When the goal judge's LLM call fails, the continuation-prompt reason must
surface the provider's actual error body (truncated), not just the SDK
exception class name — a bare "PermissionDeniedError" hides the real cause
(e.g. OpenRouter 403 "model not available in your region").
"""

from unittest.mock import MagicMock, patch

from hermes_cli import goals


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
