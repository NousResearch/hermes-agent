"""Tests for run_job partial-success handling and per-job max_turns override.

Covers the two behaviours added in the PR under test:

1. Per-job max_turns override: job["max_turns"] (positive int) takes precedence
   over _cfg["agent"]["max_turns"] / _cfg["max_turns"] / 90 default.

2. Partial-success path: when failed is NOT True AND turn_exit_reason starts
   with "max_iterations_reached" or "budget_exhausted" AND the final_response
   has >= 300 chars, run_job records PARTIAL success (prepends warning banner,
   returns success=True, no RuntimeError) instead of raising.

Mocking pattern mirrors TestRunJobSessionPersistence in test_scheduler.py:
  - patch cron.scheduler._hermes_home   → tmp_path (avoids real FS)
  - patch cron.scheduler._resolve_origin → None
  - patch dotenv.load_dotenv             → no-op
  - patch hermes_state.SessionDB        → MagicMock fake_db
  - patch hermes_cli.runtime_provider.resolve_runtime_provider → minimal dict
  - patch run_agent.AIAgent             → MagicMock whose run_conversation
                                           returns a controlled result dict
"""

from unittest.mock import MagicMock, patch

import pytest

from cron.scheduler import run_job

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_FAKE_RUNTIME = {
    "api_key": "test-key",
    "base_url": "https://example.invalid/v1",
    "provider": "openrouter",
    "api_mode": "chat_completions",
}

# A report long enough to satisfy the >= 300 char floor.
_LONG_REPORT = (
    "Detailed handoff brief. " * 20  # 480 chars
)
assert len(_LONG_REPORT.strip()) >= 300, "fixture is too short — adjust"

# A report deliberately shorter than 300 chars.
_SHORT_REPORT = "Brief summary. " * 10  # 150 chars
assert len(_SHORT_REPORT.strip()) < 300, "fixture is too long — adjust"


def _make_job(**extra):
    """Minimal valid job dict, merging any extra keys."""
    base = {
        "id": "partial-budget-test",
        "name": "partial budget test",
        "prompt": "do work",
    }
    base.update(extra)
    return base


def _run_job_with_agent_result(tmp_path, agent_result, job_extra=None):
    """Run run_job with a controlled agent result dict.

    Returns (success, output, final_response, error) plus the
    mock_agent_cls so callers can inspect call_args.
    """
    job = _make_job(**(job_extra or {}))
    fake_db = MagicMock()

    with patch("cron.scheduler._hermes_home", tmp_path), \
         patch("cron.scheduler._resolve_origin", return_value=None), \
         patch("dotenv.load_dotenv"), \
         patch("hermes_state.SessionDB", return_value=fake_db), \
         patch(
             "hermes_cli.runtime_provider.resolve_runtime_provider",
             return_value=_FAKE_RUNTIME,
         ), \
         patch("run_agent.AIAgent") as mock_agent_cls:
        mock_agent = MagicMock()
        mock_agent.run_conversation.return_value = agent_result
        mock_agent_cls.return_value = mock_agent

        result = run_job(job)

    return result, mock_agent_cls


# ---------------------------------------------------------------------------
# Case (a) — partial success: budget_exhausted with long report
# ---------------------------------------------------------------------------

class TestPartialSuccessBudgetExhausted:
    def test_budget_exhausted_long_report_succeeds(self, tmp_path):
        """failed=False, completed=False, turn_exit_reason='budget_exhausted',
        final_response >= 300 chars → success=True, no error, banner present."""
        agent_result = {
            "failed": False,
            "completed": False,
            "turn_exit_reason": "budget_exhausted",
            "final_response": _LONG_REPORT,
        }
        (success, output, final_response, error), _ = _run_job_with_agent_result(
            tmp_path, agent_result
        )

        assert success is True, "Expected partial success to be truthy"
        assert not error, f"Expected error to be falsy, got: {error!r}"
        assert "⚠️ PARTIAL" in final_response, (
            "Expected PARTIAL banner in final_response"
        )
        assert "budget_exhausted" in final_response, (
            "Expected exit reason in final_response banner"
        )
        # The original report text must be preserved below the banner.
        assert _LONG_REPORT.strip() in final_response, (
            "Original report text should be included after the banner"
        )

    def test_max_iterations_reached_long_report_succeeds(self, tmp_path):
        """turn_exit_reason='max_iterations_reached(90/90)' also triggers partial."""
        exit_reason = "max_iterations_reached(90/90)"
        agent_result = {
            "failed": False,
            "completed": False,
            "turn_exit_reason": exit_reason,
            "final_response": _LONG_REPORT,
        }
        (success, output, final_response, error), _ = _run_job_with_agent_result(
            tmp_path, agent_result
        )

        assert success is True
        assert not error
        assert "⚠️ PARTIAL" in final_response
        assert exit_reason in final_response
        assert _LONG_REPORT.strip() in final_response


# ---------------------------------------------------------------------------
# Case (b) — short report does NOT qualify for partial path
# ---------------------------------------------------------------------------

class TestShortReportStillFails:
    def test_budget_exhausted_short_report_fails(self, tmp_path):
        """Same turn_exit_reason but final_response < 300 chars → failure, not partial."""
        agent_result = {
            "failed": False,
            "completed": False,
            "turn_exit_reason": "budget_exhausted",
            "final_response": _SHORT_REPORT,
        }
        (success, output, final_response, error), _ = _run_job_with_agent_result(
            tmp_path, agent_result
        )

        assert not success, "Short-report partial should NOT succeed"
        assert error, "Expected error to be set"

    def test_max_iterations_reached_short_report_fails(self, tmp_path):
        """max_iterations_reached with < 300 char report → failure."""
        agent_result = {
            "failed": False,
            "completed": False,
            "turn_exit_reason": "max_iterations_reached(5/5)",
            "final_response": _SHORT_REPORT,
        }
        (success, output, final_response, error), _ = _run_job_with_agent_result(
            tmp_path, agent_result
        )

        assert not success
        assert error

    def test_empty_response_with_budget_exit_fails(self, tmp_path):
        """Empty final_response cannot be partial — must fail."""
        agent_result = {
            "failed": False,
            "completed": False,
            "turn_exit_reason": "budget_exhausted",
            "final_response": "",
        }
        (success, output, final_response, error), _ = _run_job_with_agent_result(
            tmp_path, agent_result
        )

        assert not success
        assert error


# ---------------------------------------------------------------------------
# Case (c) — failed=True always propagates as failure
# ---------------------------------------------------------------------------

class TestGenuineFailureNotPartial:
    def test_failed_true_with_long_response_still_fails(self, tmp_path):
        """failed=True must NOT be treated as partial even with a long response."""
        agent_result = {
            "failed": True,
            "completed": False,
            "turn_exit_reason": "budget_exhausted",
            "final_response": _LONG_REPORT,
        }
        (success, output, final_response, error), _ = _run_job_with_agent_result(
            tmp_path, agent_result
        )

        assert not success, "failed=True should always report failure"
        assert error, "Expected error to be set when failed=True"
        # Banner must NOT appear — this is a genuine failure, not a partial.
        assert "⚠️ PARTIAL" not in (final_response or ""), (
            "PARTIAL banner must not appear for a genuine failure"
        )

    def test_failed_true_with_max_iterations_reason_still_fails(self, tmp_path):
        """Even when turn_exit_reason looks like a budget stop, failed=True wins."""
        agent_result = {
            "failed": True,
            "completed": False,
            "turn_exit_reason": "max_iterations_reached(90/90)",
            "final_response": _LONG_REPORT,
        }
        (success, output, final_response, error), _ = _run_job_with_agent_result(
            tmp_path, agent_result
        )

        assert not success
        assert error

    def test_plain_api_failure_reports_error(self, tmp_path):
        """A normal API-exhaustion failure dict is surfaced as failure."""
        agent_result = {
            "final_response": "API call failed after 3 retries: Request timed out.",
            "failed": True,
            "completed": False,
            "error": "API call failed after 3 retries: Request timed out.",
        }
        (success, output, final_response, error), _ = _run_job_with_agent_result(
            tmp_path, agent_result
        )

        assert not success
        assert error
        assert "API call failed" in error


# ---------------------------------------------------------------------------
# Case (d) — per-job max_turns override
# ---------------------------------------------------------------------------

class TestPerJobMaxTurnsOverride:
    def test_job_max_turns_passed_to_agent_as_max_iterations(self, tmp_path):
        """job['max_turns'] (positive int) must be forwarded to AIAgent as
        max_iterations, overriding the global config default of 90.

        The existing harness exposes AIAgent's constructor kwargs via
        mock_agent_cls.call_args.kwargs — the same seam used by e.g.
        test_run_job_passes_enabled_toolsets_to_agent.  We inspect
        kwargs['max_iterations'] directly.
        """
        job_extra = {"max_turns": 150}
        agent_result = {"final_response": "done"}

        (success, *_), mock_agent_cls = _run_job_with_agent_result(
            tmp_path, agent_result, job_extra=job_extra
        )

        assert success is True
        kwargs = mock_agent_cls.call_args.kwargs
        assert kwargs["max_iterations"] == 150, (
            f"Expected max_iterations=150 (from job['max_turns']), "
            f"got {kwargs['max_iterations']!r}"
        )

    def test_zero_max_turns_falls_back_to_global_default(self, tmp_path):
        """job['max_turns']=0 is not a positive int → falls back to config/default."""
        job_extra = {"max_turns": 0}
        agent_result = {"final_response": "done"}

        (success, *_), mock_agent_cls = _run_job_with_agent_result(
            tmp_path, agent_result, job_extra=job_extra
        )

        assert success is True
        kwargs = mock_agent_cls.call_args.kwargs
        # With no config.yaml in tmp_path, the fallback is 90.
        assert kwargs["max_iterations"] == 90, (
            f"Expected fallback max_iterations=90, got {kwargs['max_iterations']!r}"
        )

    def test_negative_max_turns_falls_back_to_global_default(self, tmp_path):
        """job['max_turns']=-1 is also not positive → falls back."""
        job_extra = {"max_turns": -1}
        agent_result = {"final_response": "done"}

        (success, *_), mock_agent_cls = _run_job_with_agent_result(
            tmp_path, agent_result, job_extra=job_extra
        )

        assert success is True
        kwargs = mock_agent_cls.call_args.kwargs
        assert kwargs["max_iterations"] == 90

    def test_string_max_turns_falls_back_to_global_default(self, tmp_path):
        """job['max_turns']='150' is not an int → falls back (type guard)."""
        job_extra = {"max_turns": "150"}
        agent_result = {"final_response": "done"}

        (success, *_), mock_agent_cls = _run_job_with_agent_result(
            tmp_path, agent_result, job_extra=job_extra
        )

        assert success is True
        kwargs = mock_agent_cls.call_args.kwargs
        assert kwargs["max_iterations"] == 90

    def test_bool_max_turns_falls_back_to_global_default(self, tmp_path):
        """job['max_turns']=True must NOT be treated as 1: bool is an int
        subclass, so it's excluded explicitly and falls back to the default."""
        job_extra = {"max_turns": True}
        agent_result = {"final_response": "done"}

        (success, *_), mock_agent_cls = _run_job_with_agent_result(
            tmp_path, agent_result, job_extra=job_extra
        )

        assert success is True
        kwargs = mock_agent_cls.call_args.kwargs
        assert kwargs["max_iterations"] == 90, (
            f"bool max_turns must fall back to 90, got {kwargs['max_iterations']!r}"
        )

    def test_max_turns_config_yaml_used_when_no_job_override(self, tmp_path):
        """When no job['max_turns'], the config.yaml agent.max_turns is used."""
        (tmp_path / "config.yaml").write_text(
            "agent:\n  max_turns: 42\n",
            encoding="utf-8",
        )
        agent_result = {"final_response": "done"}

        (success, *_), mock_agent_cls = _run_job_with_agent_result(
            tmp_path, agent_result
        )

        assert success is True
        kwargs = mock_agent_cls.call_args.kwargs
        assert kwargs["max_iterations"] == 42, (
            f"Expected config.yaml max_turns=42 to be used, "
            f"got {kwargs['max_iterations']!r}"
        )

    def test_job_max_turns_beats_config_yaml(self, tmp_path):
        """job['max_turns'] wins over config.yaml agent.max_turns."""
        (tmp_path / "config.yaml").write_text(
            "agent:\n  max_turns: 42\n",
            encoding="utf-8",
        )
        job_extra = {"max_turns": 200}
        agent_result = {"final_response": "done"}

        (success, *_), mock_agent_cls = _run_job_with_agent_result(
            tmp_path, agent_result, job_extra=job_extra
        )

        assert success is True
        kwargs = mock_agent_cls.call_args.kwargs
        assert kwargs["max_iterations"] == 200, (
            "job['max_turns'] must override config.yaml agent.max_turns"
        )
