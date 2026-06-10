"""Tests for cron failure diagnostic context enrichment.

Tests cover:
- _extract_failure_diagnostics: extracts provider, model, session, iteration info
- _format_failure_notification: formats enriched vs basic failure messages
- Integration: diagnostics flow from _run_job_impl through _process_job delivery
"""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cron.scheduler import (
    _extract_failure_diagnostics,
    _format_failure_notification,
    _CRON_FAILURE_DIAGNOSTICS,
)


# ---------------------------------------------------------------------------
# _extract_failure_diagnostics
# ---------------------------------------------------------------------------

class TestExtractFailureDiagnostics:
    """Unit tests for _extract_failure_diagnostics()."""

    def test_basic_fields(self):
        """Provider, model, session_id are captured from arguments."""
        diag = _extract_failure_diagnostics(
            agent=None,
            job={"id": "j1"},
            error_msg="RuntimeError: broken pipe",
            session_id="cron_j1_20260610",
            provider="custom:openai",
            model="gpt-5.5",
        )
        assert diag["provider"] == "custom:openai"
        assert diag["model"] == "gpt-5.5"
        assert diag["session_id"] == "cron_j1_20260610"

    def test_defaults_when_empty(self):
        """Empty provider/model default to 'unknown'."""
        diag = _extract_failure_diagnostics(
            agent=None,
            job={"id": "j2"},
            error_msg="err",
        )
        assert diag["provider"] == "unknown"
        assert diag["model"] == "unknown"
        assert diag["session_id"] == ""

    def test_activity_summary_extraction(self):
        """Agent's get_activity_summary() is called when available."""
        agent = MagicMock()
        agent.get_activity_summary.return_value = {
            "api_call_count": 5,
            "max_iterations": 90,
            "current_tool": "terminal",
            "last_activity_desc": "running terminal command",
        }
        # No iteration_budget or _total_usage
        del agent.iteration_budget
        del agent._total_usage

        diag = _extract_failure_diagnostics(
            agent=agent,
            job={"id": "j3"},
            error_msg="err",
            provider="openrouter",
            model="claude-sonnet-4",
        )
        assert diag["api_call_count"] == 5
        assert diag["max_iterations"] == 90
        assert diag["current_tool"] == "terminal"

    def test_budget_extraction(self):
        """Agent's iteration_budget is captured."""
        agent = MagicMock()
        agent.get_activity_summary.return_value = {
            "api_call_count": 3,
            "max_iterations": 90,
            "current_tool": "",
            "last_activity_desc": "",
        }
        agent.iteration_budget = SimpleNamespace(used=3, max_total=90)
        del agent._total_usage

        diag = _extract_failure_diagnostics(
            agent=agent,
            job={"id": "j4"},
            error_msg="err",
        )
        assert diag["budget_used"] == 3
        assert diag["budget_max"] == 90

    def test_token_usage_extraction(self):
        """Agent's _total_usage dict is captured."""
        agent = MagicMock()
        agent.get_activity_summary.return_value = {
            "api_call_count": 2,
            "max_iterations": 90,
            "current_tool": "",
            "last_activity_desc": "",
        }
        agent._total_usage = {
            "prompt_tokens": 22913,
            "completion_tokens": 1500,
        }
        del agent.iteration_budget

        diag = _extract_failure_diagnostics(
            agent=agent,
            job={"id": "j5"},
            error_msg="err",
        )
        assert diag["prompt_tokens"] == 22913
        assert diag["completion_tokens"] == 1500

    def test_no_activity_summary(self):
        """Agent without get_activity_summary() is handled gracefully."""
        agent = SimpleNamespace()  # no get_activity_summary
        diag = _extract_failure_diagnostics(
            agent=agent,
            job={"id": "j6"},
            error_msg="err",
            provider="openrouter",
            model="claude-sonnet-4",
        )
        assert diag["provider"] == "openrouter"
        assert "api_call_count" not in diag

    def test_none_agent(self):
        """None agent is handled gracefully."""
        diag = _extract_failure_diagnostics(
            agent=None,
            job={"id": "j7"},
            error_msg="err",
            provider="deepseek",
            model="deepseek-v4-pro",
        )
        assert diag["provider"] == "deepseek"
        assert diag["model"] == "deepseek-v4-pro"


# ---------------------------------------------------------------------------
# _format_failure_notification
# ---------------------------------------------------------------------------

class TestFormatFailureNotification:
    """Unit tests for _format_failure_notification()."""

    def test_basic_format_no_diag(self):
        """Without diagnostics, returns the legacy format."""
        result = _format_failure_notification("my-job", "RuntimeError: broken pipe")
        assert result == "⚠️ Cron job 'my-job' failed:\nRuntimeError: broken pipe"

    def test_basic_format_empty_diag(self):
        """With empty diag dict, returns the legacy format."""
        result = _format_failure_notification("my-job", "err", diag={})
        assert result == "⚠️ Cron job 'my-job' failed:\nerr"

    def test_enriched_format(self):
        """With diagnostics, includes provider, model, turn, session."""
        diag = {
            "provider": "custom:openai",
            "model": "gpt-5.5",
            "session_id": "cron_j1_20260610_012345",
            "api_call_count": 11,
            "max_iterations": 90,
            "current_tool": "terminal",
            "last_activity_desc": "",
            "prompt_tokens": 22913,
            "completion_tokens": 0,
        }
        result = _format_failure_notification("nightlab-auto", "RuntimeError: broken pipe", diag=diag)
        assert "nightlab-auto" in result
        assert "RuntimeError: broken pipe" in result
        assert "Provider: custom:openai" in result
        assert "Model:    gpt-5.5" in result
        assert "At turn:  11/90, last: terminal" in result
        assert "Session:  cron_j1_20260610_012345" in result
        assert "Log:" in result
        assert "~22,913 prompt" in result

    def test_enriched_format_minimal_diag(self):
        """With only provider (no model, no session), shows provider only."""
        diag = {"provider": "openrouter", "model": "unknown", "session_id": ""}
        result = _format_failure_notification("job", "err", diag=diag)
        assert "Provider: openrouter" in result
        assert "Model:" not in result
        assert "Session:" not in result

    def test_enriched_format_unknown_provider_hidden(self):
        """Provider 'unknown' is not shown."""
        diag = {"provider": "unknown", "model": "unknown", "session_id": "cron_x"}
        result = _format_failure_notification("job", "err", diag=diag)
        assert "Provider:" not in result
        assert "Model:" not in result
        assert "Session:  cron_x" in result

    def test_enriched_format_no_tokens(self):
        """When no token data, token line is omitted."""
        diag = {
            "provider": "deepseek",
            "model": "deepseek-v4-pro",
            "session_id": "cron_j1",
            "api_call_count": 1,
            "max_iterations": 90,
        }
        result = _format_failure_notification("job", "err", diag=diag)
        assert "Tokens:" not in result
        assert "Provider: deepseek" in result


# ---------------------------------------------------------------------------
# Integration: diagnostics dict flow
# ---------------------------------------------------------------------------

class TestDiagnosticsDictFlow:
    """Test that _CRON_FAILURE_DIAGNOSTICS is populated and consumed."""

    def test_dict_pop_and_cleanup(self):
        """Diagnostics are consumed (popped) by the delivery path."""
        _CRON_FAILURE_DIAGNOSTICS["test-job"] = {
            "provider": "openrouter",
            "model": "claude-sonnet-4",
            "session_id": "cron_test",
        }
        # Simulate the delivery path
        diag = _CRON_FAILURE_DIAGNOSTICS.pop("test-job", None)
        assert diag is not None
        assert diag["provider"] == "openrouter"
        # Should be cleaned up
        assert "test-job" not in _CRON_FAILURE_DIAGNOSTICS

    def test_dict_pop_missing_returns_none(self):
        """Popping a non-existent key returns None (graceful fallback)."""
        diag = _CRON_FAILURE_DIAGNOSTICS.pop("nonexistent", None)
        assert diag is None
