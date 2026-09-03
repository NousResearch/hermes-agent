"""Tests for the kanban rate-limit exit-code helper.

A kanban worker whose run failed purely on a provider quota wall must exit
with ``KANBAN_RATE_LIMIT_EXIT_CODE`` (75) so the dispatcher requeues the task
as ``rate_limited`` instead of counting it as a protocol violation and
crash-looping it. Regression for CR-20260902-01: the plain ``chat -q`` path
(normal kanban workers) previously never applied this exit code.
"""

import pytest

from cli import _kanban_rate_limit_exit_code
from hermes_cli.kanban_db import KANBAN_RATE_LIMIT_EXIT_CODE


def _run_result(**overrides):
    result = {
        "final_response": "",
        "failed": True,
        "failure_reason": "billing",
    }
    result.update(overrides)
    return result


class TestKanbanRateLimitExitCode:
    @pytest.mark.parametrize(
        "reason", ["billing", "rate_limit"]
    )
    def test_kanban_quota_wall_returns_75(self, monkeypatch, reason):
        monkeypatch.setenv("HERMES_KANBAN_TASK", "t_test")
        assert _kanban_rate_limit_exit_code(
            _run_result(failure_reason=reason)
        ) == KANBAN_RATE_LIMIT_EXIT_CODE

    def test_non_quota_failure_returns_none(self, monkeypatch):
        monkeypatch.setenv("HERMES_KANBAN_TASK", "t_test")
        # A generic failure is not a rate-limit/billing wall; keep existing
        # exit behaviour (do not override to 75).
        assert (
            _kanban_rate_limit_exit_code(
                _run_result(failure_reason="context_overflow")
            )
            is None
        )

    def test_non_failed_run_returns_none(self, monkeypatch):
        monkeypatch.setenv("HERMES_KANBAN_TASK", "t_test")
        assert _kanban_rate_limit_exit_code(
            _run_result(failed=False, failure_reason="billing")
        ) is None

    def test_non_kanban_returns_none(self, monkeypatch):
        monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
        assert _kanban_rate_limit_exit_code(
            _run_result(failure_reason="billing")
        ) is None

    def test_non_dict_returns_none(self, monkeypatch):
        monkeypatch.setenv("HERMES_KANBAN_TASK", "t_test")
        assert _kanban_rate_limit_exit_code("not a dict") is None
        assert _kanban_rate_limit_exit_code(None) is None
