"""Tests for ``hermes -z`` exit-code contract — #74659.

The oneshot exit code is the contract unattended callers (cron, systemd,
subprocess) branch on. Before #74659 it routinely lied: an HTTP 402
billing wall (and a provider safety refusal) set ``failed=True`` on the
run result but ALSO populated ``final_response`` with a guidance string,
so the old ``if failed and not response: return 2`` branch never fired
and the process exited 0. Pipelines read that as success.

These tests pin the new contract:

  0 → success (response present, agent did not flag failure)
  1 → generic crash / no response at all
  2 → agent-flagged failure (billing, content_filter, …) — distinguishes
      "retry the provider" (2) from "investigate" (1)
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from hermes_cli.oneshot import run_oneshot


@pytest.fixture
def capture_streams(monkeypatch, capsys):
    """Collect stdout/stderr written by run_oneshot.

    The function redirects stdio internally (to devnull) for the agent
    call, then writes the final response / diagnostics to the *real*
    stdout/stderr. ``capsys`` captures what lands on the real streams
    because the redirect is via contextmanager, not a swap of ``sys.__*__``.
    """
    # Force the early-validation branches off so the test reaches the
    # exit-code logic. The agent itself is mocked below, so none of these
    # touch a real provider.
    monkeypatch.setenv("HERMES_INFERENCE_MODEL", "test-model")
    capsys.readouterr()  # drain any setup noise
    yield capsys


def _mock_agent_result(response: str | None, **result_fields) -> tuple[str, dict]:
    """Build the (response, result) tuple ``_run_agent`` returns."""
    base = {"failed": False, "partial": False, "completed": True, "final_response": response or ""}
    base.update(result_fields)
    return (response or ""), base


class TestSuccessPath:
    def test_clean_response_exits_zero(self, capture_streams):
        with patch(
            "hermes_cli.oneshot._run_agent",
            return_value=_mock_agent_result("done"),
        ):
            rc = run_oneshot("do something")
        assert rc == 0
        assert capture_streams.readouterr().out.strip() == "done"

    def test_response_without_trailing_newline_gets_one(self, capture_streams):
        with patch(
            "hermes_cli.oneshot._run_agent",
            return_value=_mock_agent_result("no newline"),
        ):
            rc = run_oneshot("p")
        assert rc == 0
        assert capture_streams.readouterr().out.endswith("\n")


class TestBillingFailure:
    def test_http_402_billing_exits_two_not_zero(self, capture_streams):
        # Mirrors agent/conversation_loop.py's billing-wall return shape:
        # final_response is a guidance string AND failed is True.
        with patch(
            "hermes_cli.oneshot._run_agent",
            return_value=_mock_agent_result(
                "Billing or credits exhausted: add credits and retry.",
                failed=True,
                completed=False,
                failure_reason="billing",
            ),
        ):
            rc = run_oneshot("p")
        assert rc == 2, "billing wall must not exit 0 — #74659"
        # The guidance text still goes to stdout so users see the message.
        assert "Billing or credits exhausted" in capture_streams.readouterr().out

    def test_billing_failure_writes_stderr_marker(self, capture_streams):
        with patch(
            "hermes_cli.oneshot._run_agent",
            return_value=_mock_agent_result(
                "Billing exhausted",
                failed=True,
                failure_reason="billing",
            ),
        ):
            rc = run_oneshot("p")
        assert rc == 2
        err = capture_streams.readouterr().err
        assert "turn ended without completing the request" in err
        assert "failure_reason=billing" in err


class TestContentFilterRefusal:
    def test_safety_refusal_exits_two(self, capture_streams):
        # Mirrors the content-policy-blocked return shape from
        # agent/conversation_loop.py (final_response is the refusal
        # explanation, failed is True, failure_reason content_filter).
        with patch(
            "hermes_cli.oneshot._run_agent",
            return_value=_mock_agent_result(
                "⚠️  The model declined to respond to this request (safety refusal).",
                failed=True,
                completed=False,
                failure_reason="content_filter",
            ),
        ):
            rc = run_oneshot("p")
        assert rc == 2


class TestNoResponsePath:
    def test_empty_response_and_not_failed_exits_one(self, capture_streams):
        with patch(
            "hermes_cli.oneshot._run_agent",
            return_value=_mock_agent_result(""),
        ):
            rc = run_oneshot("p")
        assert rc == 1
        err = capture_streams.readouterr().err
        assert "no final response" in err

    def test_failed_and_empty_exits_two(self, capture_streams):
        with patch(
            "hermes_cli.oneshot._run_agent",
            return_value=_mock_agent_result("", failed=True),
        ):
            rc = run_oneshot("p")
        assert rc == 2


class TestPartialPath:
    def test_partial_with_response_still_exits_zero(self, capture_streams):
        # A partial run that still produced output is "the turn did
        # something" — keeping this at 0 preserves callers that interpret
        # nonzero as "abort the pipeline" and would rather keep partial
        # work than discard it.
        with patch(
            "hermes_cli.oneshot._run_agent",
            return_value=_mock_agent_result("partial output", partial=True),
        ):
            rc = run_oneshot("p")
        assert rc == 0

    def test_partial_with_no_response_exits_two(self, capture_streams):
        with patch(
            "hermes_cli.oneshot._run_agent",
            return_value=_mock_agent_result("", partial=True),
        ):
            rc = run_oneshot("p")
        assert rc == 2


class TestUsageFileFailureReason:
    def test_usage_file_records_failure_reason_for_billing(self, tmp_path, capture_streams):
        path = tmp_path / "usage.json"
        with patch(
            "hermes_cli.oneshot._run_agent",
            return_value=_mock_agent_result(
                "Billing exhausted",
                failed=True,
                failure_reason="billing",
            ),
        ):
            rc = run_oneshot("p", usage_file=str(path))
        assert rc == 2
        report = json.loads(path.read_text())
        assert report["failed"] is True
        assert report["failure_reason"] == "billing"

    def test_usage_file_failure_reason_null_on_success(self, tmp_path, capture_streams):
        path = tmp_path / "usage.json"
        with patch(
            "hermes_cli.oneshot._run_agent",
            return_value=_mock_agent_result("done"),
        ):
            rc = run_oneshot("p", usage_file=str(path))
        assert rc == 0
        report = json.loads(path.read_text())
        assert report["failed"] is False
        assert "failure_reason" in report
        assert report["failure_reason"] is None
