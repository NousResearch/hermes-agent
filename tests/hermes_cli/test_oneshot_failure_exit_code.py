"""Tests for hermes -z / run_oneshot failure exit codes (#92502, #74659).

Unattended callers (scripts, CI, fleet runners) rely on exit code 0 indicating
success. When run_conversation returns failed=True (e.g. content policy refusal,
rate limit exhaustion, agent failure) or partial=True, run_oneshot must return
exit code 2 even when final_response contains a non-empty refusal/error message.
"""

from __future__ import annotations

import io
from unittest.mock import patch

import hermes_cli.oneshot as oneshot


def test_oneshot_returns_2_on_failed_result_with_non_empty_response():
    """When the turn fails (e.g. content policy refusal), exit code must be 2 even with output."""
    refusal_text = "I cannot fulfill this request due to content policy."
    fake_result = {
        "final_response": refusal_text,
        "completed": False,
        "failed": True,
        "error": "content_policy_blocked: refusal",
    }

    stdout_buf = io.StringIO()
    with patch("hermes_cli.oneshot._run_agent", return_value=(refusal_text, fake_result)), \
         patch("sys.stdout", stdout_buf):
        rc = oneshot.run_oneshot("test prompt")

    assert rc == 2
    assert refusal_text in stdout_buf.getvalue()


def test_oneshot_returns_2_on_failed_result_with_empty_response():
    """When the turn fails with empty response, exit code must be 2."""
    fake_result = {
        "final_response": "",
        "completed": False,
        "failed": True,
        "error": "api_error: HTTP 500",
    }

    stdout_buf = io.StringIO()
    with patch("hermes_cli.oneshot._run_agent", return_value=("", fake_result)), \
         patch("sys.stdout", stdout_buf):
        rc = oneshot.run_oneshot("test prompt")

    assert rc == 2


def test_oneshot_returns_2_on_partial_result_with_non_empty_response():
    """When the turn is partial (e.g. tool loop interrupted), exit code must be 2."""
    partial_text = "Task was interrupted before completion."
    fake_result = {
        "final_response": partial_text,
        "completed": False,
        "partial": True,
    }

    stdout_buf = io.StringIO()
    with patch("hermes_cli.oneshot._run_agent", return_value=(partial_text, fake_result)), \
         patch("sys.stdout", stdout_buf):
        rc = oneshot.run_oneshot("test prompt")

    assert rc == 2
    assert partial_text in stdout_buf.getvalue()


def test_oneshot_returns_0_on_success_with_response():
    """Successful turn with non-empty response returns 0."""
    success_text = "Everything succeeded."
    fake_result = {
        "final_response": success_text,
        "completed": True,
        "failed": False,
        "partial": False,
    }

    stdout_buf = io.StringIO()
    with patch("hermes_cli.oneshot._run_agent", return_value=(success_text, fake_result)), \
         patch("sys.stdout", stdout_buf):
        rc = oneshot.run_oneshot("test prompt")

    assert rc == 0
    assert success_text in stdout_buf.getvalue()


def test_oneshot_returns_1_on_empty_response_when_not_failed():
    """When no response is produced without explicit failure dict, return 1."""
    fake_result = {
        "final_response": "",
        "completed": True,
        "failed": False,
        "partial": False,
    }

    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    with patch("hermes_cli.oneshot._run_agent", return_value=("", fake_result)), \
         patch("sys.stdout", stdout_buf), \
         patch("sys.stderr", stderr_buf):
        rc = oneshot.run_oneshot("test prompt")

    assert rc == 1
    assert "no final response was produced" in stderr_buf.getvalue()
