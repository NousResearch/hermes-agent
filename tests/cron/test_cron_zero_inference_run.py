"""Zero-inference cron runs must not be recorded as successes (#100180).

A manual `cronjob(action=run)` fire that was interrupted before reaching the
model returned `result: ok` / `last_status: ok` with `API calls: 0` and a
transcript truncated mid-first-tool-call. Operators monitoring `last_status`
saw green while the scheduled maintenance silently never ran — the inverse of
#70427 (empty *successful* runs mis-recorded as failures).

These tests exercise the guard helper directly: `run_job` cannot be driven
end-to-end in a bare test HERMES_HOME (no provider, real retry/backoff), so
the classification logic is what's pinned.
"""

from __future__ import annotations

import pytest

from cron.scheduler import _zero_inference_failure_reason


def test_zero_api_calls_is_a_failure():
    """api_calls=0 means the model was never reached: must fail the run."""
    reason = _zero_inference_failure_reason({
        "final_response": "Starting weekly Mnemosyne maintenance. Step 1 —",
        "api_calls": 0,
        "completed": True,
        "failed": False,
    })
    assert reason, "a zero-inference run must not be reported as success"
    assert "zero inference calls" in reason.lower()


@pytest.mark.parametrize("api_calls", [1, 2, 17])
def test_nonzero_api_calls_is_a_success(api_calls):
    """A run that reached the model stays on the success path."""
    assert _zero_inference_failure_reason({
        "final_response": "Full report body",
        "api_calls": api_calls,
        "completed": True,
        "failed": False,
    }) == ""


def test_missing_api_calls_field_is_not_treated_as_zero():
    """Absent api_calls (older result shapes / test doubles) must not fail
    the run — the guard fires only on an explicit 0."""
    assert _zero_inference_failure_reason({
        "final_response": "ok",
        "completed": True,
        "failed": False,
    }) == ""


@pytest.mark.parametrize("value", [None, "0", "", 1.5, True, False])
def test_non_integer_api_calls_is_not_treated_as_zero(value):
    """Only a real int 0 counts. A bool is rejected explicitly: True/False
    are ints in Python and would otherwise classify False as zero."""
    assert _zero_inference_failure_reason({
        "final_response": "ok",
        "api_calls": value,
        "completed": True,
        "failed": False,
    }) == ""


def test_negative_api_calls_is_a_failure():
    """A negative count is as impossible as zero — treat it as unreached."""
    assert _zero_inference_failure_reason({"api_calls": -1}) != ""


def test_run_job_fails_when_api_calls_is_zero(tmp_path):
    """run_job must fail the run and record failure when api_calls is 0."""
    from unittest.mock import MagicMock, patch

    from cron.scheduler import run_job

    job = {"id": "test-job", "name": "test", "prompt": "hello"}
    fake_db = MagicMock()
    fake_db.get_compression_tip.side_effect = lambda session_id: session_id

    with patch("cron.scheduler._hermes_home", tmp_path), \
         patch("cron.scheduler._resolve_origin", return_value=None), \
         patch("hermes_cli.env_loader.load_hermes_dotenv"), \
         patch("hermes_cli.env_loader.reset_secret_source_cache"), \
         patch("hermes_state.SessionDB", return_value=fake_db), \
         patch(
             "hermes_cli.runtime_provider.resolve_runtime_provider",
             return_value={
                 "api_key": "test-key",
                 "base_url": "https://example.invalid/v1",
                 "provider": "openrouter",
                 "api_mode": "chat_completions",
             },
         ), \
         patch("run_agent.AIAgent") as mock_agent_cls:
        mock_agent = MagicMock()
        mock_agent.run_conversation.return_value = {
            "final_response": "interrupted mid-turn",
            "api_calls": 0,
        }
        mock_agent_cls.return_value = mock_agent

        success, output, final_response, error = run_job(job)

    assert success is False
    assert error is not None
    assert "zero inference calls" in str(error).lower()
