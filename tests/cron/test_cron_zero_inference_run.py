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


def test_guard_is_wired_into_the_run_job_success_path():
    """The helper must actually gate run_job's success return, not just
    exist — pin the call site so a refactor can't silently drop it."""
    import inspect

    import cron.scheduler as scheduler

    source = inspect.getsource(scheduler.run_job)
    assert "_zero_inference_failure_reason(result)" in source
    # It must run BEFORE the success tuple is built.
    guard_at = source.index("_zero_inference_failure_reason(result)")
    success_at = source.index("return True, output, final_response, None")
    assert guard_at < success_at
