"""Regression tests for concise cron failure delivery summaries."""

from cron.scheduler import _summarize_cron_failure_for_delivery


JOB = {"id": "job-1", "name": "nightly-report"}


def summarize(error: str) -> str:
    return _summarize_cron_failure_for_delivery(JOB, error)


def test_local_script_timeout_is_not_labelled_as_provider_failure():
    result = summarize("Script timed out after 120s: /tmp/foo.py")

    assert "local script timeout" in result
    assert "provider timeout" not in result
    assert "Fallback chain" not in result
    assert "/tmp/foo.py" not in result


def test_decimal_local_script_timeout_is_classified_locally():
    result = summarize("Script timed out after 1.5s: /tmp/foo.py")

    assert "local script timeout" in result
    assert "provider timeout" not in result


def test_local_timeout_path_containing_429_does_not_become_rate_limit():
    result = summarize("Script timed out after 30s: /tmp/429-report.py")

    assert "local script timeout" in result
    assert "provider rate limit" not in result


def test_local_timeout_path_containing_rate_limit_stays_local():
    result = summarize("Script timed out after 30s: /tmp/rate-limit-report.py")

    assert "local script timeout" in result
    assert "provider rate limit" not in result


def test_readtimeout_remains_a_provider_timeout():
    result = summarize("ReadTimeout: request timed out")

    assert "provider timeout" in result
    assert "Fallback chain" in result


def test_normal_script_failure_remains_generic_and_sanitized():
    result = summarize("Script execution failed: permission denied")

    assert result == "⚠️ Cron 'nightly-report' failed: Script execution failed: permission denied"
    assert "provider timeout" not in result


def test_incidental_timeout_wording_does_not_match_local_script_envelope():
    result = summarize("Script reported timeout statistics")

    assert "local script timeout" not in result
    assert "provider timeout" in result
