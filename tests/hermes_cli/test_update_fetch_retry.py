"""A transient failed fetch does not abort an otherwise healthy update (#139086)."""

import subprocess

from hermes_cli import update_cmd


def test_fetch_retries_transient_failures_before_returning_success():
    attempts = []
    delays = []

    def fetch():
        attempts.append(None)
        return subprocess.CompletedProcess([], 1 if len(attempts) < 3 else 0, stderr="temporary failure")

    result = update_cmd._fetch_with_transient_retry(fetch, sleep=delays.append)

    assert result.returncode == 0
    assert len(attempts) == 3
    assert delays == [5, 10]


def test_fetch_returns_the_last_failure_after_the_bounded_retries():
    attempts = []
    delays = []

    def fetch():
        attempts.append(None)
        return subprocess.CompletedProcess([], 1, stderr=f"failure {len(attempts)}")

    result = update_cmd._fetch_with_transient_retry(fetch, sleep=delays.append)

    assert result.returncode == 1
    assert result.stderr == "failure 4"
    assert len(attempts) == 4
    assert delays == [5, 10, 15]
