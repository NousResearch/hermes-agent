"""A run that failed on a spent quota tells API clients when the quota lifts."""

from gateway.platforms.api_server_runs import terminal_run_status


def test_failed_run_carries_the_quota_reset_time():
    status, fields = terminal_run_status({"failed": True, "failure_resets_at": 1793581479})
    assert status == "failed"
    assert fields["failure_resets_at"] == 1793581479.0


def test_reset_time_only_rides_an_unfinished_run_and_must_be_a_number():
    _, completed = terminal_run_status({"completed": True, "failure_resets_at": 1793581479})
    _, garbage = terminal_run_status({"failed": True, "failure_resets_at": "soon"})
    _, flag = terminal_run_status({"failed": True, "failure_resets_at": True})
    _, missing = terminal_run_status({"failed": True})
    assert "failure_resets_at" not in completed
    assert "failure_resets_at" not in garbage
    assert "failure_resets_at" not in missing
    assert "failure_resets_at" not in flag
