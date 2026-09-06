"""Lease renewal must never use the jobs lock's degraded write mode."""
import contextlib
from datetime import datetime, timedelta
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("failure", ["timeout", "missing", "oserror"])
def test_heartbeat_fails_closed_without_cross_process_lock(tmp_path, monkeypatch, failure, nested):
    import cron.jobs as jobs

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    job = jobs.create_job(prompt="x", schedule="every 5m")
    assert jobs.claim_job_for_fire(job["id"])
    claim = jobs.get_job(job["id"])["fire_claim"]
    now = datetime.fromisoformat(claim["at"]) + timedelta(seconds=30)
    monkeypatch.setattr(jobs, "_hermes_now", lambda: now)
    before = (tmp_path / "cron" / "jobs.json").read_bytes()

    if failure == "missing":
        monkeypatch.setattr(jobs, "fcntl", None)
        monkeypatch.setattr(jobs, "msvcrt", None)
    else:
        acquire = (
            Mock(return_value=False) if failure == "timeout"
            else Mock(side_effect=OSError("lock failed"))
        )
        monkeypatch.setattr(jobs, "_acquire_flock", acquire)
    load = Mock(wraps=jobs.load_jobs)
    monkeypatch.setattr(jobs, "load_jobs", load)

    error = RuntimeError if nested or failure == "missing" else (
        TimeoutError if failure == "timeout" else OSError
    )
    with jobs._jobs_lock() if nested else contextlib.nullcontext():
        with pytest.raises(error):
            jobs.heartbeat_fire_claim(job["id"], expected_owner=claim["by"])
    load.assert_not_called()
    assert (tmp_path / "cron" / "jobs.json").read_bytes() == before
    # An unsuccessful acquisition must close its descriptor, including an OSError.
    if failure != "missing":
        assert all(call.args[0].closed for call in acquire.call_args_list)
    assert jobs._jobs_lock_state.depth == 0


@pytest.mark.parametrize("strict_outer", [False, True])
def test_heartbeat_reuses_real_outer_lock_and_preserves_owner(tmp_path, monkeypatch, strict_outer):
    import cron.jobs as jobs

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    job = jobs.create_job(prompt="x", schedule="every 5m")
    assert jobs.claim_job_for_fire(job["id"])
    claim = jobs.get_job(job["id"])["fire_claim"]
    now = datetime.fromisoformat(claim["at"]) + timedelta(seconds=30)
    monkeypatch.setattr(jobs, "_hermes_now", lambda: now)
    with jobs._jobs_lock(strict=strict_outer):
        acquire = Mock(side_effect=AssertionError("nested lock must reuse actual flock"))
        monkeypatch.setattr(jobs, "_acquire_flock", acquire)
        assert jobs.heartbeat_fire_claim(job["id"], expected_owner=claim["by"])
        refreshed = jobs.get_job(job["id"])["fire_claim"]
        assert refreshed["at"] != claim["at"]
        assert refreshed["by"] == claim["by"]
        assert not jobs.heartbeat_fire_claim(job["id"], expected_owner="stale-owner")
        assert jobs.get_job(job["id"])["fire_claim"] == refreshed
        acquire.assert_not_called()
    assert jobs._jobs_lock_state.depth == 0
