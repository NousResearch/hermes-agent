"""Bounded-stop invariant: a recurring deliver=local cron job must carry a
reachable finite repeat bound, otherwise its stop condition is unsatisfiable
(the terminal gate only releases on a delivered platform receipt, which local
delivery never produces) and it fires forever."""
import pytest

from cron.jobs import (
    create_job,
    update_job,
    mark_job_run,
    LOCAL_RECURRING_MAX_REPEAT,
)


@pytest.fixture()
def tmp_cron_dir(tmp_path, monkeypatch):
    """Redirect cron storage to a temp directory."""
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
    return tmp_path


def test_omitted_repeat_on_recurring_local_defaults_to_ceiling(tmp_cron_dir):
    job = create_job(prompt="p", schedule="every 5m")  # local, repeat omitted
    assert job["repeat"]["times"] == LOCAL_RECURRING_MAX_REPEAT


def test_explicit_forever_on_recurring_local_is_refused(tmp_cron_dir):
    with pytest.raises(ValueError, match="reachable"):
        create_job(prompt="p", schedule="every 5m", repeat="forever")


def test_above_ceiling_on_recurring_local_is_refused(tmp_cron_dir):
    with pytest.raises(ValueError, match="at most"):
        create_job(prompt="p", schedule="every 5m", repeat=LOCAL_RECURRING_MAX_REPEAT + 1)


def test_bounded_local_is_allowed(tmp_cron_dir):
    job = create_job(prompt="p", schedule="every 5m", repeat=10)
    assert job["repeat"]["times"] == 10


def test_external_channel_forever_is_allowed(tmp_cron_dir):
    job = create_job(prompt="p", schedule="every 5m", repeat="forever", deliver="discord")
    assert job["repeat"]["times"] is None  # forever is fine on an external channel


def test_once_is_unaffected(tmp_cron_dir):
    job = create_job(prompt="p", schedule="every monday 9am", repeat="once")
    assert job["repeat"]["times"] == 1


def test_update_door_flip_to_local_unbounded_is_refused(tmp_cron_dir):
    job = create_job(prompt="p", schedule="every 5m", repeat="forever", deliver="discord")
    with pytest.raises(ValueError, match="reachable"):
        update_job(job["id"], {"deliver": "local"})
