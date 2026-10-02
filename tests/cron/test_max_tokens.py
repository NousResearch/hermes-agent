"""Regression coverage for per-job cron output-token caps."""

import pytest

from cron.jobs import create_job, update_job


@pytest.fixture()
def tmp_cron_dir(tmp_path, monkeypatch):
    """Redirect cron storage to a temp directory."""
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
    return tmp_path


def test_max_tokens_round_trips_through_create_and_update(tmp_cron_dir):
    job = create_job(prompt="bounded task", schedule="every 1h", max_tokens=4096)
    assert job["max_tokens"] == 4096

    updated = update_job(job["id"], {"max_tokens": 2048})
    assert updated["max_tokens"] == 2048


def test_max_tokens_rejects_non_positive_values(tmp_cron_dir):
    with pytest.raises(ValueError, match="max_tokens must be a positive integer"):
        create_job(prompt="invalid", schedule="every 1h", max_tokens=0)

    job = create_job(prompt="valid", schedule="every 1h")
    with pytest.raises(ValueError, match="max_tokens must be a positive integer"):
        update_job(job["id"], {"max_tokens": -1})
