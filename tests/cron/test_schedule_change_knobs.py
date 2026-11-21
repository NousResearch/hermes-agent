"""A cron-only knob must not outlive a schedule change away from cron.

``grace_minutes`` only applies to cron schedules, but a job switched to an interval kept it. The
leftover value still made catch-up skip missed runs, while ``hermes cron doctor`` called it inert.
"""

import pytest

from cron.jobs import create_job, get_job, parse_schedule, update_job
from cron.missed_runs import job_catches_up


@pytest.fixture(autouse=True)
def _cron_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")


@pytest.mark.parametrize("grace", [0, 45])
def test_leaving_cron_drops_the_grace_override(grace):
    job = create_job(prompt="x", schedule="0 9 * * *", grace_minutes=grace)

    update_job(job["id"], {"schedule": "every 2h"})

    stored = get_job(job["id"])
    assert stored["schedule"]["kind"] == "interval"
    assert "grace_minutes" not in stored
    plain = get_job(create_job(prompt="y", schedule="every 2h")["id"])
    assert job_catches_up(stored) == job_catches_up(plain)


def test_leaving_cron_while_changing_the_timezone_is_not_refused():
    """A timezone in the same update re-checks the knobs; the stale grace must not fail that check."""
    job = create_job(prompt="x", schedule="0 9 * * *", grace_minutes=30)

    update_job(job["id"], {"schedule": parse_schedule("every 2h"), "timezone": "UTC"})

    stored = get_job(job["id"])
    assert stored["timezone"] == "UTC"
    assert "grace_minutes" not in stored


def test_a_cron_to_cron_change_keeps_the_grace_override():
    job = create_job(prompt="x", schedule="0 9 * * *", grace_minutes=45)

    update_job(job["id"], {"schedule": "30 8 * * *"})

    assert get_job(job["id"])["grace_minutes"] == 45
