"""A paused profile-definition import must not retain cron-only grace on an interval."""

from cron import jobs
from cron.job_definition import import_job_definitions
from cron.missed_runs import job_catches_up


def test_paused_profile_import_leaving_cron_restores_interval_catch_up(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    with jobs.use_cron_store(tmp_path / "cron"):
        job = jobs.create_job(
            prompt="report", schedule="0 9 * * *", grace_minutes=0, paused=True)

        # A previously authored cron definition is changed to an interval while its old
        # grace override remains in the shipped record. Paused imports skip schedule refresh.
        import_job_definitions({job["id"]: {
            "prompt": "report", "schedule": "every 2h", "grace_minutes": 0,
        }}, paused_reason="Profile import")

        stored = jobs.get_job(job["id"])
        assert stored["state"] == "paused"
        assert stored["schedule"]["kind"] == "interval"
        assert "grace_minutes" not in stored

        jobs.resume_job(job["id"])
        assert job_catches_up(jobs.get_job(job["id"])) is True
