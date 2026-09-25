"""Regression coverage for scheduler-owned occurrence metadata in cron scripts."""

from cron import scheduler_script


def test_run_job_script_applies_explicit_job_env(tmp_path, monkeypatch):
    monkeypatch.setattr(scheduler_script._sched, "_get_hermes_home", lambda: tmp_path)
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir(parents=True)
    script = scripts_dir / "occurrence.py"
    script.write_text(
        "import os\n"
        "print(os.environ['HERMES_CRON_JOB_ID'])\n"
        "print(os.environ['HERMES_CRON_OCCURRENCE_AT'])\n",
        encoding="utf-8",
    )

    success, output = scheduler_script._run_job_script(
        "occurrence.py",
        job_env={
            "HERMES_CRON_JOB_ID": "job-123",
            "HERMES_CRON_OCCURRENCE_AT": "2026-09-16T14:00:00+00:00",
        },
    )

    assert success is True
    assert output.splitlines() == ["job-123", "2026-09-16T14:00:00+00:00"]


def test_occurrence_env_is_limited_to_no_agent_jobs(monkeypatch):
    observed = []

    def fake_run(script_path, workdir=None, cancel_event=None, job_env=None):
        observed.append(job_env)
        return True, ""

    monkeypatch.setattr(scheduler_script, "_run_job_script", fake_run)

    base = {
        "id": "job-123",
        "next_run_at": "2026-09-16T14:00:00+00:00",
        "schedule": {"kind": "cron"},
    }
    scheduler_script._run_job_script_with_claim_heartbeat(
        {**base, "no_agent": True}, "occurrence.py"
    )
    scheduler_script._run_job_script_with_claim_heartbeat(
        {**base, "no_agent": False}, "occurrence.py"
    )

    assert observed == [
        {
            "HERMES_CRON_JOB_ID": "job-123",
            "HERMES_CRON_OCCURRENCE_AT": "2026-09-16T14:00:00+00:00",
        },
        None,
    ]
