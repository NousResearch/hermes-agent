"""`hermes cron doctor` run from a cron job must not report that job (#133135)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.fixture
def cron_env(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    (hermes_home / "cron" / "output").mkdir(parents=True)
    (hermes_home / "scripts").mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("HERMES_CRON_JOB_ID", raising=False)

    import cron.jobs as jobs_mod
    monkeypatch.setattr(jobs_mod, "HERMES_DIR", hermes_home)
    monkeypatch.setattr(jobs_mod, "CRON_DIR", hermes_home / "cron")
    monkeypatch.setattr(jobs_mod, "JOBS_FILE", hermes_home / "cron" / "jobs.json")
    monkeypatch.setattr(jobs_mod, "OUTPUT_DIR", hermes_home / "cron" / "output")
    return hermes_home


def _failed_job(name: str, error: str) -> dict:
    from cron import jobs

    job = jobs.create_job(prompt=name, schedule="every 1h", name=name)
    jobs.mark_job_run(job["id"], success=False, error=error)
    return job


def test_doctor_skips_the_job_that_invoked_it(cron_env, capsys, monkeypatch):
    from hermes_cli.cron import cron_doctor

    wrapper = _failed_job("doctor-wrapper", "Cron doctor found 1 issue(s) ... previous run")
    monkeypatch.setenv("HERMES_CRON_JOB_ID", wrapper["id"])

    assert cron_doctor() == 0
    output = capsys.readouterr().out
    assert "previous run" not in output
    assert "no issues" in output


def test_doctor_still_reports_other_failed_jobs_when_invoked_from_cron(cron_env, capsys, monkeypatch):
    from hermes_cli.cron import cron_doctor

    wrapper = _failed_job("doctor-wrapper", "wrapper-own-error")
    other = _failed_job("nightly-backup", "backup-broke")
    monkeypatch.setenv("HERMES_CRON_JOB_ID", wrapper["id"])

    assert cron_doctor() == 1
    output = capsys.readouterr().out
    assert "backup-broke" in output
    assert other["id"] in output
    assert "wrapper-own-error" not in output


def test_doctor_reports_every_job_outside_cron(cron_env, capsys):
    from hermes_cli.cron import cron_doctor

    _failed_job("doctor-wrapper", "wrapper-own-error")

    assert cron_doctor() == 1
    assert "wrapper-own-error" in capsys.readouterr().out


def test_scheduled_script_sees_its_own_job_id(cron_env):
    from cron.scheduler_script import _run_job_script_with_claim_heartbeat

    (cron_env / "scripts" / "probe.py").write_text(
        'import os\nprint(os.environ.get("HERMES_CRON_JOB_ID", "<unset>"))\n'
    )
    job = {"id": "abc123", "schedule": {"kind": "interval", "minutes": 60}}

    success, output = _run_job_script_with_claim_heartbeat(job, "probe.py")
    assert success is True
    assert output == "abc123"
