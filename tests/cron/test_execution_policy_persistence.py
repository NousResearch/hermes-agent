import json

from cron.jobs import create_job, get_job, mark_job_run
from tools.cronjob_tools import cronjob


def test_create_job_persists_execution_policy(tmp_path, monkeypatch):
    monkeypatch.setattr("cron.jobs.HERMES_DIR", tmp_path)
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")

    job = create_job(
        prompt="say hi",
        schedule="every 30m",
        execution_policy={"mode": "enforce", "deny_tools": ["terminal"]},
    )

    stored = get_job(job["id"])
    assert stored["execution_policy"]["deny_tools"] == ["terminal"]
    assert stored["last_policy_audit_events"] == []


def test_mark_job_run_persists_policy_audit_events(tmp_path, monkeypatch):
    monkeypatch.setattr("cron.jobs.HERMES_DIR", tmp_path)
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
    job = create_job(prompt="say hi", schedule="every 30m")
    event = {"action": "block", "tool_name": "skill_manage", "code": "tool_denied"}

    mark_job_run(job["id"], True, policy_audit_events=[event])

    stored = get_job(job["id"])
    assert stored["last_policy_audit_events"] == [event]


def test_cronjob_tool_accepts_execution_policy(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
    monkeypatch.setattr("cron.jobs.HERMES_DIR", tmp_path)
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")

    raw = cronjob(
        action="create",
        prompt="say hi",
        schedule="every 30m",
        execution_policy={"mode": "enforce", "deny_tools": ["terminal"]},
    )
    data = json.loads(raw)

    assert data["success"] is True
    assert data["job"]["execution_policy"]["deny_tools"] == ["terminal"]
