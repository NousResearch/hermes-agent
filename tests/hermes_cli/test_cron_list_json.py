"""`hermes cron list --json` emits the job records as parseable JSON."""
import json

from hermes_cli import cron as cron_cli


def test_cron_list_json_emits_records(monkeypatch, capsys):
    jobs = [{"id": "a1", "enabled": True, "schedule": "0 * * * *"},
            {"id": "b2", "enabled": False, "schedule": "5 * * * *"}]
    monkeypatch.setattr("cron.jobs.list_jobs", lambda include_disabled=True: list(jobs))
    monkeypatch.setattr("cron.jobs.effective_job_state", lambda job: "active" if job.get("enabled") else "disabled")
    cron_cli.cron_list(show_all=True, as_json=True)
    assert [j["id"] for j in json.loads(capsys.readouterr().out)] == ["a1", "b2"]


def test_cron_list_json_respects_filter_and_empty(monkeypatch, capsys):
    monkeypatch.setattr("cron.jobs.list_jobs", lambda include_disabled=True: [{"id": "b2", "enabled": False}])
    monkeypatch.setattr("cron.jobs.effective_job_state", lambda job: "disabled")
    cron_cli.cron_list(show_all=False, as_json=True)
    assert json.loads(capsys.readouterr().out) == []
