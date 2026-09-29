"""Runtime import failures must be diagnosable and recover without a restart (#127182)."""

import builtins
import json
import logging
import sys
from datetime import datetime, timedelta, timezone

import pytest

from cron import jobs


@pytest.fixture
def failing_import(monkeypatch):
    real_import = builtins.__import__
    state = {"fail": True, "attempts": 0}

    def import_with_outage(name, *args, **kwargs):
        if name == "croniter" and args and isinstance(args[0], dict) and args[0].get("__name__") == jobs.__name__:
            state["attempts"] += 1
            if state["fail"]:
                raise ImportError("croniter runtime path unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(jobs, "croniter", None)
    monkeypatch.setattr(jobs, "HAS_CRONITER", None)
    monkeypatch.setattr(jobs, "_croniter_retry_at", 0.0, raising=False)
    monkeypatch.setattr(jobs, "_croniter_import_error", None, raising=False)
    monkeypatch.setattr(builtins, "__import__", import_with_outage)
    return state


def test_import_retries_after_backoff_and_logs_the_actual_runtime(failing_import, monkeypatch, caplog):
    schedule = {"kind": "cron", "expr": "0 8 * * *"}
    with caplog.at_level(logging.WARNING, logger="cron.jobs"):
        assert jobs.compute_next_run(schedule) is None
        assert jobs.compute_next_run(schedule) is None
    assert failing_import["attempts"] == 1
    assert "croniter runtime path unavailable" in caplog.text
    assert sys.executable in caplog.text
    assert sys.version.split()[0] in caplog.text
    assert len([r for r in caplog.records if r.name == "cron.jobs"]) == 1

    failing_import["fail"] = False
    monkeypatch.setattr(jobs, "_croniter_retry_at", 0.0)
    assert jobs.compute_next_run(schedule) is not None
    assert jobs.compute_next_run(schedule) is not None
    assert failing_import["attempts"] == 2  # successful imports remain cached


@pytest.mark.parametrize("entry", ["after_run", "missing_next_run"])
@pytest.mark.parametrize("recovery", ["tick", "edit", "resume"])
def test_persisted_jobs_show_scheduling_failure_and_self_heal(entry, recovery, failing_import, monkeypatch, capsys):
    from hermes_cli import cron as cron_cli
    from hermes_cli.cli_commands_mixin import CLICommandsMixin
    from tools.cronjob_tools import cronjob

    now = datetime(2026, 9, 29, 7, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(jobs, "_hermes_now", lambda: now)
    job = jobs.create_job(prompt="Report status.", schedule="every 1h")
    records = jobs.load_jobs()
    records[0].update(schedule={"kind": "cron", "expr": "0 8 * * *"},
                      last_status="error", last_error="previous provider timeout",
                      last_run_at=now.isoformat())
    if entry == "missing_next_run":
        records[0]["next_run_at"] = None
    jobs.save_jobs(records)
    if entry == "after_run":
        jobs.mark_job_run(job["id"], success=False, error="previous provider timeout")
    else:
        assert jobs.get_due_jobs() == []

    stored = jobs.get_job(job["id"])
    assert stored["enabled"] is True
    assert stored["state"] == "error"
    assert stored["next_run_at"] is None
    assert "croniter runtime path unavailable" in stored["schedule_error"]
    assert stored["last_error"] == "previous provider timeout"
    listed = json.loads(cronjob("list"))["jobs"][0]
    assert listed["schedule_error"] == stored["schedule_error"]
    monkeypatch.setattr(cron_cli, "_warn_if_gateway_not_running", lambda: None)
    cron_cli.cron_list()
    output = capsys.readouterr().out
    assert "[error]" in output and "croniter runtime path unavailable" in output
    CLICommandsMixin()._handle_cron_command("/cron list")
    assert "croniter runtime path unavailable" in capsys.readouterr().out

    failing_import["fail"] = False
    monkeypatch.setattr(jobs, "_croniter_retry_at", 0.0)
    if recovery == "edit":
        jobs.update_job(job["id"], {"schedule": "0 8 * * *"})
    elif recovery == "resume":
        jobs.resume_job(job["id"])
    else:
        assert jobs.get_due_jobs() == []
    recovered = jobs.get_job(job["id"])
    assert recovered["state"] == "scheduled"
    assert not recovered.get("schedule_error")
    assert recovered["last_error"] == "previous provider timeout"
    assert recovered["enabled"] is True
    now = datetime.fromisoformat(recovered["next_run_at"]) + timedelta(seconds=1)
    assert [j["id"] for j in jobs.get_due_jobs()] == [job["id"]]
