"""Regression for #114309: a past scheduled slot is not an upcoming run."""

from argparse import Namespace
from datetime import timedelta

import pytest


@pytest.mark.parametrize("surface", ["status", "list"])
def test_past_slot_is_labelled_overdue_without_changing_store(tmp_path, monkeypatch, capsys, surface):
    from cron import jobs
    from hermes_cli import cron
    from hermes_time import now

    monkeypatch.setattr(cron, "_builtin_gateway_liveness", lambda: True)
    monkeypatch.setattr("hermes_cli.gateway.find_gateway_pids", lambda: [])
    monkeypatch.setattr("hermes_cli.gateway.named_profile_served_by_running_multiplexer", lambda: False)
    monkeypatch.setattr("gateway.status.is_gateway_runtime_lock_active", lambda: False)
    with jobs.use_cron_store(tmp_path):
        jobs.create_job(prompt="Report", schedule="every 1h")
        records = jobs.load_jobs()
        scheduled = (now() - timedelta(hours=7)).isoformat()
        records[0]["next_run_at"] = scheduled
        jobs.save_jobs(records)
        store = jobs._current_cron_store().jobs_file
        before = store.read_bytes()
        assert cron.cron_command(Namespace(cron_command=surface, all=False)) == 0
        output = capsys.readouterr().out
        assert "Overdue since:" in output
        assert scheduled in output
        assert "Next run:" not in output
        assert store.read_bytes() == before


@pytest.mark.parametrize("surface", ["status", "list"])
def test_future_slot_remains_upcoming(tmp_path, monkeypatch, capsys, surface):
    from cron import jobs
    from hermes_cli import cron
    from hermes_time import now

    monkeypatch.setattr(cron, "_builtin_gateway_liveness", lambda: True)
    monkeypatch.setattr("hermes_cli.gateway.find_gateway_pids", lambda: [])
    monkeypatch.setattr("hermes_cli.gateway.named_profile_served_by_running_multiplexer", lambda: False)
    monkeypatch.setattr("gateway.status.is_gateway_runtime_lock_active", lambda: False)
    with jobs.use_cron_store(tmp_path):
        jobs.create_job(prompt="Report", schedule="every 1h")
        records = jobs.load_jobs()
        scheduled = (now() + timedelta(hours=7)).isoformat()
        records[0]["next_run_at"] = scheduled
        jobs.save_jobs(records)
        assert cron.cron_command(Namespace(cron_command=surface, all=False)) == 0
        output = capsys.readouterr().out
        assert "Next run:" in output
        assert scheduled in output
        assert "Overdue since:" not in output

