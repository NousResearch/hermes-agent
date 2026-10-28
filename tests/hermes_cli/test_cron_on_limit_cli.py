"""``hermes cron create/edit --on-limit``: the per-job choice the usage-limit gate reads
(cron/quota_hold.py: ``wait`` runs once at the reset, ``skip`` drops the runs inside the window)."""

from __future__ import annotations

import argparse

import pytest

from cron.jobs import create_job, get_job, list_jobs
from hermes_cli.cron import cron_command
from hermes_cli.subcommands.cron import build_cron_parser


@pytest.fixture()
def tmp_cron_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
    monkeypatch.setattr("hermes_cli.cron._warn_if_gateway_not_running", lambda: None)
    return tmp_path


def _parse(*argv):
    parser = argparse.ArgumentParser(prog="hermes")
    build_cron_parser(parser.add_subparsers(dest="command"), cmd_cron=cron_command)
    return parser.parse_args(["cron", *argv])


def test_create_and_edit_set_on_limit(tmp_cron_dir, capsys):
    assert cron_command(_parse("create", "every 1h", "Daily report", "--on-limit", "skip")) in (0, None)
    (job,) = list_jobs(include_disabled=True)
    assert job["on_limit"] == "skip"
    assert "On usage limit: skip" in capsys.readouterr().out

    # --on-limit alone is a complete edit, not "no updates provided".
    assert cron_command(_parse("edit", job["id"], "--on-limit", "wait")) in (0, None)
    assert get_job(job["id"])["on_limit"] == "wait"
    assert "Updated job" in capsys.readouterr().out


def test_edit_without_on_limit_leaves_it_and_bad_values_are_refused(tmp_cron_dir):
    job = create_job(prompt="Daily report", schedule="every 1h")
    cron_command(_parse("edit", job["id"], "--name", "renamed"))
    assert "on_limit" not in get_job(job["id"])
    with pytest.raises(SystemExit):
        _parse("edit", job["id"], "--on-limit", "retry")


def test_a_refused_edit_leaves_on_limit_unchanged(tmp_cron_dir, capsys):
    """--on-limit rides the one validated update: a bad --schedule in the same edit changes nothing."""
    job = create_job(prompt="Daily report", schedule="every 1h")
    assert cron_command(_parse("edit", job["id"], "--on-limit", "skip", "--schedule", "not a schedule")) == 1
    assert "on_limit" not in get_job(job["id"])
    assert "Failed to update job" in capsys.readouterr().out
