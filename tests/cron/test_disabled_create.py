"""Regression tests for atomic disabled cron creation."""

import argparse

import pytest

from cron.jobs import create_job, get_due_jobs, get_job
from hermes_cli.subcommands.cron import build_cron_parser


@pytest.fixture()
def tmp_cron_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.LASTGOOD_FILE", tmp_path / "cron" / "jobs.json.lastgood")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
    return tmp_path


def test_create_disabled_is_persisted_paused_and_never_due(tmp_cron_dir):
    job = create_job(
        prompt="safe disabled job",
        schedule="every 1m",
        deliver="local",
        enabled=False,
    )

    assert job["enabled"] is False
    assert job["state"] == "paused"
    stored = get_job(job["id"])
    assert stored is not None
    assert stored["enabled"] is False
    assert stored["state"] == "paused"
    assert job["id"] not in {candidate["id"] for candidate in get_due_jobs()}


def test_create_defaults_to_enabled_for_existing_callers(tmp_cron_dir):
    job = create_job(prompt="existing default", schedule="every 1m", deliver="local")

    assert job["enabled"] is True
    assert job["state"] == "scheduled"


def test_cron_create_parser_accepts_disabled_flag():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    build_cron_parser(subparsers, cmd_cron=lambda args: 0)

    args = parser.parse_args(["cron", "create", "every 1m", "safe job", "--disabled"])

    assert args.disabled is True
