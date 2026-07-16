"""Focused tests for per-job cron reasoning effort overrides."""

import json
from argparse import Namespace

import pytest


@pytest.fixture()
def cron_reasoning_env(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    (home / "cron" / "output").mkdir(parents=True)
    (home / "scripts").mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))

    import cron.jobs as jobs

    monkeypatch.setattr(jobs, "HERMES_DIR", home)
    monkeypatch.setattr(jobs, "CRON_DIR", home / "cron")
    monkeypatch.setattr(jobs, "JOBS_FILE", home / "cron" / "jobs.json")
    monkeypatch.setattr(jobs, "OUTPUT_DIR", home / "cron" / "output")
    return home


def test_job_reasoning_effort_persists_and_omission_preserves_shape(cron_reasoning_env):
    from cron.jobs import create_job, get_job, list_jobs, update_job

    omitted = create_job(prompt="Default", schedule="every 1h")
    assert "reasoning_effort" not in omitted
    assert "reasoning_effort" not in get_job(omitted["id"])

    pinned = create_job(
        prompt="Think hard",
        schedule="every 1h",
        reasoning_effort="high",
    )
    assert pinned["reasoning_effort"] == "high"
    assert get_job(pinned["id"])["reasoning_effort"] == "high"
    assert next(j for j in list_jobs() if j["id"] == pinned["id"])["reasoning_effort"] == "high"

    updated = update_job(pinned["id"], {"reasoning_effort": "low"})
    assert updated["reasoning_effort"] == "low"
    assert get_job(pinned["id"])["reasoning_effort"] == "low"


def test_invalid_reasoning_effort_is_rejected_on_create_and_update(cron_reasoning_env):
    from cron.jobs import create_job, update_job

    with pytest.raises(ValueError, match="reasoning_effort"):
        create_job(prompt="Bad", schedule="every 1h", reasoning_effort="turbo")

    job = create_job(prompt="Valid", schedule="every 1h")
    with pytest.raises(ValueError, match="reasoning_effort"):
        update_job(job["id"], {"reasoning_effort": "turbo"})


@pytest.mark.parametrize("value", ["none", "false", "disabled", "minimal", "low", "medium", "high", "xhigh", "max"])
def test_canonical_reasoning_effort_values_are_accepted(cron_reasoning_env, value):
    from cron.jobs import create_job

    job = create_job(prompt="Valid", schedule="every 1h", reasoning_effort=value)
    assert job["reasoning_effort"] == value


def test_scheduler_reasoning_precedence_and_unset_fallback():
    from cron.scheduler import _resolve_cron_reasoning_config

    assert _resolve_cron_reasoning_config(
        {"reasoning_effort": "high"}, {"agent": {"reasoning_effort": "low"}}
    ) == {"enabled": True, "effort": "high"}
    assert _resolve_cron_reasoning_config(
        {}, {"agent": {"reasoning_effort": "low"}}
    ) == {"enabled": True, "effort": "low"}
    assert _resolve_cron_reasoning_config({}, {}) is None


def test_cronjob_tool_plumbs_reasoning_effort_and_exposes_it(cron_reasoning_env, monkeypatch):
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")
    from tools.cronjob_tools import cronjob

    created = json.loads(cronjob(
        action="create",
        prompt="Think hard",
        schedule="every 1h",
        reasoning_effort="high",
    ))
    assert created["success"] is True
    assert created["job"]["reasoning_effort"] == "high"

    updated = json.loads(cronjob(
        action="update",
        job_id=created["job_id"],
        reasoning_effort="none",
    ))
    assert updated["success"] is True
    assert updated["job"]["reasoning_effort"] == "none"

    listed = json.loads(cronjob(action="list"))
    assert listed["jobs"][0]["reasoning_effort"] == "none"


def test_cronjob_tool_rejects_invalid_reasoning_effort(cron_reasoning_env, monkeypatch):
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")
    from tools.cronjob_tools import cronjob

    result = json.loads(cronjob(
        action="create",
        prompt="Bad",
        schedule="every 1h",
        reasoning_effort="turbo",
    ))
    assert result["success"] is False
    assert "reasoning_effort" in result["error"]


def test_cron_cli_parser_and_handlers_plumb_reasoning_effort(cron_reasoning_env, monkeypatch):
    import argparse
    from hermes_cli import cron as cron_cli
    from hermes_cli.subcommands.cron import build_cron_parser

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    build_cron_parser(subparsers, cmd_cron=lambda args: None)

    create_args = parser.parse_args([
        "cron", "create", "every 1h", "Think hard", "--reasoning-effort", "high"
    ])
    assert create_args.reasoning_effort == "high"

    captured = {}

    def fake_api(**kwargs):
        captured.update(kwargs)
        return {
            "success": True,
            "job_id": "job-1",
            "name": "Think hard",
            "schedule": "every 60m",
            "next_run_at": "later",
            "job": {"job_id": "job-1", "name": "Think hard", "schedule": "every 60m"},
        }

    monkeypatch.setattr(cron_cli, "_cron_api", fake_api)
    assert cron_cli.cron_create(create_args) == 0
    assert captured["reasoning_effort"] == "high"

    edit_args = parser.parse_args([
        "cron", "edit", "job-1", "--reasoning-effort", "none"
    ])
    assert edit_args.reasoning_effort == "none"


def test_cron_cli_rejects_invalid_reasoning_effort(cron_reasoning_env, monkeypatch, capsys):
    from hermes_cli import cron as cron_cli

    monkeypatch.setattr("hermes_cli.cron.resolve_job_ref", lambda job_id: None, raising=False)
    args = Namespace(
        cron_command="create",
        schedule="every 1h",
        prompt="Bad",
        name=None,
        deliver=None,
        repeat=None,
        skill=None,
        skills=None,
        script=None,
        workdir=None,
        no_agent=False,
        reasoning_effort="turbo",
    )
    assert cron_cli.cron_create(args) == 1
    assert "reasoning_effort" in capsys.readouterr().out


def test_cronjob_schema_advertises_reasoning_effort():
    from tools.cronjob_tools import CRONJOB_SCHEMA

    prop = CRONJOB_SCHEMA["parameters"]["properties"]["reasoning_effort"]
    assert prop["type"] == "string"
    assert "none" in prop["description"]
    assert "high" in prop["description"]
