"""Tests for syncing distribution-owned cron job templates."""

import importlib
import json

import pytest

from hermes_cli.cron import cron_list


def _load_modules(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    import cron.jobs as jobs

    jobs = importlib.reload(jobs)
    job_templates = importlib.import_module("cron.job_templates")
    job_templates = importlib.reload(job_templates)
    return hermes_home, jobs, job_templates


def _write_template(hermes_home, filename, **template):
    templates_dir = hermes_home / "cron" / "templates"
    templates_dir.mkdir(parents=True, exist_ok=True)
    (templates_dir / filename).write_text(
        json.dumps(template, indent=2),
        encoding="utf-8",
    )


def test_sync_cron_templates_creates_job(tmp_path, monkeypatch):
    hermes_home, jobs, job_templates = _load_modules(tmp_path, monkeypatch)
    _write_template(
        hermes_home,
        "repo-self-improvement.json",
        template_key="repo-self-improvement",
        version="1",
        name="Repo self improvement",
        schedule="every 1h",
        prompt="Review the repository and suggest one improvement.",
        deliver="slack",
        delivery_mode="slack_thread",
        thread_title_template="Repo improvement - {date}",
    )

    result = job_templates.sync_cron_templates()

    assert result["created"] == ["repo-self-improvement"]
    assert result["updated"] == []
    stored = jobs.list_jobs(include_disabled=True)
    assert len(stored) == 1
    assert stored[0]["template_key"] == "repo-self-improvement"
    assert stored[0]["template_version"] == "1"
    assert stored[0]["delivery_mode"] == "slack_thread"
    assert stored[0]["thread_title_template"] == "Repo improvement - {date}"


def test_sync_cron_templates_expands_profile_workdir(tmp_path, monkeypatch):
    hermes_home, jobs, job_templates = _load_modules(tmp_path, monkeypatch)
    _write_template(
        hermes_home,
        "repo-self-improvement.json",
        template_key="repo-self-improvement",
        version="1",
        name="Repo self improvement",
        schedule="every 1h",
        prompt="Review the repository and suggest one improvement.",
        workdir="${HERMES_HOME}",
    )

    result = job_templates.sync_cron_templates()

    assert result["created"] == ["repo-self-improvement"]
    stored = jobs.list_jobs(include_disabled=True)
    assert stored[0]["workdir"] == str(hermes_home.resolve())


def test_sync_cron_templates_updates_prompt_but_preserves_runtime_state(tmp_path, monkeypatch):
    hermes_home, jobs, job_templates = _load_modules(tmp_path, monkeypatch)
    existing = jobs.create_job(
        prompt="old",
        schedule="every 1h",
        name="Existing template job",
        deliver="slack",
        template_key="repo-self-improvement",
        template_version="1",
    )
    jobs.update_job(
        existing["id"],
        {
            "last_run_at": "2026-05-12T09:00:00+00:00",
            "last_status": "ok",
            "last_error": "keep this",
            "last_delivery_error": "delivery retry later",
            "repeat": {"times": 5, "completed": 2},
            "enabled": False,
            "state": "paused",
            "paused_at": "2026-05-12T09:05:00+00:00",
            "paused_reason": "maintenance",
            "next_run_at": "2026-05-13T09:00:00+00:00",
        },
    )
    before = jobs.get_job(existing["id"])
    _write_template(
        hermes_home,
        "repo-self-improvement.json",
        template_key="repo-self-improvement",
        version="2",
        name="Existing template job",
        schedule="every 1h",
        prompt="new",
        deliver="slack",
        delivery_mode="slack_thread",
        thread_title_template="Repo improvement - {date}",
    )

    result = job_templates.sync_cron_templates()

    assert result["created"] == []
    assert result["updated"] == ["repo-self-improvement"]
    stored = jobs.get_job(existing["id"])
    assert stored["id"] == existing["id"]
    assert stored["created_at"] == before["created_at"]
    assert stored["next_run_at"] == before["next_run_at"]
    assert stored["prompt"] == "new"
    assert stored["template_version"] == "2"
    assert stored["delivery_mode"] == "slack_thread"
    assert stored["last_run_at"] == "2026-05-12T09:00:00+00:00"
    assert stored["last_status"] == "ok"
    assert stored["last_error"] == "keep this"
    assert stored["last_delivery_error"] == "delivery retry later"
    assert stored["repeat"] == {"times": 5, "completed": 2}
    assert stored["enabled"] is False
    assert stored["state"] == "paused"
    assert stored["paused_at"] == "2026-05-12T09:05:00+00:00"
    assert stored["paused_reason"] == "maintenance"


def test_sync_cron_templates_preserves_missing_next_run_when_schedule_unchanged(tmp_path, monkeypatch):
    hermes_home, jobs, job_templates = _load_modules(tmp_path, monkeypatch)
    existing = jobs.create_job(
        prompt="old",
        schedule="every 1h",
        name="Existing template job",
        deliver="slack",
        template_key="repo-self-improvement",
        template_version="1",
    )
    stored_jobs = jobs.load_jobs()
    stored_jobs[0]["next_run_at"] = None
    jobs.save_jobs(stored_jobs)
    _write_template(
        hermes_home,
        "repo-self-improvement.json",
        template_key="repo-self-improvement",
        version="2",
        name="Existing template job",
        schedule="every 1h",
        prompt="new",
        deliver="slack",
    )

    job_templates.sync_cron_templates()

    stored = jobs.get_job(existing["id"])
    assert stored["next_run_at"] is None
    assert stored["prompt"] == "new"


def test_sync_cron_templates_relative_one_shot_does_not_shift_on_second_sync(tmp_path, monkeypatch):
    hermes_home, jobs, job_templates = _load_modules(tmp_path, monkeypatch)
    _write_template(
        hermes_home,
        "repo-self-improvement.json",
        template_key="repo-self-improvement",
        version="1",
        name="Repo self improvement",
        schedule="30m",
        prompt="Review the repository and suggest one improvement.",
        deliver="slack",
    )
    job_templates.sync_cron_templates()
    before = jobs.list_jobs(include_disabled=True)[0]
    first_schedule = before["schedule"]
    first_next_run_at = before["next_run_at"]

    job_templates.sync_cron_templates()

    after = jobs.list_jobs(include_disabled=True)[0]
    assert after["schedule"] == first_schedule
    assert after["next_run_at"] == first_next_run_at


def test_sync_cron_templates_legacy_relative_one_shot_detects_changed_duration(tmp_path, monkeypatch):
    hermes_home, jobs, job_templates = _load_modules(tmp_path, monkeypatch)
    existing = jobs.create_job(
        prompt="old",
        schedule="30m",
        name="Legacy one-shot template job",
        template_key="repo-self-improvement",
        template_version="1",
    )
    before = jobs.get_job(existing["id"])
    assert before["schedule_display"] == "once in 30m"
    assert "template_schedule" not in before

    _write_template(
        hermes_home,
        "repo-self-improvement.json",
        template_key="repo-self-improvement",
        version="2",
        name="Legacy one-shot template job",
        schedule="60m",
        prompt="new",
    )

    job_templates.sync_cron_templates()

    stored = jobs.get_job(existing["id"])
    assert stored["schedule_display"] == "once in 60m"
    assert stored["schedule"]["run_at"] != before["schedule"]["run_at"]
    assert stored["next_run_at"] == stored["schedule"]["run_at"]
    assert stored["template_schedule"] == "60m"


def test_sync_cron_templates_duplicate_keys_raise_before_mutating_jobs(tmp_path, monkeypatch):
    hermes_home, jobs, job_templates = _load_modules(tmp_path, monkeypatch)
    _write_template(
        hermes_home,
        "one.json",
        template_key="repo-self-improvement",
        version="1",
        name="First",
        schedule="every 1h",
        prompt="first",
    )
    _write_template(
        hermes_home,
        "two.json",
        template_key="repo-self-improvement",
        version="1",
        name="Second",
        schedule="every 2h",
        prompt="second",
    )

    with pytest.raises(ValueError, match="Duplicate cron template_key"):
        job_templates.sync_cron_templates()

    assert jobs.list_jobs(include_disabled=True) == []


def test_sync_cron_templates_rejects_no_agent_without_script_on_update(tmp_path, monkeypatch):
    hermes_home, jobs, job_templates = _load_modules(tmp_path, monkeypatch)
    existing = jobs.create_job(
        prompt="old",
        schedule="every 1h",
        name="Existing template job",
        template_key="repo-self-improvement",
        template_version="1",
    )
    _write_template(
        hermes_home,
        "repo-self-improvement.json",
        template_key="repo-self-improvement",
        version="2",
        name="Existing template job",
        schedule="every 1h",
        prompt="new",
        no_agent=True,
    )

    with pytest.raises(ValueError, match="no_agent=True requires a script"):
        job_templates.sync_cron_templates()

    stored = jobs.get_job(existing["id"])
    assert stored["no_agent"] is False
    assert stored["prompt"] == "old"


def test_cron_list_shows_thread_title_template(tmp_path, monkeypatch, capsys):
    _hermes_home, jobs, _job_templates = _load_modules(tmp_path, monkeypatch)
    jobs.create_job(
        prompt="report",
        schedule="every 1h",
        name="Threaded report",
        deliver="slack",
        delivery_mode="slack_thread",
        thread_title_template="Repo improvement - {date}",
        template_key="repo-self-improvement",
        template_version="1",
    )
    monkeypatch.setattr("hermes_cli.gateway.find_gateway_pids", lambda: [1234])

    cron_list(show_all=True)

    out = capsys.readouterr().out
    assert "Thread title:" in out
    assert "Repo improvement - {date}" in out
