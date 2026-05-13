"""Tests for cron delivery metadata stored on scheduled jobs."""

from __future__ import annotations

import json


def test_create_job_stores_delivery_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")

    from cron.jobs import create_job, get_job

    job = create_job(
        prompt="Summarize repo changes",
        schedule="every 1h",
        delivery_mode="slack_thread",
        thread_title_template="{name} (job_id: {job_id})",
    )

    stored = get_job(job["id"])
    assert stored["delivery_mode"] == "slack_thread"
    assert stored["thread_title_template"] == "{name} (job_id: {job_id})"


def test_cronjob_tool_roundtrips_delivery_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")

    from tools.cronjob_tools import cronjob

    created = json.loads(
        cronjob(
            action="create",
            prompt="Summarize repo changes",
            schedule="every 1h",
            delivery_mode="slack_thread",
            thread_title_template="{name} (job_id: {job_id})",
        )
    )

    assert created["success"] is True
    assert created["job"]["delivery_mode"] == "slack_thread"
    assert created["job"]["thread_title_template"] == "{name} (job_id: {job_id})"

    listed = json.loads(cronjob(action="list"))
    assert listed["jobs"][0]["delivery_mode"] == "slack_thread"
    assert listed["jobs"][0]["thread_title_template"] == "{name} (job_id: {job_id})"
