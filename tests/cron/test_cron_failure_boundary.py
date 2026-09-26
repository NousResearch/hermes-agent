"""Security-boundary regressions for cron failure diagnostics (#123264)."""

from __future__ import annotations

import os

from cron import executions, incidents, jobs, scheduler
from cron.scheduler_failure_copy import blocked_config_notice, generic_failure_notice
from tools.cronjob_job_args import _format_job
from tools import cronjob_tools


HOST_PATH = "/Users/alice/private/leakcanary-7f3a/app.py"
KNOWN_TOKEN = "sk-leakcanary0123456789abcdefghijklmnop"
UNKNOWN_TOKEN = "acme_live_leakcanary9f8e7d6c5b4a3928"
STDERR_MARKER = "KeyError: 'leakcanary-stderr-91c2'"
RAW_FAILURE = f"{STDERR_MARKER} at {HOST_PATH}; known={KNOWN_TOKEN}; unknown={UNKNOWN_TOKEN}"
MARKERS = (HOST_PATH, KNOWN_TOKEN, UNKNOWN_TOKEN, STDERR_MARKER, "leakcanary")


def _assert_public(value) -> None:
    text = str(value)
    for marker in MARKERS:
        assert marker not in text


def test_failure_notices_never_copy_arbitrary_detail():
    generic = generic_failure_notice("nightly", "job-1", RAW_FAILURE)
    blocked = blocked_config_notice("nightly", RAW_FAILURE)

    assert "job_failed" in generic
    _assert_public(generic)
    _assert_public(blocked)


def test_historical_job_failure_fields_are_sanitized_at_read_boundary(tmp_path):
    with jobs.use_cron_store(tmp_path):
        job = jobs.create_job(prompt="check", schedule="every 1h")
        stored = jobs.load_jobs()
        stored[0]["last_error"] = RAW_FAILURE
        stored[0]["last_delivery_error"] = RAW_FAILURE
        stored[0]["last_fire_error"] = {"at": "2026-09-26T00:00:00+00:00", "detail": RAW_FAILURE}
        jobs.save_jobs(stored)

        loaded = jobs.get_job(job["id"])
        listed = jobs.list_jobs(include_disabled=True)[0]
        formatted = _format_job(loaded)

    assert loaded["last_error"] == "job_failed"
    assert loaded["last_delivery_error"] == "job_failed"
    assert loaded["last_fire_error"]["detail"] == "job_failed"
    _assert_public(loaded)
    _assert_public(listed)
    _assert_public(formatted)


def test_historical_execution_error_is_sanitized_in_job_listing(tmp_path, monkeypatch):
    monkeypatch.setattr(executions, "EXECUTIONS_FILE", tmp_path / "cron" / "executions.db")
    with jobs.use_cron_store(tmp_path):
        job = jobs.create_job(prompt="check", schedule="every 1h")
        row = executions.create_execution(job["id"], source="test")
        with executions._transaction() as conn:
            conn.execute(
                "UPDATE executions SET status='failed', error=? WHERE id=?",
                (RAW_FAILURE, row["id"]),
            )

        direct = executions.get_execution(row["id"])
        listed = jobs.list_jobs(include_disabled=True)[0]

    assert direct["error"] == "job_failed"
    assert listed["latest_execution"]["error"] == "job_failed"
    _assert_public(direct)
    _assert_public(listed)


def test_run_pipeline_keeps_raw_failure_only_in_private_output(monkeypatch):
    saved = []
    delivered = []
    marked = []
    finished = []

    monkeypatch.setattr(scheduler, "create_execution", lambda *_a, **_kw: {"id": "exec-safe"})
    monkeypatch.setattr(scheduler, "claim_dispatch", lambda _job_id: True)
    monkeypatch.setattr(scheduler, "mark_execution_running", lambda _execution_id: {})
    monkeypatch.setattr(
        scheduler,
        "run_job",
        lambda *_a, **_kw: (False, "script stdout", "", RAW_FAILURE),
    )
    monkeypatch.setattr(
        scheduler,
        "save_job_output",
        lambda _job_id, output: saved.append(output) or "/private/output.md",
    )
    monkeypatch.setattr(
        scheduler,
        "_deliver_result",
        lambda _job, content, **_kw: delivered.append(content) or None,
    )
    monkeypatch.setattr(
        scheduler,
        "mark_job_run",
        lambda *args, **kwargs: marked.append((args, kwargs)) or True,
    )
    monkeypatch.setattr(
        scheduler,
        "finish_execution",
        lambda *args, **kwargs: finished.append((args, kwargs)),
    )
    monkeypatch.setattr(
        scheduler,
        "_upsert_incident_for_failure",
        lambda *_a, **_kw: (False, "incidentsafe"),
    )
    monkeypatch.setattr(scheduler, "load_config", lambda: {})

    assert scheduler.run_one_job(
        {"id": "job-safe", "name": "safe boundary", "deliver": "telegram", "no_agent": True}
    ) is True

    assert len(saved) == 1
    assert RAW_FAILURE in saved[0]
    assert len(delivered) == 1
    _assert_public(delivered[0])
    assert "script_failed" in delivered[0]
    assert marked[0][0][2] == "script_failed (incident incidentsafe)"
    assert finished[0][1]["error"] == "script_failed (incident incidentsafe)"
    _assert_public(marked)
    _assert_public(finished)


def test_private_run_output_preserves_diagnostics_and_is_owner_only(tmp_path):
    with jobs.use_cron_store(tmp_path):
        output_path = jobs.save_job_output("job-private", RAW_FAILURE)

    assert output_path.read_text(encoding="utf-8") == RAW_FAILURE
    if os.name != "nt":
        assert output_path.stat().st_mode & 0o777 == 0o600


def test_failed_background_completion_never_reinjects_private_excerpt(monkeypatch):
    monkeypatch.setattr(cronjob_tools, "get_job", lambda _job_id: {})
    monkeypatch.setattr(
        cronjob_tools,
        "_latest_job_output_excerpt",
        lambda _job_id: RAW_FAILURE,
    )

    result = cronjob_tools._manual_run_completion(
        {"success": False, "error": "job_failed"},
        "job-safe",
        "safe boundary",
        "telegram",
        0.0,
    )

    assert result["error"] == "job_failed"
    assert "JOB OUTPUT" not in result["summary"]
    _assert_public(result)


def test_incident_exposes_closed_label_and_no_output_path(tmp_path, monkeypatch):
    monkeypatch.setattr(incidents, "EXECUTIONS_FILE", tmp_path / "cron" / "executions.db")

    incident_id, _ = incidents.upsert_incident(
        "job-safe", RAW_FAILURE, output_file="/Users/alice/private/output.md"
    )
    record = incidents.get_incident(incident_id)

    assert record["error"] == "job_failed"
    assert record["output_file"] is None
    _assert_public(record)


def test_profile_output_store_does_not_bleed_across_a_b_a(tmp_path):
    profile_a = tmp_path / "profile-a"
    profile_b = tmp_path / "profile-b"

    with jobs.use_cron_store(profile_a):
        first_a = jobs.save_job_output("same-job", "profile-a-first")
    with jobs.use_cron_store(profile_b):
        only_b = jobs.save_job_output("same-job", "profile-b")
    with jobs.use_cron_store(profile_a):
        second_a = jobs.save_job_output("same-job", "profile-a-second")

    assert profile_a.resolve() in first_a.resolve().parents
    assert profile_b.resolve() in only_b.resolve().parents
    assert profile_a.resolve() in second_a.resolve().parents
    assert "profile-b" not in second_a.read_text(encoding="utf-8")
