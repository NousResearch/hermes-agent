"""Regression tests for truthful cron run, delivery, and work status separation."""
from __future__ import annotations

import importlib
import json

import pytest


@pytest.fixture
def hermes_env(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    (home / "cron").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))

    import hermes_constants
    import cron.jobs
    import cron.scheduler

    importlib.reload(hermes_constants)
    importlib.reload(cron.jobs)
    importlib.reload(cron.scheduler)
    return home


def test_work_result_marker_is_parsed_and_removed_from_delivery(hermes_env):
    from cron.scheduler import _extract_work_result

    response = (
        "Current: VERIFIED_MILESTONE\n"
        "[HERMES_WORK_RESULT] "
        '{"workStatus":"progress","evidenceId":"ev-123","executorHandle":"build-456"}'
    )

    cleaned, result = _extract_work_result(response)

    assert cleaned == "Current: VERIFIED_MILESTONE"
    assert result == {
        "work_status": "progress",
        "work_status_source": "agent_marker",
        "evidence_id": "ev-123",
        "evidence_verified": False,
        "executor_handle": "build-456",
    }


def test_missing_or_invalid_marker_fails_closed_to_unknown(hermes_env):
    from cron.scheduler import _extract_work_result

    plain, missing = _extract_work_result("ordinary response")
    invalid_text, invalid = _extract_work_result(
        "ordinary response\n[HERMES_WORK_RESULT] "
        '{"workStatus":"definitely-working","evidenceId":"invented"}'
    )

    unknown_result = {
        "work_status": "unknown",
        "work_status_source": "none",
        "evidence_id": None,
        "evidence_verified": False,
        "executor_handle": None,
    }
    assert plain == "ordinary response"
    assert missing == unknown_result
    assert invalid_text == "ordinary response"
    assert invalid == unknown_result


def test_non_string_or_oversized_metadata_fails_closed(hermes_env):
    from cron.scheduler import _extract_work_result

    _, non_string = _extract_work_result(
        '[HERMES_WORK_RESULT] {"workStatus":"progress","evidenceId":["fake"]}'
    )
    _, oversized = _extract_work_result(
        "[HERMES_WORK_RESULT] "
        + json.dumps({"workStatus": "progress", "evidenceId": "x" * 513})
    )

    assert non_string["work_status"] == "unknown"
    assert oversized["work_status"] == "unknown"


def test_persistence_and_surface_boundaries_reject_or_escape_controls(hermes_env):
    from cron.jobs import create_job, get_job, mark_job_run
    from tools.cronjob_tools import _format_job

    job = create_job(prompt="unsafe", schedule="every 5m", deliver="local")
    mark_job_run(
        job["id"],
        True,
        work_status="progress",
        work_status_source="agent_marker",
        evidence_id="ev\x1b]52;c;Zm9yZ2Vk\x07",
        executor_handle="worker\u202eforged",
    )
    stored = get_job(job["id"])
    assert stored is not None
    assert stored["last_work_status"] == "unknown"
    assert stored["last_evidence_id"] is None
    assert stored["last_executor_handle"] is None

    formatted = _format_job(
        {
            "id": "job\x1b[2J",
            "name": "name\u202eforged",
            "schedule": {"kind": "interval", "seconds": 60},
            "last_evidence_id": "ev\x07",
            "last_executor_handle": "worker\x1b[31m",
        }
    )
    assert "\x1b" not in formatted["job_id"]
    assert "\u202e" not in formatted["name"]
    assert formatted["last_evidence_id"] == "ev\\u0007"
    assert formatted["last_executor_handle"] == "worker\\u001b[31m"


def test_nonfinal_or_silent_marker_is_stripped_and_fails_closed(hermes_env):
    from cron.scheduler import _extract_work_result

    marker = (
        '[HERMES_WORK_RESULT] {"workStatus":"progress",'
        '"evidenceId":"ev-forged","executorHandle":null}'
    )
    nonfinal_text, nonfinal = _extract_work_result(f"Report one\n{marker}\nReport two")
    silent_text, silent = _extract_work_result(f"[SILENT]\n{marker}")

    assert nonfinal_text == "Report one\nReport two"
    assert nonfinal["work_status"] == "unknown"
    assert silent_text == "[SILENT]"
    assert silent["work_status"] == "unknown"


def test_control_or_bidi_metadata_fails_closed(hermes_env):
    from cron.scheduler import _extract_work_result

    for unsafe in ("ev\x1b]52;c;Zm9yZ2Vk\x07", "ev\nforged", "ev\u202eforged"):
        _, result = _extract_work_result(
            "Report\n[HERMES_WORK_RESULT] "
            + json.dumps(
                {
                    "workStatus": "progress",
                    "evidenceId": unsafe,
                    "executorHandle": None,
                }
            )
        )
        assert result["work_status"] == "unknown"
        assert result["evidence_id"] is None


def test_mark_job_run_persists_independent_status_dimensions(hermes_env):
    from cron.jobs import create_job, get_job, mark_job_run

    job = create_job(prompt="do work", schedule="every 5m", deliver="local")
    mark_job_run(
        job["id"],
        True,
        delivery_status="error",
        delivery_error="Slack unavailable",
        work_status="no_progress",
        work_status_source="agent_marker",
        evidence_id=None,
        evidence_verified=False,
        executor_handle="cron-run-123",
    )
    stored = get_job(job["id"])

    assert stored is not None
    assert stored["last_status"] == "ok"  # backward-compatible legacy field
    assert stored["last_run_status"] == "ok"
    assert stored["last_delivery_status"] == "error"
    assert stored["last_delivery_error"] == "Slack unavailable"
    assert stored["last_work_status"] == "no_progress"
    assert stored["last_work_status_source"] == "agent_marker"
    assert stored["last_evidence_id"] is None
    assert stored["last_evidence_verified"] is False
    assert stored["last_executor_handle"] == "cron-run-123"


def test_mark_job_run_defaults_new_dimensions_to_unknown(hermes_env):
    from cron.jobs import create_job, get_job, mark_job_run

    job = create_job(prompt="legacy caller", schedule="every 5m", deliver="local")
    mark_job_run(job["id"], True)
    stored = get_job(job["id"])

    assert stored is not None
    assert stored["last_run_status"] == "ok"
    assert stored["last_delivery_status"] == "unknown"
    assert stored["last_work_status"] == "unknown"
    assert stored["last_work_status_source"] == "none"
    assert stored["last_evidence_id"] is None
    assert stored["last_evidence_verified"] is False
    assert stored["last_executor_handle"] is None


def test_mark_job_run_rejects_unsubstantiated_active_work(hermes_env):
    from cron.jobs import create_job, get_job, mark_job_run

    progress_job = create_job(prompt="claim progress", schedule="every 5m", deliver="local")
    mark_job_run(
        progress_job["id"],
        True,
        work_status="progress",
        work_status_source="agent_marker",
        evidence_id=None,
        executor_handle="worker-1",
    )
    waiting_job = create_job(prompt="claim waiting", schedule="every 5m", deliver="local")
    mark_job_run(
        waiting_job["id"],
        True,
        work_status="waiting_external",
        work_status_source="agent_marker",
        evidence_id="ev-1",
        executor_handle=None,
    )

    stored_progress = get_job(progress_job["id"])
    stored_waiting = get_job(waiting_job["id"])
    assert stored_progress is not None
    assert stored_waiting is not None
    assert stored_progress["last_work_status"] == "unknown"
    assert stored_progress["last_work_status_source"] == "none"
    assert stored_waiting["last_work_status"] == "unknown"
    assert stored_waiting["last_work_status_source"] == "none"


def test_duplicate_evidence_cannot_be_reported_as_new_progress(hermes_env):
    from cron.jobs import create_job, get_job, mark_job_run

    job = create_job(prompt="repeat old artifact", schedule="every 5m", deliver="local")
    for _ in range(3):
        mark_job_run(
            job["id"],
            True,
            work_status="progress",
            work_status_source="agent_marker",
            evidence_id="same-evidence",
            executor_handle="worker-1",
        )

    stored = get_job(job["id"])
    assert stored is not None
    assert stored["last_work_status"] == "no_progress"
    assert stored["last_evidence_id"] is None
    assert stored["last_evidence_verified"] is False


def test_nonconsecutive_duplicate_evidence_cannot_be_reported_as_progress(hermes_env):
    from cron.jobs import create_job, get_job, mark_job_run

    job = create_job(prompt="cycle old artifacts", schedule="every 5m", deliver="local")
    for evidence_id in ("evidence-a", "evidence-b", "evidence-a"):
        mark_job_run(
            job["id"],
            True,
            work_status="progress",
            work_status_source="agent_marker",
            evidence_id=evidence_id,
            executor_handle="worker-1",
        )

    stored = get_job(job["id"])
    assert stored is not None
    assert stored["last_work_status"] == "no_progress"
    assert stored["last_evidence_id"] is None


def test_evidence_registry_fails_closed_at_capacity_without_eviction(hermes_env):
    from cron.jobs import create_job, get_job, mark_job_run

    job = create_job(prompt="many artifacts", schedule="every 5m", deliver="local")
    for index in range(256):
        mark_job_run(
            job["id"],
            True,
            work_status="progress",
            work_status_source="agent_marker",
            evidence_id=f"ev-{index}",
            executor_handle="worker-1",
        )
    mark_job_run(
        job["id"],
        True,
        work_status="progress",
        work_status_source="agent_marker",
        evidence_id="ev-over-capacity",
        executor_handle="worker-1",
    )
    at_capacity = get_job(job["id"])
    assert at_capacity is not None
    assert at_capacity["last_work_status"] == "unknown"
    assert at_capacity["last_evidence_id"] is None
    assert len(at_capacity["work_evidence_ids"]) == 256

    mark_job_run(
        job["id"],
        True,
        work_status="progress",
        work_status_source="agent_marker",
        evidence_id="ev-0",
        executor_handle="worker-1",
    )
    replay = get_job(job["id"])
    assert replay is not None
    assert replay["last_work_status"] == "no_progress"
    assert replay["last_evidence_id"] is None


def test_pause_survives_late_run_completion(hermes_env):
    from cron.jobs import create_job, get_job, mark_job_run, pause_job

    job = create_job(prompt="do work", schedule="every 5m", deliver="local")
    pause_job(job["id"], reason="operator pause during run")
    mark_job_run(job["id"], True, delivery_status="ok", work_status="no_progress")

    stored = get_job(job["id"])
    assert stored is not None
    assert stored["enabled"] is False
    assert stored["state"] == "paused"
    assert stored["paused_reason"] == "operator pause during run"


def test_pause_survives_late_completion_for_finite_and_oneshot_jobs(hermes_env):
    from cron.jobs import create_job, get_job, mark_job_run, pause_job

    finite = create_job(
        prompt="finite", schedule="every 5m", repeat=1, deliver="local"
    )
    oneshot = create_job(
        prompt="once", schedule="2026-07-18T12:00:00Z", deliver="local"
    )
    for job in (finite, oneshot):
        pause_job(job["id"], reason="operator pause during run")
        mark_job_run(job["id"], True, delivery_status="ok", work_status="no_progress")
        stored = get_job(job["id"])
        assert stored is not None
        assert stored["enabled"] is False
        assert stored["state"] == "paused"
        assert stored["paused_reason"] == "operator pause during run"


def test_completed_oneshot_retains_truthful_status_for_inspection(hermes_env):
    from cron.jobs import create_job, get_job, mark_job_run

    job = create_job(
        prompt="once", schedule="2026-07-18T12:00:00Z", deliver="local"
    )
    mark_job_run(
        job["id"],
        True,
        delivery_status="error",
        delivery_error="Slack unavailable",
        work_status="progress",
        work_status_source="agent_marker",
        evidence_id="ev-once",
        executor_handle="worker-once",
    )

    stored = get_job(job["id"])
    assert stored is not None
    assert stored["enabled"] is False
    assert stored["state"] == "completed"
    assert stored["last_run_status"] == "ok"
    assert stored["last_delivery_status"] == "error"
    assert stored["last_work_status"] == "reported_progress"
    assert stored["last_evidence_id"] == "ev-once"


def test_run_one_job_strips_marker_and_records_all_statuses(hermes_env, monkeypatch):
    import cron.scheduler as scheduler
    import agent.secret_scope as secret_scope

    delivered = []
    marked = []
    response = (
        "Verified artifact created\n"
        "[HERMES_WORK_RESULT] "
        '{"workStatus":"progress","evidenceId":"ev-789","executorHandle":"worker-1"}'
    )
    monkeypatch.setattr(scheduler, "create_execution", lambda *a, **k: {"id": "exec-1"})
    monkeypatch.setattr(scheduler, "claim_dispatch", lambda *a, **k: True)
    monkeypatch.setattr(scheduler, "mark_execution_running", lambda *a, **k: None)
    monkeypatch.setattr(
        scheduler,
        "run_job",
        lambda *a, **k: (True, "audit output", response, None),
    )
    monkeypatch.setattr(scheduler, "save_job_output", lambda *a, **k: "/tmp/output")
    monkeypatch.setattr(scheduler, "_resolve_delivery_targets", lambda *a, **k: [])
    monkeypatch.setattr(
        scheduler,
        "_deliver_result",
        lambda job, content, **kwargs: delivered.append(content),
    )
    monkeypatch.setattr(scheduler, "_is_interrupted", lambda *a, **k: False)
    monkeypatch.setattr(scheduler, "_consume_interrupted_flag", lambda *a, **k: False)
    monkeypatch.setattr(
        scheduler,
        "mark_job_run",
        lambda *args, **kwargs: marked.append((args, kwargs)),
    )
    monkeypatch.setattr(scheduler, "finish_execution", lambda *a, **k: None)
    monkeypatch.setattr(secret_scope, "build_profile_secret_scope", lambda *a, **k: object())
    monkeypatch.setattr(secret_scope, "set_secret_scope", lambda *a, **k: object())
    monkeypatch.setattr(secret_scope, "reset_secret_scope", lambda *a, **k: None)

    processed = scheduler.run_one_job(
        {"id": "job-1", "name": "status integration", "deliver": "local"}
    )

    assert processed is True
    assert delivered == ["Verified artifact created"]
    assert len(marked) == 1
    args, kwargs = marked[0]
    assert args[:3] == ("job-1", True, None)
    assert kwargs == {
        "delivery_error": None,
        "delivery_status": "not_requested",
        "work_status": "progress",
        "work_status_source": "agent_marker",
        "evidence_id": "ev-789",
        "evidence_verified": False,
        "executor_handle": "worker-1",
    }


def test_cron_prompt_requires_explicit_work_result_metadata(hermes_env):
    from cron.scheduler import _build_job_prompt

    prompt = _build_job_prompt(
        {"id": "job-1", "name": "status contract", "prompt": "perform one bounded slice"}
    )

    assert prompt is not None
    assert "[HERMES_WORK_RESULT]" in prompt
    assert '"workStatus":"no_progress"' in prompt
    assert "Progress requires a concrete evidenceId" in prompt


def test_latest_recovered_execution_overrides_stale_progress_in_surfaces(hermes_env):
    from tools.cronjob_tools import _format_job

    formatted = _format_job(
        {
            "id": "job-crashed",
            "name": "crashed worker",
            "schedule": {"kind": "interval", "seconds": 60},
            "last_run_at": "2026-07-17T16:00:00Z",
            "last_status": "ok",
            "last_run_status": "ok",
            "last_delivery_status": "ok",
            "last_work_status": "reported_progress",
            "last_work_status_source": "agent_marker",
            "last_evidence_id": "stale-evidence",
            "last_evidence_verified": False,
            "last_executor_handle": "stale-worker",
            "latest_execution": {
                "id": "exec-crashed",
                "job_id": "job-crashed",
                "status": "unknown",
                "claimed_at": "2026-07-17T16:05:00Z",
                "started_at": "2026-07-17T16:05:01Z",
                "finished_at": "2026-07-17T16:06:00Z",
                "error": "scheduler process exited before completion",
            },
        }
    )

    assert formatted["last_run_status"] == "unknown"
    assert formatted["last_delivery_status"] == "unknown"
    assert formatted["last_work_status"] == "unknown"
    assert formatted["last_work_status_source"] == "none"
    assert formatted["last_evidence_id"] is None
    assert formatted["last_executor_handle"] is None


def test_cron_tool_format_surfaces_separate_status_dimensions(hermes_env):
    from tools.cronjob_tools import _format_job

    formatted = _format_job(
        {
            "id": "job-1",
            "name": "truthful status",
            "schedule": {"kind": "interval", "seconds": 60},
            "last_status": "ok",
            "last_run_status": "ok",
            "last_delivery_status": "error",
            "last_work_status": "no_progress",
            "last_work_status_source": "agent_marker",
            "last_evidence_id": None,
            "last_evidence_verified": False,
            "last_executor_handle": None,
        }
    )

    assert formatted["last_status"] == "ok"
    assert formatted["last_run_status"] == "ok"
    assert formatted["last_delivery_status"] == "error"
    assert formatted["last_work_status"] == "no_progress"
    assert formatted["last_work_status_source"] == "agent_marker"
    assert formatted["last_evidence_id"] is None
    assert formatted["last_evidence_verified"] is False
    assert formatted["last_executor_handle"] is None
