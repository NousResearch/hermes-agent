"""Fail-closed admission tests for context-free cron fixtures.

Regression: stale tests created recurring jobs named ``claim job`` with prompt
``x`` in the live store. Each fire constructed SessionDB/AIAgent and spent model
tokens despite carrying no task contract.
"""

from datetime import timedelta
from unittest.mock import patch

import cron.scheduler as scheduler
from cron.executions import latest_execution
from cron.jobs import (
    context_free_fixture_reason,
    create_job,
    get_job,
    remove_job,
    update_job,
    use_cron_store,
)
from cron.scheduler import SILENT_MARKER, run_one_job
from hermes_time import now as hermes_now


HISTORICAL_FIXTURE_CREATED_AT = "2026-07-16T12:00:00+00:00"


def _fixture_job(**updates):
    job = {
        "id": "fixture-job",
        "name": "claim job",
        "prompt": "x",
        "script": None,
        "skills": None,
        "skill": None,
        "workdir": None,
        "context_from": None,
        "origin": None,
        "model": None,
        "provider": None,
        "base_url": None,
        "enabled_toolsets": None,
        "no_agent": False,
        "deliver": "local",
        "attach_to_session": None,
        "created_at": HISTORICAL_FIXTURE_CREATED_AT,
    }
    job.update(updates)
    return job


def _mark_historical_fixture(job):
    updated = update_job(
        job["id"], {"created_at": HISTORICAL_FIXTURE_CREATED_AT}
    )
    assert updated is not None
    return updated


def test_context_free_fixture_is_detected():
    for name, prompt in (
        ("claim job", "x"),
        ("paused job", "x"),
        ("daily build", "build"),
        ("hourly build", "build"),
        ("w", "echo hi"),
    ):
        reason = context_free_fixture_reason(_fixture_job(name=name, prompt=prompt))
        assert reason is not None, (name, prompt)
        assert "context-free fixture prompt" in reason


def test_meaningful_job_context_prevents_quarantine():
    meaningful = [
        {"script": "collector.py"},
        {"skills": ["overnight-assistant-operations"]},
        {"skill": "overnight-assistant-operations"},
        {"workdir": "/tmp/project"},
        {"context_from": ["upstream-job"]},
        {"origin": {"platform": "discord", "chat_id": "123"}},
        {"model": "gpt-5.6-terra"},
        {"provider": "openai-codex"},
        {"base_url": "https://inference.example.test/v1"},
        {"enabled_toolsets": ["file"]},
        {"deliver": "discord:#ops"},
        {"attach_to_session": True},
        {"no_agent": True},
    ]
    for updates in meaningful:
        assert context_free_fixture_reason(_fixture_job(**updates)) is None, updates


def test_non_fixture_prompt_is_not_detected():
    assert (
        context_free_fixture_reason(
            _fixture_job(
                prompt="Summarize the latest local report and save a review card."
            )
        )
        is None
    )


def test_known_prompt_with_unrelated_name_is_not_detected():
    """A concise legitimate job must not be quarantined by prompt alone."""
    assert (
        context_free_fixture_reason(
            _fixture_job(name="Project build reminder", prompt="build")
        )
        is None
    )


def test_same_fixture_pair_created_outside_incident_window_is_not_detected():
    """A future legitimate concise job must not match the historical batch."""
    assert (
        context_free_fixture_reason(
            _fixture_job(created_at="2026-07-21T12:00:00+00:00")
        )
        is None
    )


def test_stale_fixture_snapshot_does_not_pause_updated_meaningful_job():
    created = _mark_historical_fixture(
        create_job(name="claim job", schedule="0 7 * * *", prompt="x")
    )
    stale_snapshot = dict(created)
    updated = update_job(
        created["id"],
        {
            "prompt": "Summarize the local report for review.",
            "origin": {"platform": "discord", "chat_id": "123"},
        },
    )
    assert updated is not None

    with (
        patch.object(
            scheduler,
            "run_job",
            return_value=(True, "# receipt", SILENT_MARKER, None),
        ) as run_mock,
        patch.object(scheduler, "_deliver_result") as deliver_mock,
    ):
        assert run_one_job(stale_snapshot) is True

    stored = get_job(created["id"])
    assert stored is not None
    assert stored["enabled"] is True
    assert stored["state"] != "paused"
    assert run_mock.call_args.args[0]["prompt"] == updated["prompt"]
    deliver_mock.assert_not_called()


def test_run_one_job_keeps_one_shot_fixture_paused_and_saves_receipt():
    run_at = (hermes_now() + timedelta(minutes=5)).isoformat()
    job = _mark_historical_fixture(
        create_job(name="claim job", schedule=run_at, prompt="x")
    )

    with (
        patch.object(
            scheduler, "run_job", side_effect=AssertionError("agent path reached")
        ),
        patch.object(scheduler, "mark_job_run") as mark_mock,
        patch.object(scheduler, "_deliver_result") as deliver_mock,
        patch.object(
            scheduler, "save_job_output", wraps=scheduler.save_job_output
        ) as save_mock,
    ):
        assert run_one_job(job) is True

    stored = get_job(job["id"])
    assert stored is not None
    assert stored["enabled"] is False
    assert stored["state"] == "paused"
    assert "context-free fixture prompt" in (stored["paused_reason"] or "")
    mark_mock.assert_not_called()
    deliver_mock.assert_not_called()
    save_mock.assert_called_once()
    saved_doc = save_mock.call_args.args[1]
    assert "QUARANTINED" in saved_doc
    ledger = latest_execution(job["id"])
    assert ledger is not None
    assert ledger["status"] == "failed"
    assert ledger["started_at"] is None
    assert "admission rejected" in (ledger["error"] or "").lower()


def test_quarantine_receipt_failure_preserves_one_shot_and_notifies_provider():
    run_at = (hermes_now() + timedelta(minutes=5)).isoformat()
    job = _mark_historical_fixture(
        create_job(name="claim job", schedule=run_at, prompt="x")
    )
    original_repeat = dict(job["repeat"])

    with (
        patch.object(
            scheduler, "run_job", side_effect=AssertionError("agent path reached")
        ),
        patch.object(
            scheduler, "save_job_output", side_effect=OSError("receipt disk full")
        ),
        patch.object(scheduler, "mark_job_run") as mark_mock,
        patch.object(scheduler, "_notify_provider_jobs_changed") as notify_mock,
    ):
        assert run_one_job(job) is True

    stored = get_job(job["id"])
    assert stored is not None
    assert stored["state"] == "paused"
    assert stored["repeat"] == original_repeat
    mark_mock.assert_not_called()
    notify_mock.assert_called_once()
    ledger = latest_execution(job["id"])
    assert ledger is not None
    assert ledger["status"] == "failed"
    assert ledger["started_at"] is None


def test_quarantine_ledger_failure_cannot_fall_through_to_repeat_accounting():
    run_at = (hermes_now() + timedelta(minutes=5)).isoformat()
    job = _mark_historical_fixture(
        create_job(name="claim job", schedule=run_at, prompt="x")
    )
    original_repeat = dict(job["repeat"])

    with (
        patch.object(
            scheduler, "run_job", side_effect=AssertionError("agent path reached")
        ),
        patch.object(
            scheduler, "finish_execution", side_effect=OSError("ledger disk full")
        ),
        patch.object(scheduler, "mark_job_run") as mark_mock,
        patch.object(scheduler, "_notify_provider_jobs_changed") as notify_mock,
    ):
        assert run_one_job(job) is True

    stored = get_job(job["id"])
    assert stored is not None
    assert stored["state"] == "paused"
    assert stored["repeat"] == original_repeat
    mark_mock.assert_not_called()
    notify_mock.assert_called_once()


def test_admission_store_failure_rejects_before_agent_and_repeat_accounting():
    job = _mark_historical_fixture(
        create_job(name="claim job", schedule="0 7 * * *", prompt="x")
    )

    with (
        patch.object(
            scheduler,
            "quarantine_context_free_fixture_job",
            side_effect=OSError("jobs store unavailable"),
        ),
        patch.object(
            scheduler, "run_job", side_effect=AssertionError("agent path reached")
        ) as run_mock,
        patch.object(scheduler, "mark_job_run") as mark_mock,
    ):
        assert run_one_job(job) is False

    run_mock.assert_not_called()
    mark_mock.assert_not_called()
    ledger = latest_execution(job["id"])
    assert ledger is not None
    assert ledger["status"] == "failed"
    assert ledger["started_at"] is None
    assert "admission check failed" in (ledger["error"] or "").lower()


def test_recurring_fixture_quarantine_preserves_repeat_and_is_idempotent():
    job = _mark_historical_fixture(
        create_job(
            name="daily build",
            schedule="every 60m",
            prompt="build",
            repeat=3,
        )
    )
    original_repeat = dict(job["repeat"])

    with (
        patch.object(
            scheduler, "run_job", side_effect=AssertionError("agent path reached")
        ),
        patch.object(scheduler, "claim_dispatch") as claim_mock,
        patch.object(scheduler, "mark_job_run") as mark_mock,
        patch.object(scheduler, "_deliver_result") as deliver_mock,
        patch.object(
            scheduler, "save_job_output", wraps=scheduler.save_job_output
        ) as save_mock,
        patch.object(scheduler, "_notify_provider_jobs_changed") as notify_mock,
    ):
        assert run_one_job(dict(job)) is True
        assert run_one_job(dict(job)) is True

    stored = get_job(job["id"])
    assert stored is not None
    assert stored["enabled"] is False
    assert stored["state"] == "paused"
    assert stored["repeat"] == original_repeat
    claim_mock.assert_not_called()
    mark_mock.assert_not_called()
    deliver_mock.assert_not_called()
    save_mock.assert_called_once()
    notify_mock.assert_called_once()


def test_deleted_persisted_snapshot_is_rejected_before_agent_path():
    job = _mark_historical_fixture(
        create_job(name="claim job", schedule="0 7 * * *", prompt="x")
    )
    assert remove_job(job["id"]) is True

    with (
        patch.object(
            scheduler, "run_job", side_effect=AssertionError("agent path reached")
        ) as run_mock,
        patch.object(scheduler, "mark_job_run") as mark_mock,
    ):
        assert run_one_job(job, require_persisted=True) is True

    run_mock.assert_not_called()
    mark_mock.assert_not_called()
    ledger = latest_execution(job["id"])
    assert ledger is not None
    assert ledger["status"] == "failed"
    assert ledger["started_at"] is None
    assert "no longer exists" in (ledger["error"] or "").lower()


def test_quarantine_execution_ledger_follows_context_local_profile(tmp_path):
    profile_home = tmp_path / "profile"
    with use_cron_store(profile_home):
        job = _mark_historical_fixture(
            create_job(name="claim job", schedule="0 7 * * *", prompt="x")
        )
        with patch.object(
            scheduler, "run_job", side_effect=AssertionError("agent path reached")
        ):
            assert run_one_job(job) is True

        ledger = latest_execution(job["id"])
        assert ledger is not None
        assert ledger["status"] == "failed"

    assert (profile_home / "cron" / "executions.db").exists()


def test_scripted_job_with_fixture_pair_still_executes_normally():
    job = _mark_historical_fixture(
        create_job(
            name="daily build",
            schedule="every 60m",
            prompt="build",
            script="collector.py",
        )
    )
    with patch.object(
        scheduler,
        "run_job",
        return_value=(True, "# script receipt", SILENT_MARKER, None),
    ) as run_mock:
        assert run_one_job(job) is True

    stored = get_job(job["id"])
    assert stored is not None
    assert stored["enabled"] is True
    assert stored["state"] != "paused"
    run_mock.assert_called_once()
