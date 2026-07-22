"""Fail-closed admission tests for context-free cron fixtures.

Regression: stale tests created recurring jobs named ``claim job`` with prompt
``x`` in the live store. Each fire constructed SessionDB/AIAgent and spent model
tokens despite carrying no task contract.
"""

from datetime import timedelta
from unittest.mock import patch

import cron.scheduler as scheduler
from cron.jobs import context_free_fixture_reason, create_job, get_job, update_job
from cron.scheduler import SILENT_MARKER, run_one_job
from hermes_time import now as hermes_now


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
    }
    job.update(updates)
    return job


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


def test_stale_fixture_snapshot_does_not_pause_updated_meaningful_job():
    created = create_job(name="claim job", schedule="0 7 * * *", prompt="x")
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
    job = create_job(name="claim job", schedule=run_at, prompt="x")

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


def test_recurring_fixture_quarantine_preserves_repeat_and_is_idempotent():
    job = create_job(
        name="daily build",
        schedule="every 60m",
        prompt="build",
        repeat=3,
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


def test_scripted_job_with_fixture_pair_still_executes_normally():
    job = create_job(
        name="daily build",
        schedule="every 60m",
        prompt="build",
        script="collector.py",
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
