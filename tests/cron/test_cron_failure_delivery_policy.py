"""Failure delivery routing must not pollute user-facing cron destinations."""

import pytest

import cron.scheduler as scheduler


COACHING_TARGET = "telegram:-1003727537327:435"
SYSTEM_TARGET = "telegram:-1003727537327:1"


def _patch_pipeline(monkeypatch, *, success, final_response="Masa check-in", error=None):
    delivered = []
    monkeypatch.setattr(
        scheduler,
        "run_job",
        lambda *_args, **_kwargs: (success, "raw output", final_response, error),
    )
    monkeypatch.setattr(scheduler, "save_job_output", lambda *_args: "/tmp/output.txt")
    monkeypatch.setattr(
        scheduler,
        "_deliver_result",
        lambda job, content, **_kwargs: delivered.append((job["deliver"], content)) or None,
    )
    monkeypatch.setattr(scheduler, "mark_job_run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(scheduler, "finish_execution", lambda *_args, **_kwargs: None)
    return delivered


def test_successful_coaching_output_uses_only_the_coaching_target(monkeypatch):
    delivered = _patch_pipeline(monkeypatch, success=True)
    job = {
        "id": "masa-success",
        "name": "Masa morning check-in",
        "deliver": COACHING_TARGET,
        "failure_deliver": SYSTEM_TARGET,
    }

    assert scheduler.run_one_job(job) is True
    assert delivered == [(COACHING_TARGET, "Masa check-in")]


def test_silent_coaching_run_delivers_nothing(monkeypatch):
    delivered = _patch_pipeline(monkeypatch, success=True, final_response="[SILENT]")
    job = {
        "id": "masa-silent",
        "name": "Masa morning check-in",
        "deliver": COACHING_TARGET,
        "failure_deliver": SYSTEM_TARGET,
    }

    assert scheduler.run_one_job(job) is True
    assert delivered == []


@pytest.mark.parametrize(
    "failure_kind,error",
    [
        ("provider", "Provider request timed out"),
        ("configuration", "[blocked_config] Model/provider configuration is invalid"),
        ("script", "Script exited with code 1: refresh_masa_data.sh"),
    ],
)
def test_operational_failures_never_deliver_to_coaching_topic(
    monkeypatch, failure_kind, error
):
    delivered = _patch_pipeline(monkeypatch, success=False, error=error)
    job = {
        "id": f"masa-{failure_kind}",
        "name": "Masa coaching job",
        "deliver": COACHING_TARGET,
        "failure_deliver": SYSTEM_TARGET,
    }

    assert scheduler.run_one_job(job) is True
    assert delivered
    assert [target for target, _content in delivered] == [SYSTEM_TARGET]
    assert all(target != COACHING_TARGET for target, _content in delivered)


def test_suppress_failure_policy_never_attempts_delivery(monkeypatch):
    delivered = _patch_pipeline(monkeypatch, success=False, error="Provider unavailable")
    job = {
        "id": "masa-suppressed-failure",
        "name": "Masa coaching job",
        "deliver": COACHING_TARGET,
        "failure_deliver": "suppress",
    }

    assert scheduler.run_one_job(job) is True
    assert delivered == []


def test_failure_delivery_policy_is_exposed_through_the_cron_tool_schema():
    from tools.cronjob_tools import CRONJOB_SCHEMA

    policy = CRONJOB_SCHEMA["parameters"]["properties"]["failure_deliver"]
    assert policy["type"] == "string"
    assert "suppress" in policy["description"]
    assert "local" in policy["description"]
