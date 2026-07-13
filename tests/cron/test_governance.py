"""Deterministic policy tests for recurring-job governance (OPS-1308)."""

import json
from unittest.mock import Mock

from cron import scheduler
from cron.governance import append_governance_audit, evaluate_cron_governance_policy


def authorized_job(**overrides):
    job = {
        "id": "cron-1",
        "name": "Governed local script",
        "enabled": True,
        "state": "scheduled",
        "deliver": "local",
        "no_agent": True,
        "schedule": {"kind": "interval", "minutes": 120},
        "governance": {
            "owner_issue": "OPS-1308",
            "cadence": {"min_minutes": 60},
        },
    }
    job.update(overrides)
    return job


def test_paused_job_is_blocked_even_if_enabled():
    decision = evaluate_cron_governance_policy(
        authorized_job(state="paused", enabled=True)
    )

    assert decision.allow is False
    assert decision.code == "paused_state"


def test_job_without_accountable_issue_is_blocked():
    job = authorized_job(governance={"cadence": {"min_minutes": 60}})

    decision = evaluate_cron_governance_policy(job)

    assert decision.allow is False
    assert decision.code == "missing_owner_issue"


def test_agent_job_without_model_budget_policy_is_blocked():
    job = authorized_job(
        no_agent=False,
        governance={
            "owner_issue": "OPS-1308",
            "cadence": {"min_minutes": 60},
        },
    )

    decision = evaluate_cron_governance_policy(job)

    assert decision.allow is False
    assert decision.code == "missing_model_budget_policy"


def test_short_interval_is_blocked_before_execution():
    job = authorized_job(schedule={"kind": "interval", "minutes": 30})

    decision = evaluate_cron_governance_policy(job)

    assert decision.allow is False
    assert decision.code == "over_cadence"


def test_daily_cron_schedule_can_satisfy_cadence_policy():
    job = authorized_job(schedule={"kind": "cron", "expr": "0 13 * * *"})

    decision = evaluate_cron_governance_policy(job)

    assert decision.allow is True
    assert decision.code == "allowed"


def test_cron_expression_that_runs_too_frequently_is_blocked():
    job = authorized_job(schedule={"kind": "cron", "expr": "*/5 * * * *"})

    decision = evaluate_cron_governance_policy(job)

    assert decision.allow is False
    assert decision.code == "over_cadence"


def test_nonlocal_delivery_requires_explicit_authorization():
    job = authorized_job(deliver="origin")

    decision = evaluate_cron_governance_policy(job)

    assert decision.allow is False
    assert decision.code == "nonlocal_delivery_not_authorized"


def test_governed_agent_with_authorized_delivery_and_budget_is_allowed():
    job = authorized_job(
        no_agent=False,
        governance={
            "owner_issue": "OPS-1308",
            "cadence": {"min_minutes": 60},
            "model_policy": {
                "provider": "openai-codex",
                "model": "gpt-5.5",
                "max_runs_per_day": 1,
            },
            "budget_policy": {"max_usd_per_run": 0},
            "allow_nonlocal_delivery": True,
        },
        deliver="origin",
    )

    decision = evaluate_cron_governance_policy(job)

    assert decision.allow is True
    assert decision.code == "allowed"


def test_tick_does_not_advance_or_run_a_paused_due_job(monkeypatch, tmp_path):
    job = authorized_job(state="paused", enabled=True)
    advance = Mock()
    run = Mock(return_value=(True, "", "[SILENT]", None))
    audit = Mock()

    monkeypatch.setattr(scheduler, "_get_lock_paths", lambda: (tmp_path, tmp_path / "tick.lock"))
    monkeypatch.setattr(scheduler, "_cron_governance_enforced", lambda: True)
    monkeypatch.setattr(scheduler, "get_due_jobs", lambda: [job])
    monkeypatch.setattr(scheduler, "advance_next_run", advance)
    monkeypatch.setattr(scheduler, "run_job", run)
    monkeypatch.setattr(scheduler, "save_job_output", Mock())
    monkeypatch.setattr(scheduler, "append_governance_audit", audit)
    monkeypatch.setattr(scheduler, "_kill_orphaned_mcp_children", Mock(), raising=False)

    scheduler.tick(verbose=False)

    advance.assert_not_called()
    run.assert_not_called()
    audit.assert_called_once()


def test_governance_audit_is_secret_free_and_bounded(tmp_path):
    job = authorized_job(name="Governed job", prompt="do not persist this")
    decision = evaluate_cron_governance_policy(job)
    audit_path = tmp_path / "governance.jsonl"

    append_governance_audit(audit_path, job, decision)

    event = json.loads(audit_path.read_text().strip())
    assert event["job_id"] == "cron-1"
    assert event["decision"] == "allow"
    assert event["code"] == "allowed"
    assert "prompt" not in event
    assert event["name"] == "Governed job"
