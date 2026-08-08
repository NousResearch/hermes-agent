from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    (home / "cron").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _fixture(name: str) -> dict:
    return json.loads((Path("docs/cron-control/p0/fixtures") / name).read_text(encoding="utf-8"))


def _write_jobs(home: Path, jobs: list[dict]) -> Path:
    jobs_path = home / "cron" / "jobs.json"
    jobs_path.write_text(
        json.dumps({"jobs": jobs, "updated_at": "2026-07-29T12:00:00+00:00"}, indent=2),
        encoding="utf-8",
    )
    return jobs_path


def _manual_verdict(job_id: str, action: str, incident_id: str = "inc_manual") -> dict:
    return {
        "verdict_id": f"vd_{job_id}_{action}",
        "incident_id": incident_id,
        "job_id": job_id,
        "state": "recoverable" if action == "switch_provider" else "systemic_failure",
        "evidence_state": "complete",
        "rule_id": "TRANSIENT_TO_RECOVERABLE_FALLBACK_OK_V1" if action == "switch_provider" else "HEALTHY_TO_SYSTEMIC_FAILURE_LEDGER_V1",
        "evidence_refs": [f"ev_{job_id}_{action}"],
        "recommended_action": action,
        "automatic_action_allowed": False,
        "blocked_by": [],
        "classified_at": "2026-07-29T16:00:00+00:00",
        "classifier_version": "cron_control.evaluator/v1",
    }


def test_execute_reset_job_requires_approval(hermes_home):
    from cron.jobs import get_job
    from cron_control.actions import execute_verdict_action
    from cron_control.evaluator import evaluate_job_verdict

    fixture = _fixture("cc-audit-stale-running.json")
    job = dict(fixture["job"])
    job["run_claim"] = {"at": "2026-07-29T14:58:00+08:00", "by": "runner-1"}
    _write_jobs(hermes_home, [job])
    verdict = evaluate_job_verdict(job, fixture["evidence"])

    outcome = execute_verdict_action(verdict, approved=False)

    assert outcome["status"] == "blocked"
    assert outcome["result"] == "denied"
    assert outcome["blocked_reason"] == "approval_required"
    assert get_job(job["id"])["run_claim"] == {"at": "2026-07-29T14:58:00+08:00", "by": "runner-1"}

    control_plane_path = hermes_home / "cron" / "control-plane.db"
    with sqlite3.connect(control_plane_path) as conn:
        action_count = conn.execute("SELECT COUNT(*) FROM actions").fetchone()[0]
        audit_count = conn.execute("SELECT COUNT(*) FROM audit_events").fetchone()[0]

    assert action_count == 1
    assert audit_count == 1


def test_execute_reset_job_approved_updates_job_and_audits(hermes_home):
    from cron.jobs import get_job
    from cron_control.actions import execute_verdict_action
    from cron_control.evaluator import evaluate_job_verdict

    fixture = _fixture("cc-audit-stale-running.json")
    job = dict(fixture["job"])
    job["run_claim"] = {"at": "2026-07-29T14:58:00+08:00", "by": "runner-1"}
    _write_jobs(hermes_home, [job])
    verdict = evaluate_job_verdict(job, fixture["evidence"])

    outcome = execute_verdict_action(verdict, approved=True)

    assert outcome["status"] == "verified"
    assert outcome["result"] == "verified"
    updated = get_job(job["id"])
    assert updated["state"] == "scheduled"
    assert updated["enabled"] is True
    assert updated["run_claim"] is None
    assert updated["fire_claim"] is None

    control_plane_path = hermes_home / "cron" / "control-plane.db"
    with sqlite3.connect(control_plane_path) as conn:
        row = conn.execute(
            "SELECT status, result FROM actions WHERE action_id=?",
            (outcome["action_id"],),
        ).fetchone()
        audit_types = [r[0] for r in conn.execute("SELECT event_type FROM audit_events ORDER BY audit_id").fetchall()]

    assert row == ("verified", "verified")
    assert set(audit_types) == {"action_planned", "action_started", "action_completed"}


def test_execute_switch_provider_approved_updates_job(hermes_home):
    from cron.jobs import get_job
    from cron_control.actions import execute_verdict_action

    job = {
        "id": "switch-1",
        "name": "Switch provider",
        "enabled": True,
        "state": "scheduled",
        "deliver": "local",
        "provider": "opencode-go",
        "model": "deepseek-v4-pro",
        "schedule": {"kind": "interval", "minutes": 15, "display": "every 15m"},
        "control_policy": {
            "idempotent": True,
            "side_effect_class": "read_only",
            "rerun_policy": "automatic",
        },
    }
    _write_jobs(hermes_home, [job])
    verdict = _manual_verdict(job["id"], "switch_provider")

    outcome = execute_verdict_action(verdict, approved=True)

    updated = get_job(job["id"])
    assert updated["provider"] != job["provider"] or updated["model"] != job["model"]
    assert updated["recovery_state"]["primary_provider"] == "opencode-go"
    assert updated["recovery_state"]["fallback_provider"] == updated["provider"]
    assert outcome["status"] == "verified"
    assert outcome["result"] == "verified"


def test_execute_repair_state_store_verifies_readback(hermes_home):
    from cron_control.actions import execute_verdict_action
    from cron_control.store import open_control_plane_db

    job = {
        "id": "repair-1",
        "name": "Repair state store",
        "enabled": True,
        "state": "scheduled",
        "deliver": "local",
        "schedule": {"kind": "once", "run_at": "2026-07-29T17:00:00+08:00", "display": "once"},
        "control_policy": {
            "idempotent": True,
            "side_effect_class": "read_only",
            "rerun_policy": "automatic",
        },
    }
    _write_jobs(hermes_home, [job])
    verdict = _manual_verdict(job["id"], "repair_state_store", incident_id="inc_repair")

    outcome = execute_verdict_action(verdict, approved=True)

    assert outcome["status"] == "verified"
    assert outcome["result"] == "verified"

    control_plane_path = hermes_home / "cron" / "control-plane.db"
    with open_control_plane_db(control_plane_path) as conn:
        heartbeat = conn.execute(
            "SELECT component_id, status, detail FROM component_heartbeats WHERE component_id=?",
            ("cron-control-repair",),
        ).fetchone()

    assert tuple(heartbeat) == ("cron-control-repair", "healthy", "state-store repair read-back passed")


def test_execute_repair_state_store_fails_when_readback_missing(hermes_home, monkeypatch):
    from cron_control import actions

    job = {
        "id": "repair-2",
        "name": "Repair state store",
        "enabled": True,
        "state": "scheduled",
        "deliver": "local",
        "schedule": {"kind": "once", "run_at": "2026-07-29T17:00:00+08:00", "display": "once"},
        "control_policy": {
            "idempotent": True,
            "side_effect_class": "read_only",
            "rerun_policy": "automatic",
        },
    }
    _write_jobs(hermes_home, [job])
    verdict = _manual_verdict(job["id"], "repair_state_store", incident_id="inc_repair_fail")

    monkeypatch.setattr(actions, "record_component_heartbeat", lambda conn, heartbeat: heartbeat)

    outcome = actions.execute_verdict_action(verdict, approved=True)

    assert outcome["status"] == "failed"
    assert outcome["result"] == "verification_failed"
    assert outcome["after_state"]["expected"]["heartbeat"]["component_id"] == "cron-control-repair"
    assert outcome["after_state"]["actual"]["heartbeat"] is None
