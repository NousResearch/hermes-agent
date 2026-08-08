from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest


FIXTURES = [
    "cc-audit-stale-running.json",
    "state-store-unavailable.json",
    "suppressed-receipt-success.json",
    "receipt-conflict-429.json",
    "hmm-policy-block.json",
    "unknown-nonidempotent.json",
    "action-readback-failed.json",
]


def _fixture_path(name: str) -> Path:
    return Path("docs/cron-control/p0/fixtures") / name


@pytest.mark.parametrize("fixture_name", FIXTURES)
def test_evaluator_matches_frozen_fixtures(fixture_name):
    from cron_control.evaluator import evaluate_job_verdict

    fixture = json.loads(_fixture_path(fixture_name).read_text(encoding="utf-8"))
    verdict = evaluate_job_verdict(fixture["job"], fixture["evidence"])
    expected = fixture["expected"]

    assert verdict["state"] == expected["state"]
    assert verdict["evidence_state"] == expected["evidence_state"]
    assert verdict["recommended_action"] == expected["recommended_action"]
    assert verdict["automatic_action_allowed"] == expected["automatic_action_allowed"]
    assert verdict["rule_id"] == expected["rule_id"]
    assert set(verdict["evidence_refs"]) == {item["evidence_id"] for item in fixture["evidence"]}
    assert verdict["blocked_by"] == list(dict.fromkeys(verdict["blocked_by"]))


def test_evaluator_persists_verdicts_idempotently(tmp_path, monkeypatch):
    from cron_control.evaluator import evaluate_job_verdict, persist_verdicts

    fixture = json.loads(_fixture_path("suppressed-receipt-success.json").read_text(encoding="utf-8"))
    verdict = evaluate_job_verdict(fixture["job"], fixture["evidence"])
    control_plane_path = tmp_path / "cron" / "control-plane.db"

    persist_verdicts({"evidence": fixture["evidence"]}, [verdict], control_plane_path)
    persist_verdicts({"evidence": fixture["evidence"]}, [verdict], control_plane_path)

    with sqlite3.connect(control_plane_path) as conn:
        verdict_count = conn.execute("SELECT COUNT(*) FROM verdicts").fetchone()[0]
        incident_count = conn.execute("SELECT COUNT(*) FROM incidents").fetchone()[0]
        audit_count = conn.execute("SELECT COUNT(*) FROM audit_events").fetchone()[0]

    assert verdict_count == 1
    assert incident_count == 1
    assert audit_count == 2


def test_evaluate_main_prints_verdict_json(monkeypatch, capsys):
    from cron_control import evaluator

    monkeypatch.setattr(
        "cron_control.shadow.collect_shadow_snapshot",
        lambda **_kwargs: {
            "jobs": [{"id": "job-1"}],
            "evidence": [{"job_id": "job-1", "evidence_id": "ev_1", "incident_id": "inc_1", "kind": "job_metadata", "source": "jobs.json", "observed_at": "2026-07-29T00:00:00+00:00", "source_time": "2026-07-29T00:00:00+00:00", "value": {"id": "job-1"}, "source_ref": "jobs.json:job-1", "content_hash": "sha256:" + "0" * 64, "freshness_seconds": 0, "validation": "valid"}],
        },
    )
    monkeypatch.setattr(
        evaluator,
        "evaluate_snapshot",
        lambda snapshot: [
            {
                "verdict_id": "vd_1",
                "incident_id": "inc_1",
                "job_id": "job-1",
                "state": "healthy",
                "evidence_state": "complete",
                "rule_id": "SUSPECT_TO_HEALTHY_EVIDENCE_RECOVERED_V1",
                "evidence_refs": ["ev_1"],
                "recommended_action": "none",
                "automatic_action_allowed": True,
                "blocked_by": [],
                "classified_at": "2026-07-29T00:00:00+00:00",
                "classifier_version": "cron_control.evaluator/v1",
            }
        ],
    )

    assert evaluator.main([]) == 0
    out = capsys.readouterr().out
    assert '"state": "healthy"' in out
