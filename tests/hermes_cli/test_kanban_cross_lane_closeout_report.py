"""Integrated cross-lane closeout report tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _review_ready_child_evidence(child_id: str) -> dict:
    return {
        "schema": "kanban_closeout_evidence.v1",
        "summary": "child verified",
        "done_criteria_ledger": {"schema": "kanban_done_criteria_ledger.v1", "criteria_hash": f"criteria:{child_id}", "criteria": [{"id": "criterion-1"}]},
        "worker_evidence": {"schema": "kanban_worker_evidence.v1", "criteria_hash": f"criteria:{child_id}", "per_criterion": {"criterion-1": {"claim": "satisfied"}}, "authority_boundary_confirmed": True},
        "verifier_result": {"schema": "kanban_verifier_result.v1", "verdict": "PASS", "criteria_hash": f"criteria:{child_id}", "per_criterion": {"criterion-1": {"verdict": "PASS"}}},
        "reviewer_result": {"recommendation": "APPROVE", "quality_gate": {"decision": "PASS"}},
        "cleanup": {"proof": "clean", "worktree_clean": True},
        "authority_boundary_confirmed": True,
    }


def test_baseline_autopilot_parent_report_preserves_rollup_dimensions(kanban_home):
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="autopilot parent", triage=True)
        finished = kb.create_task(conn, title="finished child", assignee="alice")
        pending = kb.create_task(conn, title="pending child", assignee="alice")
        kb.link_tasks(conn, parent, finished, relation_type="hierarchy")
        kb.link_tasks(conn, parent, pending, relation_type="hierarchy")
        assert kb.apply_closeout_transition(conn, finished, review_phase="review_ready", closeout_evidence=_review_ready_child_evidence(finished)) is True
        conn.execute("UPDATE tasks SET status='blocked', review_phase='worker_done', closeout_evidence=? WHERE id=?", (json.dumps({"schema": "kanban_closeout_evidence.v1", "summary": "worker only"}), pending))
        report = kb.autopilot_parent_report(conn, parent)

    assert report["parentRollupState"] == "review_blocked"
    assert report["countsByRollupState"] == {"complete": 1, "review_blocked": 1}
    assert report["parent_child_matrix"]["remainingChildren"] == [pending]


def test_cross_lane_report_marks_autopilot_parent_coverage_and_reviewer_readiness(kanban_home):
    from hermes_cli.kanban_cross_lane_report import build_cross_lane_closeout_report

    with kb.connect() as conn:
        parent = kb.create_task(conn, title="autopilot parent", triage=True)
        finished = kb.create_task(conn, title="finished child", assignee="alice")
        pending = kb.create_task(conn, title="pending child", assignee="alice")
        kb.link_tasks(conn, parent, finished, relation_type="hierarchy")
        kb.link_tasks(conn, parent, pending, relation_type="hierarchy")
        assert kb.apply_closeout_transition(conn, finished, review_phase="review_ready", closeout_evidence=_review_ready_child_evidence(finished)) is True
        conn.execute("UPDATE tasks SET status='blocked', review_phase='worker_done', closeout_evidence=? WHERE id=?", (json.dumps({"schema": "kanban_closeout_evidence.v1", "summary": "worker only"}), pending))
        rollup = kb.autopilot_rollup_parent_review_ready(conn, parent, apply=False)

    report = build_cross_lane_closeout_report(evidence=rollup["package"], closeout_result={"allowed": False, "blockers": rollup["blockers"]}, parent_report=rollup["package"]["parent_child_matrix"])

    assert report["lane"] == "autopilot"
    assert report["dimensions"]["parent_child_coverage"]["status"] == "BLOCKED"
    assert report["dimensions"]["verifier_result"]["status"] == "BLOCKED"
    assert report["dimensions"]["completion_audit"]["status"] == "BLOCKED"
    assert report["dimensions"]["reviewer_readiness"]["status"] == "MISSING"
    assert "child_waiting_for_verifier" in report["blocker_keys"]


def test_cross_lane_report_marks_ultragoal_stale_evidence_and_formats_actionable_text():
    from hermes_cli.kanban_cross_lane_report import build_cross_lane_closeout_report, format_cross_lane_closeout_report

    evidence = {
        "schema": "kanban_closeout_evidence.v1",
        "ultragoal": {"runId": "BO-203", "state": "review_passed"},
        "done_criteria_ledger": {"criteria": [{"id": "criterion-1"}], "criteria_hash": "current"},
        "worker_evidence": {"criteria_hash": "stale", "per_criterion": {"criterion-1": {"claim": "satisfied"}}},
        "verifier_result": {"verdict": "PASS", "criteria_hash": "current", "per_criterion": {"criterion-1": {"verdict": "PASS"}}},
        "reviewer_result": {"recommendation": "APPROVE", "quality_gate": {"decision": "PASS"}},
        "quality_gate": {"decision": "PASS"},
        "cleanup": {"proof": "clean", "worktree_clean": True},
    }

    report = build_cross_lane_closeout_report(evidence=evidence, closeout_result={"allowed": False, "blockers": ["stale_goal_contract", "relation_drift_detected"]})
    text = format_cross_lane_closeout_report(report)

    assert report["lane"] == "ultragoal"
    assert report["dimensions"]["kanban_relation_drift"]["status"] == "BLOCKED"
    assert report["dimensions"]["lane_owned_evidence"]["status"] == "STALE"
    assert report["dimensions"]["worker_evidence"]["status"] == "STALE"
    assert report["dimensions"]["reviewer_readiness"]["status"] == "PASS"
    assert "lane: ultragoal" in text
    assert "kanban_relation_drift: BLOCKED" in text
    assert "blockers: relation_drift_detected, stale_goal_contract" in text


def test_cross_lane_report_exposes_cleanup_dimension_and_plain_text():
    from hermes_cli.kanban_cross_lane_report import build_cross_lane_closeout_report, format_cross_lane_closeout_report

    evidence = {
        "schema": "kanban_closeout_evidence.v1",
        "lane": "autopilot",
        "parent_child_matrix": {"parentRollupState": "complete", "remainingChildren": []},
        "worker_evidence": {"authority_boundary_confirmed": True, "per_criterion": {"criterion-1": {"claim": "satisfied"}}},
        "verifier_result": {"verdict": "PASS"},
        "reviewer_result": {"recommendation": "APPROVE", "quality_gate": {"decision": "PASS"}},
        "completion_audit": {"status": "PASS"},
        "cleanup": {"proof": "clean", "worktree_clean": True},
    }

    report = build_cross_lane_closeout_report(evidence=evidence, closeout_result={"allowed": True, "blockers": []})
    text = format_cross_lane_closeout_report(report)

    assert report["dimensions"]["cleanup"]["status"] == "PASS"
    assert report["dimensions"]["reviewer_readiness"]["status"] == "PASS"
    assert "cleanup: PASS" in text


def test_integrated_closeout_report_embeds_cross_lane_report():
    from hermes_cli.kanban_closeout_report import build_closeout_report

    evidence = {
        "lane": "ultragoal",
        "goal_contract": {"status": "pass"},
        "worker_evidence": {"authority_boundary_confirmed": True, "per_criterion": {"dc-1": {}}},
        "verifier_result": {"verdict": "pass"},
        "completion_audit": {"status": "PASS"},
        "cleanup": {"worktree_clean": True},
        "reviewer_readiness": {"status": "PASS"},
    }

    report = build_closeout_report(
        task_id="T-1",
        phase="review_ready",
        evidence=evidence,
        verification={"allowed": True, "blockers": []},
    )

    cross_lane = report["cross_lane_report"]
    assert cross_lane["schema"] == "kanban_cross_lane_closeout_report.v1"
    assert cross_lane["lane"] == "ultragoal"


def test_integrated_closeout_report_blocks_disallowed_verification():
    from hermes_cli.kanban_closeout_report import build_closeout_report

    evidence = {
        "lane": "ultragoal",
        "goal_contract": {"status": "pass"},
        "worker_evidence": {"authority_boundary_confirmed": True, "per_criterion": {"dc-1": {}}},
        "verifier_result": {"verdict": "pass"},
        "completion_audit": {"status": "PASS"},
        "cleanup": {"worktree_clean": True},
        "reviewer_readiness": {"status": "PASS"},
    }

    report = build_closeout_report(
        task_id="T-1",
        phase="review_ready",
        evidence=evidence,
        verification={"allowed": False, "blockers": ["verifier_result_not_pass"]},
    )

    assert report["ready"] is False
    assert report["dimensions"]["verifier_result"] == "BLOCKED"
    assert "verifier_result_not_pass" in report["blocker_keys"]


def test_integrated_closeout_report_accepts_complete_parent_review_package():
    from hermes_cli.kanban_closeout_report import build_closeout_report

    evidence = {
        "schema": "kanban_parent_review_package.v1",
        "parentRollupState": "complete",
        "remainingChildren": [],
        "parent_child_matrix": {"parentRollupState": "complete", "remainingChildren": []},
        "children": [{"taskId": "child-1", "review_phase": "review_ready"}],
    }

    report = build_closeout_report(
        task_id="parent-1",
        phase="review_ready",
        evidence=evidence,
        verification={"allowed": True, "blockers": []},
    )

    assert report["ready"] is True
    assert report["dimensions"]["worker_evidence"] == "PASS"
    assert report["dimensions"]["reviewer_readiness"] == "PASS"
    assert report["cross_lane_report"]["status"] == "PASS"
    assert report["cross_lane_report"]["dimensions"]["reviewer_readiness"]["status"] == "PASS"
    assert "worker_evidence:missing" not in report["blocker_keys"]
    assert "reviewer_readiness:missing" not in report["blocker_keys"]
