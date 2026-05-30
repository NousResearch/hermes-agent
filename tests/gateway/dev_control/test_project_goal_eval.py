from unittest.mock import patch

import pytest

from gateway.dev_control.project_goal_eval import (
    assemble_evidence,
    check_machine_criteria,
    format_evidence_digest,
    goals_tick,
    judge_project_goal,
    project_goals_tick_enabled,
    reevaluate_project_goal,
)
from gateway.dev_control.project_goals import DevProjectGoalStore, create_project_goal


@pytest.fixture
def store(tmp_path):
    goal_store = DevProjectGoalStore(tmp_path / "state.db")
    yield goal_store
    goal_store.close()


def _subgoal_with_machine_criteria(store, *, include_manual: bool = True):
    vision = create_project_goal(store=store, kind="vision", title="Vision", status="active")
    goal = create_project_goal(
        store=store,
        kind="goal",
        title="Goal",
        parent_goal_id=vision["goal_id"],
        status="active",
    )
    milestone = create_project_goal(
        store=store,
        kind="milestone",
        title="Milestone",
        parent_goal_id=goal["goal_id"],
        status="active",
    )
    criteria = [
        {
            "statement": "Project goals tests pass",
            "verification_method": "command",
            "verification_detail": "scripts/run_tests.sh tests/gateway/dev_control/test_project_goals.py",
            "machine_checkable": True,
        },
    ]
    if include_manual:
        criteria.append({
            "statement": "Felipe confirms UX",
            "verification_method": "manual",
            "verification_detail": "Review manually.",
            "machine_checkable": False,
        })
    return create_project_goal(
        store=store,
        kind="subgoal",
        title="Wire routes",
        parent_goal_id=milestone["goal_id"],
        status="active",
        acceptance_criteria=criteria,
        payload={"plan_id": "devplan-test-1"},
    )


def test_check_machine_criteria_fails_without_verification():
    subgoal = {
        "acceptance_criteria": [
            {
                "statement": "Tests pass",
                "verification_method": "command",
                "verification_detail": "scripts/run_tests.sh tests/gateway/dev_control/test_project_goals.py",
                "machine_checkable": True,
            }
        ]
    }
    report = check_machine_criteria(subgoal, {"verification": {"results": []}})
    assert report.all_passed is False
    assert report.results[0]["passed"] is False


def test_check_machine_criteria_passes_with_matching_verification():
    subgoal = {
        "acceptance_criteria": [
            {
                "statement": "Tests pass",
                "verification_method": "command",
                "verification_detail": "scripts/run_tests.sh tests/gateway/dev_control/test_project_goals.py",
                "machine_checkable": True,
            }
        ]
    }
    evidence = {
        "verification": {
            "results": [
                {
                    "statement": "Tests pass",
                    "verification_detail": "scripts/run_tests.sh tests/gateway/dev_control/test_project_goals.py",
                    "passed": True,
                    "status": "passed",
                }
            ]
        }
    }
    report = check_machine_criteria(subgoal, evidence)
    assert report.all_passed is True
    assert len(report.manual_criteria) == 0


def test_reevaluate_auto_achieves_when_only_machine_criteria_pass(store):
    subgoal = _subgoal_with_machine_criteria(store, include_manual=False)
    evidence = {
        "verification": {
            "results": [
                {
                    "statement": "Project goals tests pass",
                    "verification_detail": "scripts/run_tests.sh tests/gateway/dev_control/test_project_goals.py",
                    "passed": True,
                }
            ]
        }
    }
    with patch("gateway.dev_control.project_goal_eval.assemble_evidence", return_value=evidence):
        result = reevaluate_project_goal(store=store, goal_id=subgoal["goal_id"])
    assert result["verdict"] == "done"
    assert store.get(subgoal["goal_id"])["status"] == "achieved"


def test_reevaluate_skips_judge_when_machine_gate_fails(store):
    subgoal = _subgoal_with_machine_criteria(store)
    with patch("gateway.dev_control.project_goal_eval.assemble_evidence", return_value={"verification": {"results": []}}):
        with patch("gateway.dev_control.project_goal_eval.judge_project_goal") as judge_mock:
            result = reevaluate_project_goal(store=store, goal_id=subgoal["goal_id"])
    judge_mock.assert_not_called()
    assert result["verdict"] == "continue"
    assert store.get(subgoal["goal_id"])["status"] == "active"


def test_reevaluate_idempotent_for_achieved_subgoal(store):
    subgoal = _subgoal_with_machine_criteria(store)
    store.update(subgoal["goal_id"], {"status": "achieved"})
    result = reevaluate_project_goal(store=store, goal_id=subgoal["goal_id"])
    assert result["verdict"] == "skipped"


def test_judge_project_goal_fail_open_on_api_error():
    with patch("agent.auxiliary_client.call_llm", side_effect=RuntimeError("boom")):
        verdict, reason, parse_failed = judge_project_goal(
            {"title": "Subgoal", "markdown": ""},
            format_evidence_digest({"verification": {"results": []}}),
            manual_criteria=[{"statement": "Manual check"}],
        )
    assert verdict == "continue"
    assert parse_failed is False
    assert "judge error" in reason


def test_goals_tick_idempotent(store, monkeypatch):
    monkeypatch.setenv("HERMES_DEV_PROJECT_GOALS_TICK", "1")
    assert project_goals_tick_enabled() is True
    subgoal = _subgoal_with_machine_criteria(store, include_manual=False)
    evidence = {
        "verification": {
            "results": [
                {
                    "statement": "Project goals tests pass",
                    "verification_detail": "scripts/run_tests.sh tests/gateway/dev_control/test_project_goals.py",
                    "passed": True,
                }
            ]
        }
    }
    with patch("gateway.dev_control.project_goal_eval.assemble_evidence", return_value=evidence):
        first = goals_tick(store=store, project_id=subgoal["project_id"])
        second = goals_tick(store=store, project_id=subgoal["project_id"])
    assert len(first["transitions"]) == 1
    assert second["transitions"] == []


def test_assemble_evidence_includes_execution_plan(tmp_path):
    from gateway.dev_execution import DevExecutionStore

    db_path = tmp_path / "state.db"
    execution_store = DevExecutionStore(db_path)
    execution_store.create_plan(
        title="Plan",
        vision_brief="brief",
        tasks=[{
            "task_id": "task-1",
            "goal": "Do work",
            "prompt": "prompt",
            "profile_id": "dev",
            "permissions": "verify",
            "acceptance_criteria": [],
            "status": "done",
            "payload": {"repo": "Felippen/hermes-agent", "branch": "feature/test"},
        }],
    )
    plan = execution_store.list_plans(limit=1)[0]
    subgoal = {"goal_id": "sub-1", "plan_artifact_id": None, "payload": {"plan_id": plan["plan_id"]}}
    evidence = assemble_evidence(
        subgoal,
        execution_store=execution_store,
        ci_fetcher=lambda **kwargs: {"state": "success", "repo": kwargs["repo"], "ref": kwargs["ref"]},
    )
    execution_store.close()
    assert len(evidence["execution_plan"]["tasks"]) == 1
    assert evidence["ci"]["state"] == "success"
