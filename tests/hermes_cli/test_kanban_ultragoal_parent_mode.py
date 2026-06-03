from __future__ import annotations

import json

import pytest


def _authority(task_id="BO-203", *, snapshot_hash="sha256:a", children=None, deps=None):
    return {
        "authority": "kanban",
        "taskId": task_id,
        "publicId": task_id,
        "status": "ready",
        "routingVerdict": "direct-kanban",
        "executionApproved": True,
        "snapshotHash": snapshot_hash,
        "doneCriteriaHash": "sha256:d",
        "doneCriteria": ["ship reviewed PR"],
        "children": children or [],
        "dependencies": deps or [],
    }


def test_parent_ultragoal_never_calls_kanban_dispatcher(tmp_path, monkeypatch):
    from hermes_cli import kanban_ultragoal
    from hermes_cli.kanban_ultragoal import KanbanUltragoalStore

    def explode(*args, **kwargs):  # pragma: no cover - called only on regression
        raise AssertionError("kanban dispatcher must not be used by Ultragoal parent mode")

    monkeypatch.setattr(kanban_ultragoal, "_forbidden_dispatcher_call", explode)
    store = KanbanUltragoalStore(tmp_path)
    run = store.start(
        "BO-203",
        authority=_authority(children=[{"id": "BO-204", "relationType": "hierarchy"}]),
        root_objective="Complete parent",
        target_mode="parent",
    )
    run = store.tick("BO-203", authority=_authority(children=[{"id": "BO-204", "relationType": "hierarchy"}]))

    run_json = json.loads((tmp_path / ".hermes" / "goal-runs" / "BO-203" / "run.json").read_text())
    assert run_json["dispatcherUsed"] is False
    assert run.pending_action["executor"] == "hermes-direct-goal-loop"


def test_parent_ultragoal_uses_hierarchy_children_as_subgoals(tmp_path):
    from hermes_cli.kanban_ultragoal import KanbanUltragoalStore

    authority = _authority(
        children=[
            {"id": "BO-204", "relationType": "hierarchy", "title": "first child"},
            {"id": "BO-205", "relationType": "dependency", "title": "dependency only"},
            {"id": "BO-206", "relationType": "hierarchy", "title": "second child"},
        ]
    )
    store = KanbanUltragoalStore(tmp_path)
    store.start("BO-203", authority=authority, root_objective="Complete parent", target_mode="parent")

    goals = json.loads((tmp_path / ".hermes" / "goal-runs" / "BO-203" / "goals.json").read_text())
    assert goals["targetMode"] == "parent"
    assert [g["sourceTaskId"] for g in goals["goals"]] == ["BO-204", "BO-206"]
    assert "BO-205" not in [g["sourceTaskId"] for g in goals["goals"]]


def test_parent_ultragoal_blocks_when_child_scope_drift_detected(tmp_path):
    from hermes_cli.kanban_ultragoal import KanbanUltragoalStore

    initial = _authority(children=[{"id": "BO-204", "relationType": "hierarchy"}])
    changed = _authority(children=[{"id": "BO-999", "relationType": "hierarchy"}])
    store = KanbanUltragoalStore(tmp_path)
    store.start("BO-203", authority=initial, root_objective="Complete parent", target_mode="parent")

    with pytest.raises(ValueError, match="childSnapshotHashes"):
        store.tick("BO-203", authority=changed)


def test_parent_ultragoal_dependency_edges_are_not_scope_membership(tmp_path):
    from hermes_cli.kanban_ultragoal import KanbanUltragoalStore

    authority = _authority(
        children=[{"id": "BO-204", "relationType": "hierarchy"}],
        deps=[{"id": "BO-300", "relationType": "dependency"}],
    )
    store = KanbanUltragoalStore(tmp_path)
    run = store.start("BO-203", authority=authority, root_objective="Complete parent", target_mode="parent")

    assert run.scope["childTaskIds"] == ["BO-204"]
    assert run.scope["dependencyEdges"] == [{"id": "BO-300", "relationType": "dependency"}]


def test_parent_ultragoal_terminal_report_summarizes_child_evidence(tmp_path):
    from hermes_cli.kanban_ultragoal import KanbanUltragoalStore

    authority = _authority(children=[{"id": "BO-204", "relationType": "hierarchy"}])
    store = KanbanUltragoalStore(tmp_path)
    store.start("BO-203", authority=authority, root_objective="Complete parent", target_mode="parent")
    store.record_worker_done("BO-203", authority=authority, evidence={"commandsRun": ["pytest"], "childEvidence": {"BO-204": "ok"}})
    store.record_verifier_result("BO-203", authority=authority, result={"passed": True, "doneCriteriaEvidence": [{"criterionId": "DC-1", "evidence": "pytest"}]})
    store.record_reviewer_result("BO-203", authority=authority, result={"recommendation": "APPROVE", "securityConcerns": [], "logicErrors": []})
    store.record_pr_created("BO-203", authority=authority, pr={"url": "https://github.com/chriskim12/hermes-agent/pull/1", "number": 1, "headSha": "abc"})
    store.record_ci_result("BO-203", authority=authority, ci={"state": "success", "headSha": "abc"})
    store.record_cleanup_proof("BO-203", authority=authority, proof={"status": "passed", "retained": [], "childCleanup": {"BO-204": {"status": "clean"}}})
    run = store.mark_review_ready("BO-203", authority=authority)

    assert run.last_terminal_report["targetMode"] == "parent"
    assert run.last_terminal_report["childEvidence"][0]["taskId"] == "BO-204"
    assert run.last_terminal_report["childCleanup"]["BO-204"]["status"] == "clean"


def test_parent_child_snapshot_hashes_pair_ids_with_their_own_child_rows(tmp_path):
    from hermes_cli.kanban_ultragoal import KanbanUltragoalStore, _sha256

    child = {"id": "BO-204", "relationType": "hierarchy", "title": "real child"}
    authority = _authority(children=[{"relationType": "hierarchy", "title": "malformed"}, child])
    store = KanbanUltragoalStore(tmp_path)
    run = store.start("BO-203", authority=authority, root_objective="Complete parent", target_mode="parent")

    assert run.scope["childTaskIds"] == ["BO-204"]
    assert run.scope["childSnapshotHashes"] == {"BO-204": _sha256(child)}
