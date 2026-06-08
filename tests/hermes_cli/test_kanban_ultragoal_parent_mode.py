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


def _quality_gate():
    return {
        "architect_review": {
            "architecture_status": "CLEAR",
            "product_status": "CLEAR",
            "code_status": "CLEAR",
            "recommendation": "APPROVE",
            "evidence": ["reviews/final.json"],
            "blockers": [],
        },
        "verifier_qa": {
            "status": "passed",
            "contract_coverage": [
                {"done_criterion": "ship reviewed PR", "status": "covered", "evidence_refs": ["pytest"]}
            ],
            "surface_evidence": [
                {"surface": "cli", "invocation": "pytest", "verdict": "passed", "artifact_refs": ["pytest"]}
            ],
            "adversarial_cases": [
                {"scenario": "missing evidence", "expected_behavior": "block", "verdict": "passed", "artifact_refs": ["pytest"]}
            ],
        },
        "iteration": {"full_rerun": True, "rerun_commands": ["pytest"], "blockers": []},
    }


def _reviewer_approve(*, quality_gate=None):
    result = {"recommendation": "APPROVE", "securityConcerns": [], "logicErrors": []}
    if quality_gate is not None:
        result["quality_gate"] = quality_gate
    return result


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

    run_json = json.loads((store.root("BO-203") / "run.json").read_text())
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

    goals = json.loads((store.root("BO-203") / "goals.json").read_text())
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
    store.record_worker_done("BO-203", authority=authority, evidence={"commandsRun": ["pytest"], "authority_boundary_confirmed": True, "childEvidence": {"BO-204": {"status": "satisfied", "summary": "ok", "artifact_refs": ["pytest"]}}})
    store.record_verifier_result("BO-203", authority=authority, result={"passed": True, "doneCriteriaEvidence": [{"criterionId": "DC-1", "evidence": "pytest"}]})
    store.record_reviewer_result("BO-203", authority=authority, result=_reviewer_approve(quality_gate=_quality_gate()))
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


def test_goal_runs_are_stored_outside_target_workdir(tmp_path, monkeypatch):
    from hermes_cli.kanban_ultragoal import KanbanUltragoalStore

    hermes_home = tmp_path / "hermes-home"
    target_repo = tmp_path / "product-repo"
    hermes_home.mkdir()
    target_repo.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    store = KanbanUltragoalStore(target_repo)
    run = store.start(
        "BO-203",
        authority=_authority(children=[{"id": "BO-204", "relationType": "hierarchy"}]),
        root_objective="Complete parent",
        target_mode="parent",
    )

    assert run.run_id == "BO-203"
    assert not (target_repo / ".hermes" / "goal-runs" / "BO-203").exists()
    assert store.root("BO-203") == hermes_home / "goal-runs" / "product-repo" / "BO-203"
    assert (hermes_home / "goal-runs" / "product-repo" / "BO-203" / "ledger.jsonl").exists()


def test_build_closeout_evidence_materializes_governed_schema(tmp_path, monkeypatch):
    from hermes_cli.kanban_ultragoal import KanbanUltragoalStore

    hermes_home = tmp_path / "hermes-home"
    target_repo = tmp_path / "product-repo"
    hermes_home.mkdir()
    target_repo.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    authority = _authority(children=[{"id": "BO-204", "relationType": "hierarchy", "title": "child"}])
    store = KanbanUltragoalStore(target_repo)
    store.start("BO-203", authority=authority, root_objective="Complete parent", target_mode="parent")
    store.record_worker_done("BO-203", authority=authority, evidence={"summary": "implemented", "authority_boundary_confirmed": True, "childEvidence": {"BO-204": {"status": "satisfied", "summary": "ok", "artifact_refs": ["pytest"]}}})
    store.record_verifier_result("BO-203", authority=authority, result={"passed": True, "perCriterion": {"dc-1": "ok"}})
    store.record_reviewer_result("BO-203", authority=authority, result=_reviewer_approve(quality_gate=_quality_gate()))
    store.record_pr_created("BO-203", authority=authority, pr={"url": "https://github.com/o/r/pull/1", "number": 1, "headSha": "abc"})
    store.record_ci_result("BO-203", authority=authority, ci={"state": "success", "headSha": "abc", "checks": [{"name": "ci", "status": "completed", "conclusion": "success"}]})
    store.record_cleanup_proof("BO-203", authority=authority, proof={"status": "passed", "retained": [], "childCleanup": {"BO-204": {"status": "clean"}, "BO-205": {"status": "clean"}}})

    evidence = store.build_closeout_evidence("BO-203", authority=authority)

    assert evidence["done_criteria_ledger"]["schema"] == "kanban_done_criteria_ledger.v1"
    assert evidence["worker_evidence"]["schema"] == "kanban_worker_evidence.v1"
    assert evidence["verifier_result"]["schema"] == "kanban_verifier_result.v1"
    assert evidence["worker_evidence"]["criteria_hash"] == evidence["done_criteria_ledger"]["criteria_hash"]
    assert evidence["verifier_result"]["criteria_hash"] == evidence["done_criteria_ledger"]["criteria_hash"]
    assert evidence["authority_boundary_confirmed"] is True
    assert evidence["git"]["status_short"] == ""
    assert evidence["residue"]["items"][0]["disposition"] == "retained"
    assert evidence["review_package"]["kind"] == "pr_required"


def test_build_closeout_evidence_blocks_missing_quality_gate_matrix(tmp_path, monkeypatch):
    from hermes_cli.kanban_ultragoal import KanbanUltragoalStore

    hermes_home = tmp_path / "hermes-home"
    target_repo = tmp_path / "product-repo"
    hermes_home.mkdir()
    target_repo.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    authority = _authority(children=[{"id": "BO-204", "relationType": "hierarchy", "title": "child"}])
    store = KanbanUltragoalStore(target_repo)
    store.start("BO-203", authority=authority, root_objective="Complete parent", target_mode="parent")
    store.record_worker_done("BO-203", authority=authority, evidence={"summary": "implemented", "authority_boundary_confirmed": True, "childEvidence": {"BO-204": {"status": "satisfied", "summary": "ok", "artifact_refs": ["pytest"]}}})
    store.record_verifier_result("BO-203", authority=authority, result={"passed": True, "perCriterion": {"dc-1": "ok"}})
    store.record_reviewer_result("BO-203", authority=authority, result=_reviewer_approve())
    store.record_pr_created("BO-203", authority=authority, pr={"url": "https://github.com/o/r/pull/1", "number": 1, "headSha": "abc"})
    store.record_ci_result("BO-203", authority=authority, ci={"state": "success", "headSha": "abc", "checks": [{"name": "ci", "status": "completed", "conclusion": "success"}]})
    store.record_cleanup_proof("BO-203", authority=authority, proof={"status": "passed", "retained": [], "childCleanup": {"BO-204": {"status": "clean"}}})

    with pytest.raises(ValueError, match="quality_gate"):
        store.build_closeout_evidence("BO-203", authority=authority)


def test_parent_closeout_blocks_missing_child_evidence_matrix(tmp_path, monkeypatch):
    from hermes_cli.kanban_ultragoal import KanbanUltragoalStore

    hermes_home = tmp_path / "hermes-home"
    target_repo = tmp_path / "product-repo"
    hermes_home.mkdir()
    target_repo.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    authority = _authority(children=[{"id": "BO-204", "relationType": "hierarchy", "title": "child"}])
    store = KanbanUltragoalStore(target_repo)
    run = store.start("BO-203", authority=authority, root_objective="Complete parent", target_mode="parent")
    # Preserve the late closeout gate with a legacy/corrupt worker artifact that
    # predates the record_worker_done-time parent coverage gate.
    store._write_json(store.root("BO-203") / "evidence" / "worker.json", {"summary": "implemented", "childEvidence": {}})
    run.state = "worker_done"
    store.save_run(run)
    store.record_verifier_result("BO-203", authority=authority, result={"passed": True, "perCriterion": {"dc-1": "ok"}})
    store.record_reviewer_result("BO-203", authority=authority, result=_reviewer_approve(quality_gate=_quality_gate()))
    store.record_pr_created("BO-203", authority=authority, pr={"url": "https://github.com/o/r/pull/1", "number": 1, "headSha": "abc"})
    store.record_ci_result("BO-203", authority=authority, ci={"state": "success", "headSha": "abc", "checks": [{"name": "ci", "status": "completed", "conclusion": "success"}]})
    store.record_cleanup_proof("BO-203", authority=authority, proof={"status": "passed", "retained": [], "childCleanup": {"BO-204": {"status": "clean"}}})

    with pytest.raises(ValueError, match="childEvidence"):
        store.build_closeout_evidence("BO-203", authority=authority)


def test_parent_closeout_blocks_missing_child_cleanup_matrix(tmp_path, monkeypatch):
    from hermes_cli.kanban_ultragoal import KanbanUltragoalStore

    hermes_home = tmp_path / "hermes-home"
    target_repo = tmp_path / "product-repo"
    hermes_home.mkdir()
    target_repo.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    authority = _authority(children=[{"id": "BO-204", "relationType": "hierarchy", "title": "child"}])
    store = KanbanUltragoalStore(target_repo)
    store.start("BO-203", authority=authority, root_objective="Complete parent", target_mode="parent")
    store.record_worker_done("BO-203", authority=authority, evidence={"summary": "implemented", "authority_boundary_confirmed": True, "childEvidence": {"BO-204": {"status": "satisfied", "summary": "ok", "artifact_refs": ["pytest"]}}})
    store.record_verifier_result("BO-203", authority=authority, result={"passed": True, "perCriterion": {"dc-1": "ok"}})
    store.record_reviewer_result("BO-203", authority=authority, result=_reviewer_approve(quality_gate=_quality_gate()))
    store.record_pr_created("BO-203", authority=authority, pr={"url": "https://github.com/o/r/pull/1", "number": 1, "headSha": "abc"})
    store.record_ci_result("BO-203", authority=authority, ci={"state": "success", "headSha": "abc", "checks": [{"name": "ci", "status": "completed", "conclusion": "success"}]})
    store.record_cleanup_proof("BO-203", authority=authority, proof={"status": "passed", "retained": []})

    with pytest.raises(ValueError, match="childCleanup"):
        store.build_closeout_evidence("BO-203", authority=authority)


def test_closeout_review_ready_applies_children_before_parent(tmp_path, monkeypatch):
    from hermes_cli.kanban_ultragoal import KanbanUltragoalStore

    hermes_home = tmp_path / "hermes-home"
    target_repo = tmp_path / "product-repo"
    hermes_home.mkdir()
    target_repo.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    authority = _authority(children=[{"id": "BO-204", "relationType": "hierarchy"}, {"id": "BO-205", "relationType": "hierarchy"}])
    store = KanbanUltragoalStore(target_repo)
    store.start("BO-203", authority=authority, root_objective="Complete parent", target_mode="parent")
    store.record_worker_done("BO-203", authority=authority, evidence={"summary": "implemented", "authority_boundary_confirmed": True, "childEvidence": {"BO-204": {"status": "satisfied", "summary": "ok", "artifact_refs": ["pytest"]}, "BO-205": {"status": "satisfied", "summary": "ok", "artifact_refs": ["pytest"]}}})
    store.record_verifier_result("BO-203", authority=authority, result={"passed": True, "doneCriteriaEvidence": [{"criterionId": "uc-01", "evidence": "pytest"}]})
    store.record_reviewer_result("BO-203", authority=authority, result=_reviewer_approve(quality_gate=_quality_gate()))
    store.record_pr_created("BO-203", authority=authority, pr={"url": "https://github.com/o/r/pull/1", "number": 1, "headSha": "abc"})
    store.record_ci_result("BO-203", authority=authority, ci={"state": "success", "headSha": "abc", "checks": [{"name": "ci", "status": "completed", "conclusion": "success"}]})
    store.record_cleanup_proof("BO-203", authority=authority, proof={"status": "passed", "retained": [], "childCleanup": {"BO-204": {"status": "clean"}, "BO-205": {"status": "clean"}}})

    calls = []

    def fake_transition(conn, task_id, target_phase, evidence=None, **kwargs):
        calls.append((task_id, target_phase))
        return {"status": "transitioned", "verification": {"allowed": True, "blockers": []}}

    monkeypatch.setattr("hermes_cli.kanban_ultragoal.kanban_closeout.transition_task_closeout", fake_transition)

    result = store.closeout_review_ready("BO-203", authority=authority)

    assert result["status"] == "review_ready"
    assert calls == [
        ("BO-204", "worker_done"),
        ("BO-205", "worker_done"),
        ("BO-203", "worker_done"),
        ("BO-204", "review_ready"),
        ("BO-205", "review_ready"),
        ("BO-203", "review_ready"),
    ]


def test_cli_build_closeout_evidence_json(tmp_path, monkeypatch, capsys):
    import argparse
    from hermes_cli.kanban_ultragoal import KanbanUltragoalStore, build_parser, kanban_ultragoal_command

    hermes_home = tmp_path / "hermes-home"
    target_repo = tmp_path / "product-repo"
    hermes_home.mkdir()
    target_repo.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    authority = _authority(children=[{"id": "BO-204", "relationType": "hierarchy"}])
    store = KanbanUltragoalStore(target_repo)
    store.start("BO-203", authority=authority, root_objective="Complete parent", target_mode="parent")
    store.record_worker_done("BO-203", authority=authority, evidence={"summary": "implemented", "authority_boundary_confirmed": True, "childEvidence": {"BO-204": {"status": "satisfied", "summary": "ok", "artifact_refs": ["pytest"]}}})
    store.record_verifier_result("BO-203", authority=authority, result={"passed": True, "doneCriteriaEvidence": [{"criterionId": "uc-01", "evidence": "pytest"}]})
    store.record_reviewer_result("BO-203", authority=authority, result=_reviewer_approve(quality_gate=_quality_gate()))
    store.record_pr_created("BO-203", authority=authority, pr={"url": "https://github.com/o/r/pull/1", "number": 1, "headSha": "abc"})
    store.record_ci_result("BO-203", authority=authority, ci={"state": "success", "headSha": "abc"})
    store.record_cleanup_proof("BO-203", authority=authority, proof={"status": "passed", "retained": [], "childCleanup": {"BO-204": {"status": "clean"}}})

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")
    build_parser(sub)
    args = parser.parse_args([
        "kanban-ultragoal",
        "--workdir",
        str(target_repo),
        "--json",
        "build-closeout-evidence",
        "BO-203",
        "--authority-json",
        json.dumps(authority),
    ])

    assert kanban_ultragoal_command(args) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["schema"] == "kanban_closeout_evidence.v1"
    assert out["worker_evidence"]["schema"] == "kanban_worker_evidence.v1"


# ── Slice 1: RED tests for parent record_worker_done child coverage gate ──


def test_parent_worker_done_blocks_missing_child_coverage(tmp_path):
    from hermes_cli.kanban_ultragoal import KanbanUltragoalStore

    authority = _authority(children=[
        {"id": "BO-204", "relationType": "hierarchy", "title": "first child"},
        {"id": "BO-205", "relationType": "hierarchy", "title": "second child"},
    ])
    store = KanbanUltragoalStore(tmp_path)
    store.start("BO-203", authority=authority, root_objective="Complete parent", target_mode="parent")

    # evidence covers only BO-204, missing BO-205
    evidence = {
        "commandsRun": ["pytest"],
        "childEvidence": {
            "BO-204": {"status": "satisfied", "summary": "done", "artifact_refs": ["pr"]},
        },
    }

    with pytest.raises(ValueError, match="childEvidence"):
        store.record_worker_done("BO-203", authority=authority, evidence=evidence)


def test_parent_worker_done_accepts_satisfied_and_gated_children(tmp_path):
    from hermes_cli.kanban_ultragoal import KanbanUltragoalStore

    authority = _authority(children=[
        {"id": "BO-204", "relationType": "hierarchy", "title": "first child"},
        {"id": "BO-205", "relationType": "hierarchy", "title": "second child"},
    ])
    store = KanbanUltragoalStore(tmp_path)
    store.start("BO-203", authority=authority, root_objective="Complete parent", target_mode="parent")

    evidence = {
        "commandsRun": ["pytest"],
        "childEvidence": {
            "BO-204": {"status": "satisfied", "summary": "done", "artifact_refs": ["pr"]},
            "BO-205": {
                "status": "gated_by_forbidden_side_effect",
                "reason": "needs production access",
                "next_gate": "BO-230",
            },
        },
    }

    run = store.record_worker_done("BO-203", authority=authority, evidence=evidence)
    assert run.state == "worker_done"


def test_parent_worker_done_rejects_vague_deferred_child(tmp_path):
    from hermes_cli.kanban_ultragoal import KanbanUltragoalStore

    authority = _authority(children=[
        {"id": "BO-204", "relationType": "hierarchy", "title": "only child"},
    ])
    store = KanbanUltragoalStore(tmp_path)
    store.start("BO-203", authority=authority, root_objective="Complete parent", target_mode="parent")

    evidence = {
        "childEvidence": {
            "BO-204": {"status": "deferred", "summary": "will do later"},
        },
    }

    with pytest.raises(ValueError, match="deferred"):
        store.record_worker_done("BO-203", authority=authority, evidence=evidence)


def test_parent_worker_done_rejects_descoped_without_chris_approval(tmp_path):
    from hermes_cli.kanban_ultragoal import KanbanUltragoalStore

    authority = _authority(children=[
        {"id": "BO-204", "relationType": "hierarchy", "title": "only child"},
    ])
    store = KanbanUltragoalStore(tmp_path)
    store.start("BO-203", authority=authority, root_objective="Complete parent", target_mode="parent")

    evidence = {
        "childEvidence": {
            "BO-204": {
                "status": "explicitly_descoped_by_chris",
                "reason": "not needed",
                # missing approval.text / approvalText
            },
        },
    }

    with pytest.raises(ValueError, match="explicitly_descoped_by_chris"):
        store.record_worker_done("BO-203", authority=authority, evidence=evidence)


def test_parent_status_reports_incomplete_child_matrix(tmp_path):
    from hermes_cli.kanban_ultragoal import KanbanUltragoalStore

    authority = _authority(children=[
        {"id": "BO-204", "relationType": "hierarchy", "title": "first child"},
        {"id": "BO-205", "relationType": "hierarchy", "title": "second child"},
    ])
    store = KanbanUltragoalStore(tmp_path)
    store.start("BO-203", authority=authority, root_objective="Complete parent", target_mode="parent")

    payload = store.status_payload("BO-203")

    matrix = payload["parentChildMatrix"]
    assert matrix["parentScopeComplete"] is False
    assert matrix["missingChildEvidence"] == ["BO-204", "BO-205"]
    assert matrix["missingChildCleanup"] == ["BO-204", "BO-205"]
    assert matrix["nextRequiredChild"] == "BO-204"
    assert matrix["cannotFinalCloseoutParent"] is True


def test_parent_worker_done_rejects_unknown_child_as_coverage(tmp_path):
    from hermes_cli.kanban_ultragoal import KanbanUltragoalStore

    authority = _authority(children=[{"id": "BO-204", "relationType": "hierarchy", "title": "only child"}])
    store = KanbanUltragoalStore(tmp_path)
    store.start("BO-203", authority=authority, root_objective="Complete parent", target_mode="parent")

    with pytest.raises(ValueError, match="missing childEvidence for BO-204"):
        store.record_worker_done("BO-203", authority=authority, evidence={
            "childEvidence": {"BO-999": {"status": "satisfied", "summary": "wrong child", "artifact_refs": ["pytest"]}}
        })


def test_parent_worker_done_rejects_non_dict_child_evidence(tmp_path):
    from hermes_cli.kanban_ultragoal import KanbanUltragoalStore

    authority = _authority(children=[{"id": "BO-204", "relationType": "hierarchy", "title": "only child"}])
    store = KanbanUltragoalStore(tmp_path)
    store.start("BO-203", authority=authority, root_objective="Complete parent", target_mode="parent")

    with pytest.raises(ValueError, match="childEvidence:not_dict"):
        store.record_worker_done("BO-203", authority=authority, evidence={"childEvidence": ["BO-204"]})


def test_parent_worker_done_rejects_non_dict_child_row(tmp_path):
    from hermes_cli.kanban_ultragoal import KanbanUltragoalStore

    authority = _authority(children=[{"id": "BO-204", "relationType": "hierarchy", "title": "only child"}])
    store = KanbanUltragoalStore(tmp_path)
    store.start("BO-203", authority=authority, root_objective="Complete parent", target_mode="parent")

    with pytest.raises(ValueError, match="childEvidence_row:not_dict"):
        store.record_worker_done("BO-203", authority=authority, evidence={"childEvidence": {"BO-204": "ok"}})


def test_parent_worker_done_rejects_missing_status_or_reason_summary(tmp_path):
    from hermes_cli.kanban_ultragoal import KanbanUltragoalStore

    authority = _authority(children=[{"id": "BO-204", "relationType": "hierarchy", "title": "only child"}])
    store = KanbanUltragoalStore(tmp_path)
    store.start("BO-203", authority=authority, root_objective="Complete parent", target_mode="parent")

    with pytest.raises(ValueError, match="status:missing"):
        store.record_worker_done("BO-203", authority=authority, evidence={"childEvidence": {"BO-204": {"summary": "no status", "artifact_refs": ["pytest"]}}})

    with pytest.raises(ValueError, match="summary_or_reason_required"):
        store.record_worker_done("BO-203", authority=authority, evidence={"childEvidence": {"BO-204": {"status": "satisfied", "artifact_refs": ["pytest"]}}})



def test_build_closeout_evidence_blocks_incomplete_verifier_per_criterion_proof(tmp_path, monkeypatch):
    from hermes_cli.kanban_ultragoal import KanbanUltragoalStore

    hermes_home = tmp_path / "hermes-home"
    target_repo = tmp_path / "product-repo"
    hermes_home.mkdir()
    target_repo.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    authority = _authority(children=[{"id": "BO-204", "relationType": "hierarchy", "title": "child"}])
    authority["doneCriteria"] = ["first criterion", "second criterion"]
    store = KanbanUltragoalStore(target_repo)
    store.start("BO-203", authority=authority, root_objective="Complete parent", target_mode="parent")
    store.record_worker_done(
        "BO-203",
        authority=authority,
        evidence={
            "summary": "implemented",
            "commandsRun": ["pytest tests/hermes_cli/test_kanban_ultragoal_parent_mode.py"],
            "childEvidence": {"BO-204": {"status": "satisfied", "summary": "ok", "artifact_refs": ["pytest"]}},
        },
    )
    store.record_verifier_result(
        "BO-203",
        authority=authority,
        result={"passed": True, "doneCriteriaEvidence": [{"criterionId": "uc-01", "evidence": "pytest"}]},
    )
    store.record_reviewer_result("BO-203", authority=authority, result=_reviewer_approve(quality_gate=_quality_gate()))
    store.record_pr_created("BO-203", authority=authority, pr={"url": "https://github.com/o/r/pull/1", "number": 1, "headSha": "abc"})
    store.record_ci_result("BO-203", authority=authority, ci={"state": "success", "headSha": "abc"})
    store.record_cleanup_proof("BO-203", authority=authority, proof={"status": "passed", "retained": [], "childCleanup": {"BO-204": {"status": "clean"}}})

    with pytest.raises(ValueError, match="missing_verifier_per_criterion:uc-02"):
        store.build_closeout_evidence("BO-203", authority=authority)


def test_build_closeout_evidence_blocks_stale_worker_goal_contract_hash(tmp_path, monkeypatch):
    from hermes_cli.kanban_ultragoal import KanbanUltragoalStore

    hermes_home = tmp_path / "hermes-home"
    target_repo = tmp_path / "product-repo"
    hermes_home.mkdir()
    target_repo.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    authority = _authority(children=[{"id": "BO-204", "relationType": "hierarchy", "title": "child"}])
    authority["doneCriteria"] = ["ship current approved contract"]
    store = KanbanUltragoalStore(target_repo)
    store.start("BO-203", authority=authority, root_objective="Complete parent", target_mode="parent")
    store.record_worker_done(
        "BO-203",
        authority=authority,
        evidence={
            "summary": "implemented against stale contract",
            "commandsRun": ["pytest tests/hermes_cli/test_kanban_ultragoal_parent_mode.py"],
            "authoritySnapshotHash": "sha256:stale",
            "doneCriteriaHash": authority["doneCriteriaHash"],
            "childEvidence": {"BO-204": {"status": "satisfied", "summary": "ok", "artifact_refs": ["pytest"]}},
        },
    )
    store.record_verifier_result(
        "BO-203",
        authority=authority,
        result={"passed": True, "doneCriteriaEvidence": [{"criterionId": "uc-01", "evidence": "pytest"}]},
    )
    store.record_reviewer_result("BO-203", authority=authority, result=_reviewer_approve(quality_gate=_quality_gate()))
    store.record_pr_created("BO-203", authority=authority, pr={"url": "https://github.com/o/r/pull/1", "number": 1, "headSha": "abc"})
    store.record_ci_result("BO-203", authority=authority, ci={"state": "success", "headSha": "abc"})
    store.record_cleanup_proof("BO-203", authority=authority, proof={"status": "passed", "retained": [], "childCleanup": {"BO-204": {"status": "clean"}}})

    with pytest.raises(ValueError, match="stale_worker_authority_snapshot"):
        store.build_closeout_evidence("BO-203", authority=authority)


def test_stale_ultragoal_closeout_evidence_requires_current_goal_contract_identity(tmp_path, monkeypatch):
    from hermes_cli.kanban_ultragoal import KanbanUltragoalStore

    hermes_home = tmp_path / "hermes-home"
    target_repo = tmp_path / "product-repo"
    hermes_home.mkdir()
    target_repo.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    authority = _authority(children=[{"id": "BO-204", "relationType": "hierarchy", "title": "child"}])
    store = KanbanUltragoalStore(target_repo)
    store.start("BO-203", authority=authority, root_objective="Complete parent", target_mode="parent")
    store.record_worker_done(
        "BO-203",
        authority=authority,
        evidence={
            "summary": "implemented against a superseded contract",
            "commandsRun": ["pytest tests/hermes_cli/test_kanban_ultragoal_parent_mode.py"],
            "goalContract": {"taskId": "BO-203", "snapshotHash": "sha256:superseded", "doneCriteriaHash": "sha256:superseded"},
            "childEvidence": {"BO-204": {"status": "satisfied", "summary": "ok", "artifact_refs": ["pytest"]}},
        },
    )
    store.record_verifier_result("BO-203", authority=authority, result={"passed": True, "perCriterion": {"uc-01": "ok"}})
    store.record_reviewer_result("BO-203", authority=authority, result=_reviewer_approve(quality_gate=_quality_gate()))
    store.record_pr_created("BO-203", authority=authority, pr={"url": "https://github.com/o/r/pull/1", "number": 1, "headSha": "abc"})
    store.record_ci_result("BO-203", authority=authority, ci={"state": "success", "headSha": "abc"})
    store.record_cleanup_proof("BO-203", authority=authority, proof={"status": "passed", "retained": [], "childCleanup": {"BO-204": {"status": "clean"}}})

    with pytest.raises(ValueError) as exc:
        store.build_closeout_evidence("BO-203", authority=authority)

    assert "ultragoal_closeout_stale_contract" in str(exc.value)



def test_stale_ultragoal_closeout_evidence_reports_superseded_contract_key(tmp_path, monkeypatch):
    from hermes_cli.kanban_ultragoal import KanbanUltragoalStore

    hermes_home = tmp_path / "hermes-home"
    target_repo = tmp_path / "product-repo"
    hermes_home.mkdir()
    target_repo.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    authority = _authority(children=[{"id": "BO-204", "relationType": "hierarchy", "title": "child"}])
    store = KanbanUltragoalStore(target_repo)
    store.start("BO-203", authority=authority, root_objective="Complete parent", target_mode="parent")
    store.record_worker_done(
        "BO-203",
        authority=authority,
        evidence={
            "summary": "implemented by superseded worker evidence",
            "commandsRun": ["pytest tests/hermes_cli/test_kanban_ultragoal_parent_mode.py"],
            "superseded": True,
            "authority_boundary_confirmed": True,
            "childEvidence": {"BO-204": {"status": "satisfied", "summary": "ok", "artifact_refs": ["pytest"]}},
        },
    )
    store.record_verifier_result("BO-203", authority=authority, result={"passed": True, "perCriterion": {"uc-01": "ok"}})
    store.record_reviewer_result("BO-203", authority=authority, result=_reviewer_approve(quality_gate=_quality_gate()))
    store.record_pr_created("BO-203", authority=authority, pr={"url": "https://github.com/o/r/pull/1", "number": 1, "headSha": "abc"})
    store.record_ci_result("BO-203", authority=authority, ci={"state": "success", "headSha": "abc"})
    store.record_cleanup_proof("BO-203", authority=authority, proof={"status": "passed", "retained": [], "childCleanup": {"BO-204": {"status": "clean"}}})

    with pytest.raises(ValueError) as exc:
        store.build_closeout_evidence("BO-203", authority=authority)

    assert "ultragoal_closeout_stale_contract" in str(exc.value)


def test_incomplete_ultragoal_closeout_evidence_reports_concrete_blocker_keys(tmp_path, monkeypatch):
    from hermes_cli.kanban_ultragoal import KanbanUltragoalStore

    hermes_home = tmp_path / "hermes-home"
    target_repo = tmp_path / "product-repo"
    hermes_home.mkdir()
    target_repo.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    authority = _authority(children=[{"id": "BO-204", "relationType": "hierarchy", "title": "child"}])
    authority["doneCriteria"] = ["first criterion", "second criterion"]
    store = KanbanUltragoalStore(target_repo)
    store.start("BO-203", authority=authority, root_objective="Complete parent", target_mode="parent")
    store._write_json(store.root("BO-203") / "evidence" / "worker.json", {"childEvidence": {"BO-204": {"status": "satisfied", "summary": "ok", "artifact_refs": ["pytest"]}}})
    store._write_json(store.root("BO-203") / "verifier" / "result.json", {"passed": True, "doneCriteriaEvidence": [{"criterionId": "uc-01", "evidence": "pytest"}]})
    run = store.load_run("BO-203")
    run.state = "verification_passed"
    store.save_run(run)
    store.record_reviewer_result("BO-203", authority=authority, result=_reviewer_approve(quality_gate=_quality_gate()))
    store.record_pr_created("BO-203", authority=authority, pr={"url": "https://github.com/o/r/pull/1", "number": 1, "headSha": "abc"})
    store.record_ci_result("BO-203", authority=authority, ci={"state": "success", "headSha": "abc"})
    store.record_cleanup_proof("BO-203", authority=authority, proof={"status": "passed", "retained": [], "childCleanup": {"BO-204": {"status": "clean"}}})

    with pytest.raises(ValueError) as exc:
        store.build_closeout_evidence("BO-203", authority=authority)

    message = str(exc.value)
    assert "missing_verifier_per_criterion:uc-02" in message
    assert "ultragoal_closeout_missing_verifier_evidence" in message
    assert "ultragoal_closeout_missing_per_criterion_evidence" in message


def test_incomplete_ultragoal_closeout_evidence_requires_non_empty_verifier_matrix(tmp_path, monkeypatch):
    from hermes_cli.kanban_ultragoal import KanbanUltragoalStore

    hermes_home = tmp_path / "hermes-home"
    target_repo = tmp_path / "product-repo"
    hermes_home.mkdir()
    target_repo.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    authority = _authority(children=[{"id": "BO-204", "relationType": "hierarchy", "title": "child"}])
    authority["doneCriteria"] = ["first criterion"]
    store = KanbanUltragoalStore(target_repo)
    store.start("BO-203", authority=authority, root_objective="Complete parent", target_mode="parent")
    store.record_worker_done(
        "BO-203",
        authority=authority,
        evidence={
            "summary": "implemented",
            "commandsRun": ["pytest tests/hermes_cli/test_kanban_ultragoal_parent_mode.py"],
            "authority_boundary_confirmed": True,
            "childEvidence": {"BO-204": {"status": "satisfied", "summary": "ok", "artifact_refs": ["pytest"]}},
        },
    )
    store.record_verifier_result("BO-203", authority=authority, result={"passed": True})
    store.record_reviewer_result("BO-203", authority=authority, result=_reviewer_approve(quality_gate=_quality_gate()))
    store.record_pr_created("BO-203", authority=authority, pr={"url": "https://github.com/o/r/pull/1", "number": 1, "headSha": "abc"})
    store.record_ci_result("BO-203", authority=authority, ci={"state": "success", "headSha": "abc"})
    store.record_cleanup_proof("BO-203", authority=authority, proof={"status": "passed", "retained": [], "childCleanup": {"BO-204": {"status": "clean"}}})

    with pytest.raises(ValueError) as exc:
        store.build_closeout_evidence("BO-203", authority=authority)

    assert "ultragoal_closeout_missing_per_criterion_evidence" in str(exc.value)


def test_build_closeout_evidence_blocks_missing_worker_authority_confirmation(tmp_path, monkeypatch):
    from hermes_cli.kanban_ultragoal import KanbanUltragoalStore

    hermes_home = tmp_path / "hermes-home"
    target_repo = tmp_path / "product-repo"
    hermes_home.mkdir()
    target_repo.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    authority = _authority(children=[{"id": "BO-204", "relationType": "hierarchy", "title": "child"}])
    store = KanbanUltragoalStore(target_repo)
    store.start("BO-203", authority=authority, root_objective="Complete parent", target_mode="parent")
    store.record_worker_done(
        "BO-203",
        authority=authority,
        evidence={
            "summary": "implemented",
            "childEvidence": {"BO-204": {"status": "satisfied", "summary": "ok", "artifact_refs": ["pytest"]}},
        },
    )
    store.record_verifier_result("BO-203", authority=authority, result={"passed": True, "perCriterion": {"uc-01": {"verdict": "PASS"}}})
    store.record_reviewer_result("BO-203", authority=authority, result=_reviewer_approve(quality_gate=_quality_gate()))
    store.record_pr_created("BO-203", authority=authority, pr={"url": "https://github.com/o/r/pull/1", "number": 1, "headSha": "abc"})
    store.record_ci_result("BO-203", authority=authority, ci={"state": "success", "headSha": "abc"})
    store.record_cleanup_proof("BO-203", authority=authority, proof={"status": "passed", "retained": [], "childCleanup": {"BO-204": {"status": "clean"}}})

    with pytest.raises(ValueError) as exc:
        store.build_closeout_evidence("BO-203", authority=authority)

    assert "authority_boundary_confirmed" in str(exc.value)


def test_build_closeout_evidence_blocks_failed_verifier_per_criterion(tmp_path, monkeypatch):
    from hermes_cli.kanban_ultragoal import KanbanUltragoalStore

    hermes_home = tmp_path / "hermes-home"
    target_repo = tmp_path / "product-repo"
    hermes_home.mkdir()
    target_repo.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    authority = _authority(children=[{"id": "BO-204", "relationType": "hierarchy", "title": "child"}])
    store = KanbanUltragoalStore(target_repo)
    store.start("BO-203", authority=authority, root_objective="Complete parent", target_mode="parent")
    store.record_worker_done(
        "BO-203",
        authority=authority,
        evidence={
            "summary": "implemented",
            "authority_boundary_confirmed": True,
            "childEvidence": {"BO-204": {"status": "satisfied", "summary": "ok", "artifact_refs": ["pytest"]}},
        },
    )
    store.record_verifier_result("BO-203", authority=authority, result={"passed": True, "perCriterion": {"uc-01": {"verdict": "FAIL"}}})
    store.record_reviewer_result("BO-203", authority=authority, result=_reviewer_approve(quality_gate=_quality_gate()))
    store.record_pr_created("BO-203", authority=authority, pr={"url": "https://github.com/o/r/pull/1", "number": 1, "headSha": "abc"})
    store.record_ci_result("BO-203", authority=authority, ci={"state": "success", "headSha": "abc"})
    store.record_cleanup_proof("BO-203", authority=authority, proof={"status": "passed", "retained": [], "childCleanup": {"BO-204": {"status": "clean"}}})

    with pytest.raises(ValueError) as exc:
        store.build_closeout_evidence("BO-203", authority=authority)

    assert "verifier_per_criterion_not_pass:uc-01" in str(exc.value)


def test_build_closeout_evidence_blocks_empty_verifier_per_criterion(tmp_path, monkeypatch):
    from hermes_cli.kanban_ultragoal import KanbanUltragoalStore

    hermes_home = tmp_path / "hermes-home"
    target_repo = tmp_path / "product-repo"
    hermes_home.mkdir()
    target_repo.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    authority = _authority(children=[{"id": "BO-204", "relationType": "hierarchy", "title": "child"}])
    store = KanbanUltragoalStore(target_repo)
    store.start("BO-203", authority=authority, root_objective="Complete parent", target_mode="parent")
    store.record_worker_done(
        "BO-203",
        authority=authority,
        evidence={
            "summary": "implemented",
            "authority_boundary_confirmed": True,
            "childEvidence": {"BO-204": {"status": "satisfied", "summary": "ok", "artifact_refs": ["pytest"]}},
        },
    )
    store.record_verifier_result("BO-203", authority=authority, result={"passed": True, "perCriterion": {"uc-01": {}}})
    store.record_reviewer_result("BO-203", authority=authority, result=_reviewer_approve(quality_gate=_quality_gate()))
    store.record_pr_created("BO-203", authority=authority, pr={"url": "https://github.com/o/r/pull/1", "number": 1, "headSha": "abc"})
    store.record_ci_result("BO-203", authority=authority, ci={"state": "success", "headSha": "abc"})
    store.record_cleanup_proof("BO-203", authority=authority, proof={"status": "passed", "retained": [], "childCleanup": {"BO-204": {"status": "clean"}}})

    with pytest.raises(ValueError) as exc:
        store.build_closeout_evidence("BO-203", authority=authority)

    assert "verifier_per_criterion_not_pass:uc-01" in str(exc.value)


def test_build_closeout_evidence_blocks_conflicting_duplicate_verifier_alias(tmp_path, monkeypatch):
    from hermes_cli.kanban_ultragoal import KanbanUltragoalStore

    hermes_home = tmp_path / "hermes-home"
    target_repo = tmp_path / "product-repo"
    hermes_home.mkdir()
    target_repo.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    authority = _authority(children=[{"id": "BO-204", "relationType": "hierarchy", "title": "child"}])
    store = KanbanUltragoalStore(target_repo)
    store.start("BO-203", authority=authority, root_objective="Complete parent", target_mode="parent")
    store.record_worker_done(
        "BO-203",
        authority=authority,
        evidence={
            "summary": "implemented",
            "authority_boundary_confirmed": True,
            "childEvidence": {"BO-204": {"status": "satisfied", "summary": "ok", "artifact_refs": ["pytest"]}},
        },
    )
    store.record_verifier_result(
        "BO-203",
        authority=authority,
        result={
            "passed": True,
            "doneCriteriaEvidence": [
                {"criterionId": "uc-01", "verdict": "FAIL"},
                {"criterionId": "dc-01", "verdict": "PASS"},
            ],
        },
    )
    store.record_reviewer_result("BO-203", authority=authority, result=_reviewer_approve(quality_gate=_quality_gate()))
    store.record_pr_created("BO-203", authority=authority, pr={"url": "https://github.com/o/r/pull/1", "number": 1, "headSha": "abc"})
    store.record_ci_result("BO-203", authority=authority, ci={"state": "success", "headSha": "abc"})
    store.record_cleanup_proof("BO-203", authority=authority, proof={"status": "passed", "retained": [], "childCleanup": {"BO-204": {"status": "clean"}}})

    with pytest.raises(ValueError) as exc:
        store.build_closeout_evidence("BO-203", authority=authority)

    assert "verifier_per_criterion_not_pass:uc-01" in str(exc.value)


def test_build_closeout_evidence_blocks_empty_duplicate_verifier_alias(tmp_path, monkeypatch):
    from hermes_cli.kanban_ultragoal import KanbanUltragoalStore

    hermes_home = tmp_path / "hermes-home"
    target_repo = tmp_path / "product-repo"
    hermes_home.mkdir()
    target_repo.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    authority = _authority(children=[{"id": "BO-204", "relationType": "hierarchy", "title": "child"}])
    store = KanbanUltragoalStore(target_repo)
    store.start("BO-203", authority=authority, root_objective="Complete parent", target_mode="parent")
    store.record_worker_done(
        "BO-203",
        authority=authority,
        evidence={
            "summary": "implemented",
            "authority_boundary_confirmed": True,
            "childEvidence": {"BO-204": {"status": "satisfied", "summary": "ok", "artifact_refs": ["pytest"]}},
        },
    )
    store.record_verifier_result(
        "BO-203",
        authority=authority,
        result={
            "passed": True,
            "doneCriteriaEvidence": [
                {"criterionId": "uc-01", "verdict": "PASS"},
                {"criterionId": "dc-01"},
            ],
        },
    )
    store.record_reviewer_result("BO-203", authority=authority, result=_reviewer_approve(quality_gate=_quality_gate()))
    store.record_pr_created("BO-203", authority=authority, pr={"url": "https://github.com/o/r/pull/1", "number": 1, "headSha": "abc"})
    store.record_ci_result("BO-203", authority=authority, ci={"state": "success", "headSha": "abc"})
    store.record_cleanup_proof("BO-203", authority=authority, proof={"status": "passed", "retained": [], "childCleanup": {"BO-204": {"status": "clean"}}})

    with pytest.raises(ValueError) as exc:
        store.build_closeout_evidence("BO-203", authority=authority)

    assert "verifier_per_criterion_not_pass:uc-01" in str(exc.value)


def test_build_closeout_evidence_blocks_error_duplicate_verifier_alias(tmp_path, monkeypatch):
    from hermes_cli.kanban_ultragoal import KanbanUltragoalStore

    hermes_home = tmp_path / "hermes-home"
    target_repo = tmp_path / "product-repo"
    hermes_home.mkdir()
    target_repo.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    authority = _authority(children=[{"id": "BO-204", "relationType": "hierarchy", "title": "child"}])
    store = KanbanUltragoalStore(target_repo)
    store.start("BO-203", authority=authority, root_objective="Complete parent", target_mode="parent")
    store.record_worker_done(
        "BO-203",
        authority=authority,
        evidence={
            "summary": "implemented",
            "authority_boundary_confirmed": True,
            "childEvidence": {"BO-204": {"status": "satisfied", "summary": "ok", "artifact_refs": ["pytest"]}},
        },
    )
    store.record_verifier_result(
        "BO-203",
        authority=authority,
        result={
            "passed": True,
            "doneCriteriaEvidence": [
                {"criterionId": "uc-01", "verdict": "PASS"},
                {"criterionId": "dc-01", "verdict": "ERROR"},
            ],
        },
    )
    store.record_reviewer_result("BO-203", authority=authority, result=_reviewer_approve(quality_gate=_quality_gate()))
    store.record_pr_created("BO-203", authority=authority, pr={"url": "https://github.com/o/r/pull/1", "number": 1, "headSha": "abc"})
    store.record_ci_result("BO-203", authority=authority, ci={"state": "success", "headSha": "abc"})
    store.record_cleanup_proof("BO-203", authority=authority, proof={"status": "passed", "retained": [], "childCleanup": {"BO-204": {"status": "clean"}}})

    with pytest.raises(ValueError) as exc:
        store.build_closeout_evidence("BO-203", authority=authority)

    assert "verifier_per_criterion_not_pass:uc-01" in str(exc.value)
