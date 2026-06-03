from __future__ import annotations

import json

import pytest


def _authority(task_id="BO-203"):
    return {
        "authority": "kanban",
        "taskId": task_id,
        "publicId": task_id,
        "status": "ready",
        "routingVerdict": "direct-kanban",
        "executionApproved": True,
        "snapshotHash": "sha256:a",
        "doneCriteriaHash": "sha256:d",
        "doneCriteria": ["ship reviewed PR"],
    }


def _ready_for_ci(store, authority):
    store.start("BO-203", authority=authority, root_objective="Bring one reviewed PR")
    store.record_worker_done("BO-203", authority=authority, evidence={"commandsRun": ["pytest"]})
    store.record_verifier_result("BO-203", authority=authority, result={"passed": True, "doneCriteriaEvidence": [{"criterionId": "DC-1", "evidence": "pytest"}]})
    store.record_reviewer_result("BO-203", authority=authority, result={"recommendation": "APPROVE", "securityConcerns": [], "logicErrors": []})
    store.record_pr_created("BO-203", authority=authority, pr={"url": "https://github.com/chriskim12/hermes-agent/pull/1", "number": 1, "headSha": "abc"})
    store.record_ci_result("BO-203", authority=authority, ci={"state": "success", "headSha": "abc"})


def test_ultragoal_review_ready_blocks_without_cleanup_proof(tmp_path):
    from hermes_cli.kanban_ultragoal import KanbanUltragoalStore

    store = KanbanUltragoalStore(tmp_path)
    authority = _authority()
    _ready_for_ci(store, authority)

    with pytest.raises(ValueError, match="cleanup proof"):
        store.mark_review_ready("BO-203", authority=authority)


def test_ultragoal_records_retained_worktree_reason_and_ttl(tmp_path):
    from hermes_cli.kanban_ultragoal import KanbanUltragoalStore

    store = KanbanUltragoalStore(tmp_path)
    authority = _authority()
    store.start("BO-203", authority=authority, root_objective="Bring one reviewed PR")
    proof = store.record_cleanup_proof(
        "BO-203",
        authority=authority,
        proof={
            "status": "passed",
            "retained": [{"path": "impl-worktree", "reason": "PR review iteration", "ttl": "after PR merge/close"}],
        },
    )

    cleanup = json.loads((tmp_path / ".hermes" / "goal-runs" / "BO-203" / "cleanup.json").read_text())
    assert cleanup["readOnlyProof"] is True
    assert cleanup["retained"][0]["reason"] == "PR review iteration"
    assert cleanup["retained"][0]["ttl"] == "after PR merge/close"
    assert proof.state == "admitted"


def test_ultragoal_does_not_remove_dirty_or_active_worktree(tmp_path):
    from hermes_cli.ultragoal_cleanup import classify_cleanup_candidates

    results = classify_cleanup_candidates(
        [
            {"path": "dirty", "kind": "implementation_worktree", "dirty": True, "activeCwd": False},
            {"path": "active", "kind": "implementation_worktree", "dirty": False, "activeCwd": True},
            {"path": "cache", "kind": "cache", "dirty": False, "activeCwd": False},
        ]
    )

    assert results["dirty"]["action"] == "preserve"
    assert results["dirty"]["reason"] == "dirty_worktree"
    assert results["active"]["action"] == "preserve"
    assert results["active"]["reason"] == "active_cwd"
    assert results["cache"]["action"] == "delete_candidate"


def test_ultragoal_cleanup_policy_classifies_delete_preserve_never_touch(tmp_path):
    from hermes_cli.ultragoal_cleanup import classify_cleanup_candidates

    results = classify_cleanup_candidates(
        [
            {"path": ".pytest_cache", "kind": "cache"},
            {"path": ".hermes/goal-runs/BO-203/run.json", "kind": "run_artifact", "referencedByEvidence": True},
            {"path": ".env", "kind": "secret"},
            {"path": "/repo", "kind": "canonical_checkout"},
        ]
    )

    assert results[".pytest_cache"]["action"] == "delete_candidate"
    assert results[".hermes/goal-runs/BO-203/run.json"]["action"] == "preserve"
    assert results[".env"]["action"] == "never_touch"
    assert results["/repo"]["action"] == "never_touch"


def test_ultragoal_parent_terminal_report_includes_child_cleanup_matrix(tmp_path):
    from hermes_cli.kanban_ultragoal import KanbanUltragoalStore

    authority = _authority()
    authority["children"] = [{"id": "BO-204", "relationType": "hierarchy"}]
    store = KanbanUltragoalStore(tmp_path)
    store.start("BO-203", authority=authority, root_objective="Complete parent", target_mode="parent")
    store.record_cleanup_proof("BO-203", authority=authority, proof={"status": "passed", "childCleanup": {"BO-204": {"status": "retained", "ttl": "after PR"}}})
    cleanup = json.loads((tmp_path / ".hermes" / "goal-runs" / "BO-203" / "cleanup.json").read_text())
    assert cleanup["childCleanup"]["BO-204"]["status"] == "retained"


def test_ultragoal_post_merge_cleanup_removes_clean_inactive_worktree(tmp_path):
    from hermes_cli.ultragoal_cleanup import post_merge_cleanup_plan

    allowed = tmp_path / ".worktrees"
    clean = allowed / "impl-clean"
    active = allowed / "impl-active"
    run_root = tmp_path / ".hermes" / "goal-runs" / "BO-203"
    for path in (clean, active, run_root):
        path.mkdir(parents=True)
    plan = post_merge_cleanup_plan(
        [
            {"path": str(clean), "kind": "implementation_worktree", "dirty": False, "activeCwd": False, "mergeClosed": True},
            {"path": str(active), "kind": "implementation_worktree", "dirty": False, "activeCwd": True, "mergeClosed": True},
            {"path": str(run_root), "kind": "run_root", "mergeClosed": True},
        ],
        allowed_roots=[allowed],
    )

    assert plan[str(clean)]["action"] == "remove_worktree"
    assert plan[str(active)]["action"] == "preserve"
    assert plan[str(run_root)]["action"] == "never_touch"


def test_post_merge_cleanup_requires_explicit_allowed_root(tmp_path):
    from hermes_cli.ultragoal_cleanup import post_merge_cleanup_plan

    unsafe = post_merge_cleanup_plan([{"path": "/tmp/not-owned", "kind": "implementation_worktree", "dirty": False, "activeCwd": False, "mergeClosed": True}])
    assert unsafe["/tmp/not-owned"]["action"] == "preserve"
    assert unsafe["/tmp/not-owned"]["reason"] == "outside_allowed_roots"

    owned = tmp_path / "repo" / ".worktrees" / "impl"
    owned.mkdir(parents=True)
    safe = post_merge_cleanup_plan(
        [{"path": str(owned), "kind": "implementation_worktree", "dirty": False, "activeCwd": False, "mergeClosed": True}],
        allowed_roots=[tmp_path / "repo" / ".worktrees"],
    )
    assert safe[str(owned)]["action"] == "remove_worktree"
