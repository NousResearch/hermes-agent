from __future__ import annotations

import copy

import pytest

from hermes_cli.workflow import allocate_worktrees
from hermes_cli.workflow.policy import DEFAULT_POLICY
from hermes_cli.workflow.store import connect, create_workflow, save_dag


def _policy(**worktree_overrides):
    data = copy.deepcopy(DEFAULT_POLICY)
    data["worktrees"].update(worktree_overrides)
    return data


def _normalized_dag(*, base_ref: str | None = None, workspace_kind: str = "worktree") -> dict:
    workspace = {"kind": workspace_kind}
    if base_ref is not None:
        workspace["base_ref"] = base_ref
    return {
        "schema_version": 1,
        "workflow_id": "wf_example",
        "scale": "medium",
        "nodes": [
            {
                "id": "backend-api",
                "title": "Implement backend API",
                "role": "engineer",
                "profile": "engineer",
                "status": "waiting",
                "parents": [],
                "workspace": workspace,
                "definition_of_done": ["Tests pass."],
                "scope": {"summary": "Build backend API."},
            },
            {
                "id": "integration",
                "title": "Integrate outputs",
                "role": "integrator",
                "profile": "integrator",
                "status": "waiting",
                "parents": ["backend-api"],
                "definition_of_done": ["Integration passes."],
                "scope": {"summary": "Integrate outputs."},
            },
        ],
        "edges": [{"source": "backend-api", "target": "integration", "kind": "depends_on"}],
    }


def test_public_workflow_package_exports_worktree_allocator():
    assert callable(allocate_worktrees)


def test_allocate_worktrees_adds_deterministic_metadata_for_engineer_worktree_nodes(tmp_path):
    dag = _normalized_dag()

    allocated = allocate_worktrees(dag, policy=_policy(), workspace_path=tmp_path / "repo")

    backend = allocated["nodes"][0]
    assert backend["workspace"] == {
        "kind": "worktree",
        "base_ref": "origin/main",
        "branch": "workflow/wf_example/backend-api",
        "worktree_path": str(tmp_path / "repo" / ".worktrees" / "wf_example-backend-api"),
    }
    assert allocated["nodes"][1].get("workspace") is None


def test_allocate_worktrees_honors_node_base_ref_override(tmp_path):
    allocated = allocate_worktrees(_normalized_dag(base_ref="main"), policy=_policy(), workspace_path=tmp_path / "repo")

    assert allocated["nodes"][0]["workspace"]["base_ref"] == "main"


def test_allocate_worktrees_does_not_mutate_original_dag(tmp_path):
    dag = _normalized_dag()

    allocate_worktrees(dag, policy=_policy(), workspace_path=tmp_path / "repo")

    assert "branch" not in dag["nodes"][0]["workspace"]
    assert "worktree_path" not in dag["nodes"][0]["workspace"]
    assert "base_ref" not in dag["nodes"][0]["workspace"]


def test_allocate_worktrees_skips_when_policy_disables_worktrees(tmp_path):
    allocated = allocate_worktrees(_normalized_dag(), policy=_policy(enabled=False), workspace_path=tmp_path / "repo")

    assert allocated["nodes"][0]["workspace"] == {"kind": "worktree"}


def test_allocate_worktrees_skips_non_worktree_nodes(tmp_path):
    allocated = allocate_worktrees(_normalized_dag(workspace_kind="shared"), policy=_policy(), workspace_path=tmp_path / "repo")

    assert allocated["nodes"][0]["workspace"] == {"kind": "shared"}


def test_allocated_worktree_metadata_persists_through_save_dag(tmp_path):
    allocated = allocate_worktrees(_normalized_dag(), policy=_policy(), workspace_path=tmp_path / "repo")

    with connect(tmp_path / "workflow.db") as conn:
        create_workflow(conn, workflow_id="wf_example", title="Worktrees")
        save_dag(conn, workflow_id="wf_example", normalized_dag=allocated, now=9.0)
        row = conn.execute(
            "SELECT branch, worktree_path, base_ref FROM workflow_nodes WHERE workflow_id = ? AND node_id = ?",
            ("wf_example", "backend-api"),
        ).fetchone()

    assert row["branch"] == "workflow/wf_example/backend-api"
    assert row["worktree_path"] == str(tmp_path / "repo" / ".worktrees" / "wf_example-backend-api")
    assert row["base_ref"] == "origin/main"


def test_allocate_worktrees_rejects_duplicate_branch_or_worktree_path(tmp_path):
    dag = _normalized_dag()
    dag["nodes"].append(
        {
            "id": "frontend-ui",
            "title": "Implement frontend UI",
            "role": "engineer",
            "profile": "engineer",
            "status": "waiting",
            "parents": [],
            "workspace": {"kind": "worktree", "branch": "workflow/wf_example/backend-api"},
            "definition_of_done": ["Tests pass."],
            "scope": {"summary": "Build frontend UI."},
        }
    )

    with pytest.raises(ValueError, match="duplicate worktree branch allocation"):
        allocate_worktrees(dag, policy=_policy(), workspace_path=tmp_path / "repo")

    duplicate_path = str(tmp_path / "repo" / ".worktrees" / "wf_example-backend-api")
    dag = _normalized_dag()
    dag["nodes"].append(
        {
            "id": "frontend-ui",
            "title": "Implement frontend UI",
            "role": "engineer",
            "profile": "engineer",
            "status": "waiting",
            "parents": [],
            "workspace": {"kind": "worktree", "worktree_path": duplicate_path},
            "definition_of_done": ["Tests pass."],
            "scope": {"summary": "Build frontend UI."},
        }
    )

    with pytest.raises(ValueError, match="duplicate worktree path allocation"):
        allocate_worktrees(dag, policy=_policy(), workspace_path=tmp_path / "repo")


def test_allocate_worktrees_rejects_unsafe_branch_prefix(tmp_path):
    with pytest.raises(ValueError, match="worktrees.branch_prefix must be a non-empty safe git branch prefix"):
        allocate_worktrees(_normalized_dag(), policy=_policy(branch_prefix="../workflow"), workspace_path=tmp_path / "repo")

    with pytest.raises(ValueError, match="worktrees.branch_prefix must be a non-empty safe git branch prefix"):
        allocate_worktrees(_normalized_dag(), policy=_policy(branch_prefix="/"), workspace_path=tmp_path / "repo")


def test_allocate_worktrees_rejects_policy_root_that_escapes_workspace(tmp_path):
    with pytest.raises(ValueError, match="worktrees.root must stay inside the workspace"):
        allocate_worktrees(_normalized_dag(), policy=_policy(root="../outside"), workspace_path=tmp_path / "repo")
