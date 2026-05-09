import json

import pytest

from hermes_cli.kanban_policy import (
    BoardPolicy,
    default_policy,
    format_policy_guidance,
    load_policy,
    policy_path_for_board,
    policy_report,
    validate_workspace,
)


def test_default_policy_is_safe_and_non_project_specific():
    policy = default_policy("unknown")

    assert policy.board == "unknown"
    assert policy.project_root is None
    assert policy.worktree_root is None
    assert policy.shared_project_root_writable is False
    assert policy.scratch_repo_operations_allowed is False
    assert policy.max_active_issue_pipelines is None


def test_policy_normalizes_paths_without_requiring_existence(tmp_path):
    project = tmp_path / "Project"
    worktrees = project / ".worktrees"

    policy = BoardPolicy(
        board="demo",
        project_root=project,
        base_branch="main",
        worktree_root=worktrees,
        denied_workspace_roots=[tmp_path / ".hermes"],
    )

    assert policy.project_root == project.resolve(strict=False)
    assert policy.worktree_root == worktrees.resolve(strict=False)


def test_load_policy_from_json_file(tmp_path, monkeypatch):
    policies = tmp_path / "kanban" / "policies"
    policies.mkdir(parents=True)
    (policies / "demo.json").write_text(
        json.dumps(
            {
                "project_root": str(tmp_path / "Demo"),
                "base_branch": "development",
                "worktree_root": str(tmp_path / "Demo" / ".worktrees"),
                "denied_workspace_roots": [str(tmp_path / "kanban")],
                "max_active_issue_pipelines": 1,
            }
        )
    )
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))

    policy = load_policy("demo")

    assert policy.board == "demo"
    assert policy.base_branch == "development"
    assert policy.max_active_issue_pipelines == 1
    assert policy.project_root == (tmp_path / "Demo").resolve(strict=False)


def test_policy_path_rejects_path_traversal_board_slug(tmp_path):
    with pytest.raises(ValueError, match="invalid board slug"):
        policy_path_for_board("../evil", home=tmp_path)


def test_missing_policy_returns_default(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))

    policy = load_policy("missing")

    assert policy.board == "missing"
    assert policy.project_root is None


def test_invalid_policy_reports_error(tmp_path, monkeypatch):
    policies = tmp_path / "kanban" / "policies"
    policies.mkdir(parents=True)
    (policies / "demo.json").write_text(json.dumps({"max_active_issue_pipelines": 0}))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))

    report = policy_report("demo")

    assert report["configured"] is True
    assert report["errors"]


def test_worktree_under_policy_root_passes(tmp_path):
    project = tmp_path / "Project"
    policy = BoardPolicy(board="demo", project_root=project, worktree_root=project / ".worktrees")

    findings = validate_workspace(
        policy,
        workspace_kind="worktree",
        workspace_path=str(project / ".worktrees" / "task-1"),
        repo_touching=True,
    )

    assert findings == []


def test_worktree_outside_policy_root_fails(tmp_path):
    project = tmp_path / "Project"
    policy = BoardPolicy(board="demo", project_root=project, worktree_root=project / ".worktrees")

    findings = validate_workspace(
        policy,
        workspace_kind="worktree",
        workspace_path=str(tmp_path / "scratch" / "task-1"),
        repo_touching=True,
    )

    assert any(f.code == "worktree_outside_policy_root" for f in findings)


def test_shared_project_root_fails_for_repo_touching_card(tmp_path):
    project = tmp_path / "Project"
    policy = BoardPolicy(board="demo", project_root=project, worktree_root=project / ".worktrees")

    findings = validate_workspace(
        policy,
        workspace_kind="worktree",
        workspace_path=str(project),
        repo_touching=True,
    )

    assert any(f.code == "shared_project_root_workspace" for f in findings)


def test_denied_workspace_root_fails(tmp_path):
    project = tmp_path / "Project"
    denied = tmp_path / ".hermes" / "kanban"
    policy = BoardPolicy(
        board="demo",
        project_root=project,
        worktree_root=project / ".worktrees",
        denied_workspace_roots=[denied],
    )

    findings = validate_workspace(
        policy,
        workspace_kind="worktree",
        workspace_path=str(denied / "boards" / "demo" / "workspaces" / "t1"),
        repo_touching=True,
    )

    assert any(f.code == "workspace_under_denied_root" for f in findings)


def test_policy_guidance_includes_safe_roots(tmp_path):
    project = tmp_path / "Project"
    policy = BoardPolicy(
        board="demo",
        project_root=project,
        worktree_root=project / ".worktrees",
        denied_workspace_roots=[tmp_path / ".hermes"],
        max_active_issue_pipelines=1,
    )

    guidance = format_policy_guidance(policy)

    assert "Board policy" in guidance
    assert str(project.resolve(strict=False)) in guidance
    assert "read-only" in guidance
    assert "Max active issue pipelines: 1" in guidance
