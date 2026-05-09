from types import SimpleNamespace

from hermes_cli.kanban_policy import BoardPolicy
from hermes_cli.kanban_validation import validate_tasks


def _task(**overrides):
    data = {
        "id": "t_policy",
        "title": "Implement GitHub issue #123",
        "body": "validation_assertions_to_satisfy: VC-001",
        "assignee": "fullstack-eng",
        "status": "ready",
        "workspace_kind": "worktree",
        "workspace_path": "/outside/project/task",
    }
    data.update(overrides)
    return SimpleNamespace(**data)


def _policy(tmp_path):
    project = tmp_path / "Project"
    return BoardPolicy(
        board="demo",
        project_root=project,
        worktree_root=project / ".worktrees",
        denied_workspace_roots=[tmp_path / ".hermes" / "kanban"],
    )


def test_policy_worktree_outside_project_worktrees_is_error(tmp_path):
    findings = validate_tasks([_task()], policy=_policy(tmp_path))
    codes = {f.code for f in findings}
    assert "worktree_outside_policy_root" in codes


def test_policy_shared_project_root_worktree_is_error(tmp_path):
    policy = _policy(tmp_path)
    findings = validate_tasks([_task(workspace_path=str(policy.project_root))], policy=policy)
    codes = {f.code for f in findings}
    assert "shared_project_root_workspace" in codes


def test_policy_project_local_worktree_is_allowed_for_ready_task(tmp_path):
    policy = _policy(tmp_path)
    findings = validate_tasks([
        _task(workspace_path=str(policy.worktree_root / "issue-123"))
    ], policy=policy)
    codes = {f.code for f in findings}
    assert "worktree_outside_policy_root" not in codes
    assert "shared_project_root_workspace" not in codes
    assert "repo_completion_missing_commit" not in codes
    assert "repo_completion_not_clean" not in codes
    assert "missing_commands_run_exit_codes" not in codes


def test_policy_denied_control_plane_root_is_error(tmp_path):
    policy = _policy(tmp_path)
    findings = validate_tasks([
        _task(workspace_path=str(tmp_path / ".hermes" / "kanban" / "boards" / "demo" / "workspaces" / "t1"))
    ], policy=policy)
    codes = {f.code for f in findings}
    assert "workspace_under_denied_root" in codes
