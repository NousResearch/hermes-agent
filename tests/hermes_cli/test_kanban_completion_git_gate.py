"""Tests for kanban completion git gate.

Verifies that the gate correctly identifies:
- Landed work (branch is ancestor of origin/main)
- Unpushed work (branch exists locally but not on origin)
- Work with open PR (in-flight)
- Work without PR (unlanded)
- Non-git workspace (should be silent)
"""
import json
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from hermes_cli.kanban_completion_git_gate import (
    evaluate_git_completion,
    apply_git_gate_to_metadata,
    _is_git_repo,
    _get_current_branch,
    _has_remote_branch,
    _has_unmerged_commits,
    _find_open_pr_for_branch,
)


@pytest.fixture
def temp_git_repo(tmp_path):
    """Create a temporary git repo with origin/main and a feature branch."""
    repo = tmp_path / "test_repo"
    repo.mkdir()
    
    # Initialize repo
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo, check=True, capture_output=True)
    
    # Create initial commit on main
    (repo / "README.md").write_text("# Test")
    subprocess.run(["git", "add", "README.md"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "branch", "-M", "main"], cwd=repo, check=True, capture_output=True)
    
    # Create a bare repo to act as origin
    origin = tmp_path / "origin.git"
    subprocess.run(["git", "init", "--bare", str(origin)], check=True, capture_output=True)
    subprocess.run(["git", "remote", "add", "origin", str(origin)], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "push", "-u", "origin", "main"], cwd=repo, check=True, capture_output=True)
    
    return repo


def test_is_git_repo(temp_git_repo):
    """Test _is_git_repo detects git repos correctly."""
    assert _is_git_repo(temp_git_repo) is True
    assert _is_git_repo(temp_git_repo.parent) is False


def test_get_current_branch(temp_git_repo):
    """Test _get_current_branch returns the current branch name."""
    branch = _get_current_branch(temp_git_repo)
    assert branch == "main"
    
    # Create and checkout a feature branch
    subprocess.run(["git", "checkout", "-b", "feature/test"], cwd=temp_git_repo, check=True, capture_output=True)
    branch = _get_current_branch(temp_git_repo)
    assert branch == "feature/test"


def test_has_remote_branch(temp_git_repo):
    """Test _has_remote_branch detects remote branches."""
    # main exists on origin
    assert _has_remote_branch(temp_git_repo, "main") is True
    
    # Create a local-only branch
    subprocess.run(["git", "checkout", "-b", "local-only"], cwd=temp_git_repo, check=True, capture_output=True)
    assert _has_remote_branch(temp_git_repo, "local-only") is False
    
    # Push it
    subprocess.run(["git", "push", "-u", "origin", "local-only"], cwd=temp_git_repo, check=True, capture_output=True)
    assert _has_remote_branch(temp_git_repo, "local-only") is True


def test_has_unmerged_commits(temp_git_repo):
    """Test _has_unmerged_commits detects commits not on origin/main."""
    # main is in sync with origin/main
    assert _has_unmerged_commits(temp_git_repo, "main") is False
    
    # Create a feature branch with commits
    subprocess.run(["git", "checkout", "-b", "feature/unmerged"], cwd=temp_git_repo, check=True, capture_output=True)
    (temp_git_repo / "new.txt").write_text("new file")
    subprocess.run(["git", "add", "new.txt"], cwd=temp_git_repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "Add new file"], cwd=temp_git_repo, check=True, capture_output=True)
    
    assert _has_unmerged_commits(temp_git_repo, "feature/unmerged") is True


def test_evaluate_git_completion_unpushed(temp_git_repo):
    """Test evaluate_git_completion detects unpushed work."""
    # Create an unpushed branch
    subprocess.run(["git", "checkout", "-b", "wt/t_unpushed"], cwd=temp_git_repo, check=True, capture_output=True)
    (temp_git_repo / "work.txt").write_text("work")
    subprocess.run(["git", "add", "work.txt"], cwd=temp_git_repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "Work"], cwd=temp_git_repo, check=True, capture_output=True)
    
    state = evaluate_git_completion(str(temp_git_repo), "wt/t_unpushed", "t_unpushed")
    
    assert state is not None
    assert state["branch"] == "wt/t_unpushed"
    assert state["pushed"] is False
    assert state["landed"] is False
    assert state["unmerged_commits"] is True
    assert state["pr"] is None
    assert state["unlanded"] is True


def test_evaluate_git_completion_with_pr(temp_git_repo):
    """Test evaluate_git_completion detects work with open PR."""
    # Create a pushed branch
    subprocess.run(["git", "checkout", "-b", "wt/t_with_pr"], cwd=temp_git_repo, check=True, capture_output=True)
    (temp_git_repo / "pr_work.txt").write_text("pr work")
    subprocess.run(["git", "add", "pr_work.txt"], cwd=temp_git_repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "PR work"], cwd=temp_git_repo, check=True, capture_output=True)
    subprocess.run(["git", "push", "-u", "origin", "wt/t_with_pr"], cwd=temp_git_repo, check=True, capture_output=True)
    
    # Mock gh CLI to return a PR
    mock_pr = {
        "number": 123,
        "url": "https://github.com/test/repo/pull/123",
        "title": "Fix t_with_pr",
        "body": "Closes t_with_pr"
    }
    
    with patch("hermes_cli.kanban_completion_git_gate._run") as mock_run:
        def side_effect(cmd, **kwargs):
            if cmd[0] == "gh" and "pr" in cmd:
                result = MagicMock()
                result.returncode = 0
                result.stdout = json.dumps([mock_pr])
                return result
            # For git commands, run them normally
            return subprocess.run(cmd, capture_output=True, text=True, check=False, **kwargs)
        
        mock_run.side_effect = side_effect
        
        state = evaluate_git_completion(str(temp_git_repo), "wt/t_with_pr", "t_with_pr")
        
        assert state is not None
        assert state["branch"] == "wt/t_with_pr"
        assert state["pushed"] is True
        assert state["unmerged_commits"] is True
        assert state["pr"] is not None
        assert state["pr"]["number"] == 123
        assert state["unlanded"] is False  # Has PR, so not unlanded


def test_evaluate_git_completion_landed(temp_git_repo):
    """Test evaluate_git_completion detects landed work."""
    # main is already landed (in sync with origin/main)
    state = evaluate_git_completion(str(temp_git_repo), "main", "t_landed")
    
    assert state is not None
    assert state["branch"] == "main"
    assert state["pushed"] is True
    assert state["landed"] is True
    assert state["unmerged_commits"] is False
    assert state["unlanded"] is False


def test_evaluate_git_completion_non_git(tmp_path):
    """Test evaluate_git_completion returns None for non-git workspace."""
    state = evaluate_git_completion(str(tmp_path), None, "t_non_git")
    assert state is None


def test_apply_git_gate_to_metadata_unlanded():
    """Test apply_git_gate_to_metadata adds unlanded_work for unlanded state."""
    git_state = {
        "branch": "wt/t_unpushed",
        "pushed": False,
        "landed": False,
        "unmerged_commits": True,
        "pr": None,
        "unlanded": True,
        "degraded": False,
    }
    
    metadata = apply_git_gate_to_metadata({}, git_state, "t_unpushed")
    
    assert metadata is not None
    assert "git_completion" in metadata
    assert metadata["git_completion"]["branch"] == "wt/t_unpushed"
    assert metadata["git_completion"]["pushed"] is False
    assert metadata["git_completion"]["landed"] is False
    
    assert "unlanded_work" in metadata
    assert metadata["unlanded_work"]["branch"] == "wt/t_unpushed"
    assert metadata["unlanded_work"]["pr_state"] == "none"
    assert "nobody" in metadata["unlanded_work"]["merge_owner"]


def test_apply_git_gate_to_metadata_landed():
    """Test apply_git_gate_to_metadata does not add unlanded_work for landed state."""
    git_state = {
        "branch": "main",
        "pushed": True,
        "landed": True,
        "unmerged_commits": False,
        "pr": None,
        "unlanded": False,
        "degraded": False,
    }
    
    metadata = apply_git_gate_to_metadata({}, git_state, "t_landed")
    
    assert metadata is not None
    assert "git_completion" in metadata
    assert metadata["git_completion"]["landed"] is True
    assert "unlanded_work" not in metadata


def test_apply_git_gate_to_metadata_none():
    """Test apply_git_gate_to_metadata returns metadata unchanged when git_state is None."""
    metadata = {"existing": "data"}
    result = apply_git_gate_to_metadata(metadata, None, "t_test")
    assert result == metadata


def test_find_open_pr_for_branch_no_gh(temp_git_repo):
    """Test _find_open_pr_for_branch returns None when gh is unavailable."""
    with patch("hermes_cli.kanban_completion_git_gate._run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="gh: command not found")
        
        pr = _find_open_pr_for_branch(temp_git_repo, "feature/test")
        assert pr is None


def test_find_open_pr_for_branch_with_pr(temp_git_repo):
    """Test _find_open_pr_for_branch returns PR info when gh returns a PR."""
    mock_pr = {
        "number": 456,
        "url": "https://github.com/test/repo/pull/456",
        "title": "Test PR",
        "body": "Description"
    }
    
    with patch("hermes_cli.kanban_completion_git_gate._run") as mock_run:
        def side_effect(cmd, **kwargs):
            if cmd[0] == "gh":
                result = MagicMock()
                result.returncode = 0
                result.stdout = json.dumps([mock_pr])
                return result
            return subprocess.run(cmd, capture_output=True, text=True, check=False, **kwargs)
        
        mock_run.side_effect = side_effect
        
        pr = _find_open_pr_for_branch(temp_git_repo, "feature/test")
        
        assert pr is not None
        assert pr["number"] == 456
        assert pr["url"] == "https://github.com/test/repo/pull/456"
