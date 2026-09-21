"""P13 isolation proof for moss-branch-cleanup.py and kanban-daily-digest wrapper."""
import os
import subprocess
import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_moss_branch_dry_run_prevents_deletion(tmp_path):
    """MOSS_BRANCH_DRY_RUN=1 must prevent any git branch deletion."""
    # Create a fake repo with a fix/ branch
    fake_repo = tmp_path / "fake-upstream"
    fake_repo.mkdir()
    subprocess.run(["git", "init", "-b", "main", str(fake_repo)], capture_output=True)
    subprocess.run(["git", "-C", str(fake_repo), "config", "user.email", "t@t.com"], capture_output=True)
    subprocess.run(["git", "-C", str(fake_repo), "config", "user.name", "test"], capture_output=True)
    (fake_repo / "README.md").write_text("test")
    subprocess.run(["git", "-C", str(fake_repo), "add", "."], capture_output=True)
    subprocess.run(["git", "-C", str(fake_repo), "commit", "-m", "init"], capture_output=True)
    subprocess.run(["git", "-C", str(fake_repo), "branch", "fix/test-branch"], capture_output=True)

    # Run with dry-run
    env = dict(os.environ)
    env["MOSS_BRANCH_DRY_RUN"] = "1"
    env["MOSS_BRANCH_REPO"] = str(fake_repo)
    env["HERMES_HOME"] = str(tmp_path / "hermes")

    script = REPO_ROOT / "scripts" / "moss-branch-cleanup.py"
    r = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True, text=True, env=env, cwd=str(REPO_ROOT), timeout=30
    )
    # Branch must still exist
    branches = subprocess.run(
        ["git", "-C", str(fake_repo), "branch", "--list", "fix/test-branch"],
        capture_output=True, text=True
    )
    assert "fix/test-branch" in branches.stdout, "dry-run deleted the branch!"


def test_kanban_daily_digest_wrapper_resolves_repo_relative(tmp_path):
    """The wrapper must prefer repo-relative archive script over hardcoded path."""
    wrapper = REPO_ROOT / "scripts" / "kanban_daily_digest_noagent.sh"
    assert wrapper.exists()
    content = wrapper.read_text()
    # Must not hardcode /home/kensei/.hermes/scripts/
    assert "/home/kensei/.hermes/scripts/kanban_daily_digest_noagent.py" not in content
    # Must reference the repo-relative archive path
    assert "archive/kanban_daily_digest_noagent.py" in content