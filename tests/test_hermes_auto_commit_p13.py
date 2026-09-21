"""P13 isolation proof for scripts/gitops/hermes-auto-commit.sh.

Verifies the env override:
- GITOPS_DRY_RUN=1 prints the dry-run line and exits 0 before any git add/commit.
- The repo state is left unchanged (no new commits).
"""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "gitops" / "hermes-auto-commit.sh"


def _run(env_extra=None):
    env = dict(os.environ)
    env.pop("GITOPS_DRY_RUN", None)
    env.pop("HERMES_HOME", None)
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        ["bash", str(SCRIPT)], capture_output=True, text=True, env=env,
        cwd=str(REPO_ROOT), timeout=20,
    )


def _init_sandbox_repo(home: Path):
    """Build a tiny ~/.hermes-shaped sandbox with a git repo and a tracked change."""
    for sub in ["scripts", "skills", "profiles", "cron", "governance", "runbooks",
               "feature-artifacts"]:
        (home / sub).mkdir(parents=True)
    (home / "config.yaml").write_text("version: 1\n")
    (home / ".gitignore").write_text("# empty\n")
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=home, check=True)
    subprocess.run(["git", "config", "user.email", "kensei@local"], cwd=home, check=True)
    subprocess.run(["git", "config", "user.name", "KENSEI"], cwd=home, check=True)
    subprocess.run(["git", "add", ".gitignore", "config.yaml"], cwd=home, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "baseline"], cwd=home, check=True)
    return home


def test_dry_run_exits_zero_and_prints_plan(tmp_path):
    """GITOPS_DRY_RUN=1 prints the plan and exits 0."""
    home = tmp_path / "hermes"
    _init_sandbox_repo(home)
    r = _run({"GITOPS_DRY_RUN": "1", "HERMES_HOME": str(home)})
    assert r.returncode == 0, r.stderr
    assert "dry-run: would commit config changes" in r.stdout


def test_dry_run_leaves_repo_unchanged(tmp_path):
    """GITOPS_DRY_RUN=1 must not create any new commit, even when a tracked
    path has a pending change."""
    home = tmp_path / "hermes"
    _init_sandbox_repo(home)
    # Introduce a change to a tracked path.
    (home / "config.yaml").write_text("version: 2\n")
    before = subprocess.run(
        ["git", "rev-list", "--count", "HEAD"], cwd=home,
        capture_output=True, text=True,
    ).stdout.strip()

    r = _run({"GITOPS_DRY_RUN": "1", "HERMES_HOME": str(home)})
    assert r.returncode == 0, r.stderr

    after = subprocess.run(
        ["git", "rev-list", "--count", "HEAD"], cwd=home,
        capture_output=True, text=True,
    ).stdout.strip()
    assert before == after, f"dry-run created a commit: {before} -> {after}"
    # The working tree change must still be unstaged.
    diff = subprocess.run(
        ["git", "diff", "--name-only"], cwd=home,
        capture_output=True, text=True,
    ).stdout.strip()
    assert "config.yaml" in diff, "dry-run staged the change"


def test_no_dry_run_still_commits_change(tmp_path):
    """Without GITOPS_DRY_RUN, live mode DOES commit the tracked change —
    proves the dry-run guard is what suppresses the commit (control case)."""
    home = tmp_path / "hermes"
    _init_sandbox_repo(home)
    (home / "config.yaml").write_text("version: 2\n")
    r = _run({"HERMES_HOME": str(home)})
    assert r.returncode == 0, r.stderr
    after = subprocess.run(
        ["git", "rev-list", "--count", "HEAD"], cwd=home,
        capture_output=True, text=True,
    ).stdout.strip()
    assert int(after) >= 2, "live mode should have created a commit"
