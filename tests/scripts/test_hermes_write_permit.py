import json
import os
from pathlib import Path
import subprocess
import sys


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/hermes_write_permit.py"


def run(repo: Path, home: Path, *args: str):
    env = {**os.environ, "HERMES_HOME": str(home)}
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args, "--cwd", str(repo)],
        text=True,
        capture_output=True,
        env=env,
    )


def make_repo(tmp_path: Path) -> tuple[Path, str, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    (repo / "a.txt").write_text("a\n")
    subprocess.run(["git", "add", "a.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo, check=True, capture_output=True)
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    return repo, "main", sha


def test_second_task_cannot_take_workspace(tmp_path):
    repo, branch, sha = make_repo(tmp_path)
    home = tmp_path / "home"
    first = run(repo, home, "acquire", "--task-id", "T1", "--branch", branch, "--base-sha", sha, "--allowed-path", "a.txt", "--approval", "owner-ok")
    second = run(repo, home, "acquire", "--task-id", "T2", "--branch", branch, "--base-sha", sha, "--allowed-path", "a.txt", "--approval", "owner-ok")

    assert first.returncode == 0
    assert second.returncode == 2
    assert json.loads(second.stdout)["reason"] == "workspace_locked"


def test_check_fails_when_scope_changes(tmp_path):
    repo, branch, sha = make_repo(tmp_path)
    home = tmp_path / "home"
    run(repo, home, "acquire", "--task-id", "T1", "--branch", branch, "--base-sha", sha, "--allowed-path", "a.txt", "--approval", "owner-ok")
    result = run(repo, home, "check", "--task-id", "T1", "--branch", branch, "--base-sha", sha, "--allowed-path", "other.txt")

    assert result.returncode == 2
    assert json.loads(result.stdout)["ok"] is False


def test_release_allows_next_task(tmp_path):
    repo, branch, sha = make_repo(tmp_path)
    home = tmp_path / "home"
    run(repo, home, "acquire", "--task-id", "T1", "--branch", branch, "--base-sha", sha, "--allowed-path", "a.txt", "--approval", "owner-ok")
    released = run(repo, home, "release", "--task-id", "T1")
    second = run(repo, home, "acquire", "--task-id", "T2", "--branch", branch, "--base-sha", sha, "--allowed-path", "a.txt", "--approval", "owner-ok")

    assert released.returncode == 0
    assert second.returncode == 0
