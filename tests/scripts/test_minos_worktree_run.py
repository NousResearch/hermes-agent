import importlib.util
import json
import os
import subprocess
from pathlib import Path
from unittest import mock

import pytest


SCRIPT_ROOT = Path(os.environ.get("HERMES_SCRIPT_ROOT", str(Path.home() / ".hermes" / "scripts")))
SCRIPT_PATH = SCRIPT_ROOT / "minos_worktree_run.py"


spec = importlib.util.spec_from_file_location("minos_worktree_run", SCRIPT_PATH)
minos_worktree_run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(minos_worktree_run)


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.fixture
def clean_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.com")
    (repo / "README.md").write_text("hello\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "initial")
    return repo


def _task_pack(repo: Path) -> Path:
    task_pack = repo / "task-pack.md"
    task_pack.write_text("# task pack\n", encoding="utf-8")
    return task_pack


def test_bootstrap_rejects_dirty_repo(clean_repo: Path):
    (clean_repo / "README.md").write_text("dirty\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="dirty"):
        minos_worktree_run.bootstrap_run(
            repo_path=clean_repo,
            task_pack_path=_task_pack(clean_repo),
            run_id="run-001",
        )


def test_bootstrap_creates_worktree_in_deterministic_location(clean_repo: Path):
    result = minos_worktree_run.bootstrap_run(
        repo_path=clean_repo,
        task_pack_path=_task_pack(clean_repo),
        run_id="run-001",
    )

    expected = clean_repo / ".hermes" / "worktrees" / "run-001"
    assert Path(result["worktree_path"]) == expected
    assert expected.exists()
    assert (expected / ".git").exists()


def test_bootstrap_creates_artifact_dir(clean_repo: Path):
    result = minos_worktree_run.bootstrap_run(
        repo_path=clean_repo,
        task_pack_path=_task_pack(clean_repo),
        run_id="run-001",
    )

    artifact_dir = clean_repo / ".hermes" / "runs" / "run-001"
    assert Path(result["artifact_dir"]) == artifact_dir
    assert artifact_dir.exists()


def test_bootstrap_records_branch_name(clean_repo: Path):
    result = minos_worktree_run.bootstrap_run(
        repo_path=clean_repo,
        task_pack_path=_task_pack(clean_repo),
        run_id="run-001",
    )

    metadata_path = clean_repo / ".hermes" / "runs" / "run-001" / "bootstrap.json"
    assert metadata_path.exists()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["branch_name"] == "minos/run-001"
    assert result["branch_name"] == "minos/run-001"


def test_bootstrap_rejects_invalid_run_id(clean_repo: Path):
    for run_id in ("../bad", "bad.lock"):
        with pytest.raises(ValueError, match="run_id"):
            minos_worktree_run.bootstrap_run(
                repo_path=clean_repo,
                task_pack_path=_task_pack(clean_repo),
                run_id=run_id,
            )


def test_bootstrap_rolls_back_worktree_when_metadata_write_fails(clean_repo: Path):
    task_pack = _task_pack(clean_repo)
    with mock.patch.object(minos_worktree_run, "write_bootstrap_metadata", side_effect=OSError("disk full")):
        with pytest.raises(OSError, match="disk full"):
            minos_worktree_run.bootstrap_run(
                repo_path=clean_repo,
                task_pack_path=task_pack,
                run_id="run-001",
            )

    worktrees = _git(clean_repo, "worktree", "list", "--porcelain").stdout
    assert str(clean_repo / ".hermes" / "worktrees" / "run-001") not in worktrees
    branches = _git(clean_repo, "branch", "--list", "minos/run-001").stdout.strip()
    assert branches == ""
    assert not (clean_repo / ".hermes" / "runs" / "run-001").exists()


def test_bootstrap_does_not_leave_artifact_dir_on_branch_conflict(clean_repo: Path):
    task_pack = _task_pack(clean_repo)
    _git(clean_repo, "branch", "minos/run-001")

    with pytest.raises(RuntimeError, match="Branch already exists"):
        minos_worktree_run.bootstrap_run(
            repo_path=clean_repo,
            task_pack_path=task_pack,
            run_id="run-001",
        )

    assert not (clean_repo / ".hermes" / "runs" / "run-001").exists()


def test_execute_builder_run_captures_stdout_stderr_and_exit_code(clean_repo: Path):
    bootstrap = minos_worktree_run.bootstrap_run(
        repo_path=clean_repo,
        task_pack_path=_task_pack(clean_repo),
        run_id="run-001",
    )

    result = minos_worktree_run.execute_builder_run(
        bootstrap,
        [
            "python3",
            "-c",
            "import sys; print('hello-out'); print('hello-err', file=sys.stderr); sys.exit(3)",
        ],
    )

    assert result["exit_code"] == 3
    log_path = Path(result["builder_log_path"])
    assert log_path.exists()
    log_text = log_path.read_text(encoding="utf-8")
    assert "hello-out" in log_text
    assert "hello-err" in log_text

    result_path = Path(result["builder_result_path"])
    persisted = json.loads(result_path.read_text(encoding="utf-8"))
    assert persisted["exit_code"] == 3



def test_execute_builder_run_writes_summary_file(clean_repo: Path):
    bootstrap = minos_worktree_run.bootstrap_run(
        repo_path=clean_repo,
        task_pack_path=_task_pack(clean_repo),
        run_id="run-001",
    )

    result = minos_worktree_run.execute_builder_run(
        bootstrap,
        ["python3", "-c", "print('summary test')"],
    )

    summary_path = Path(result["builder_summary_path"])
    assert summary_path.exists()
    summary = summary_path.read_text(encoding="utf-8")
    assert "run-001" in summary
    assert "exit code: 0" in summary.lower()
    assert "builder log: builder.log" in summary.lower()
    assert "git status: git-status.txt" in summary.lower()
    assert "stdout preview: summary test" in summary.lower()



def test_execute_builder_run_records_git_status_and_diff(clean_repo: Path):
    bootstrap = minos_worktree_run.bootstrap_run(
        repo_path=clean_repo,
        task_pack_path=_task_pack(clean_repo),
        run_id="run-001",
    )

    result = minos_worktree_run.execute_builder_run(
        bootstrap,
        [
            "python3",
            "-c",
            "from pathlib import Path; p = Path('README.md'); p.write_text(p.read_text() + 'more\\n', encoding='utf-8'); Path('NEW.txt').write_text('brand new\\n', encoding='utf-8'); Path('newdir').mkdir(); Path('newdir/nested.txt').write_text('nested file\\n', encoding='utf-8')",
        ],
    )

    status_text = Path(result["git_status_path"]).read_text(encoding="utf-8")
    diff_text = Path(result["git_diff_path"]).read_text(encoding="utf-8")
    assert "README.md" in status_text
    assert "NEW.txt" in status_text
    assert "newdir" in status_text
    assert "README.md" in diff_text
    assert "+more" in diff_text
    assert "NEW.txt" in diff_text
    assert "+brand new" in diff_text
    assert "newdir/nested.txt" in diff_text
    assert "+nested file" in diff_text
    assert "Users/atlas" not in diff_text
