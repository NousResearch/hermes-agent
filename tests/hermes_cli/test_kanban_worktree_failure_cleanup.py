"""A failed ``git worktree add`` must not leave a partial checkout behind.

Git writes the worktree's ``.git`` pointer early, before checking out files.
When ``worktree add`` died (60s subprocess timeout on large monorepos, or any
non-zero exit) the partial directory survived, and the next dispatch passed
``_ensure_git_worktree``'s common-dir reuse shortcut straight into the
incomplete tree — workers ran on a partial checkout and 5/6 such tasks even
reported ``completed`` (#126004).

Under test:
- both failure paths (non-zero exit, ``TimeoutExpired``) discard the partial
  directory + admin metadata before raising;
- a retry then performs a real, complete ``worktree add``;
- ``HERMES_KANBAN_WORKTREE_TIMEOUT`` overrides the add timeout (default 600s
  replaces the hard 60s kill);
- ``_record_task_failure`` keeps 4000 chars of the failure text so the real
  git fatal line survives into ``tasks.last_failure_error``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_db_workspace as kbw


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(
        [
            "git", "-C", str(cwd),
            "-c", "user.name=Test User",
            "-c", "user.email=test@example.com",
            "-c", "commit.gpgsign=false",
            *args,
        ],
        check=True, capture_output=True, text=True,
    )


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(
        ["git", "init", "-b", "main", str(repo)],
        check=True, capture_output=True, text=True,
    )
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "init")
    return repo


def _plant_partial_worktree(repo: Path, target: Path) -> None:
    """Reproduce git's failure footprint: pointer file written, checkout not."""
    target.mkdir(parents=True)
    gitdir = repo / ".git" / "worktrees" / target.name
    gitdir.mkdir(parents=True)
    (gitdir / "gitdir").write_text(str(target / ".git"), encoding="utf-8")
    (target / ".git").write_text(f"gitdir: {gitdir}\n", encoding="utf-8")
    (target / "README.md").write_text("half a checkout\n", encoding="utf-8")


def test_failed_add_discards_partial_and_retry_succeeds(kanban_home, tmp_path, monkeypatch):
    repo = _make_repo(tmp_path)
    target = repo / ".worktrees" / "task-1"
    real_git = kbw._git
    state = {"failed": False}
    add_timeouts: list[int] = []

    def fake_git(repo_root, *args, timeout):
        if args[:2] == ("worktree", "add"):
            add_timeouts.append(timeout)
            if not state["failed"]:
                state["failed"] = True
                # Reproduce git's failure footprint before dying: pointer
                # written, checkout not completed. The add's target sits at a
                # different index with/without ``-b``, so find the absolute path.
                target_arg = next(a for a in args[2:] if a.startswith("/"))
                _plant_partial_worktree(repo_root, Path(target_arg))
                return subprocess.CompletedProcess(
                    ["git", *args], returncode=128,
                    stdout="", stderr="fatal: unable to checkout 'HEAD'\n",
                )
        return real_git(repo_root, *args, timeout=timeout)

    monkeypatch.setattr(kbw, "_git", fake_git)

    with pytest.raises(RuntimeError, match="unable to checkout"):
        kbw._ensure_git_worktree(repo, target, "wt/task-1")

    # The partial checkout (pointer + half a file) is gone, so the reuse
    # shortcut can no longer fast-path a retry into an incomplete tree.
    assert not target.exists()
    listed = subprocess.run(
        ["git", "-C", str(repo), "worktree", "list", "--porcelain"],
        capture_output=True, text=True, check=True,
    ).stdout
    assert str(target) not in listed

    # Retry: real git add runs and completes a full checkout.
    kbw._ensure_git_worktree(repo, target, "wt/task-1")
    assert kbw._is_linked_worktree_checkout(target)
    assert (target / "README.md").read_text(encoding="utf-8") == "base\n"


def test_timed_out_add_discards_partial(kanban_home, tmp_path, monkeypatch):
    repo = _make_repo(tmp_path)
    target = repo / ".worktrees" / "task-2"
    _plant_partial_worktree(repo, target)
    real_git = kbw._git
    state = {"timed_out": False}

    def fake_git(repo_root, *args, timeout):
        if args[:2] == ("worktree", "add") and not state["timed_out"]:
            state["timed_out"] = True
            raise subprocess.TimeoutExpired(cmd="git worktree add", timeout=600)
        return real_git(repo_root, *args, timeout=timeout)

    monkeypatch.setattr(kbw, "_git", fake_git)

    with pytest.raises(subprocess.TimeoutExpired):
        kbw._ensure_git_worktree(repo, target, "wt/task-2")
    assert not target.exists()

    kbw._ensure_git_worktree(repo, target, "wt/task-2")
    assert (target / "README.md").read_text(encoding="utf-8") == "base\n"


def test_add_timeout_env_override(kanban_home, tmp_path, monkeypatch):
    repo = _make_repo(tmp_path)
    target = repo / ".worktrees" / "task-3"
    real_git = kbw._git
    add_timeouts: list[int] = []

    def fake_git(repo_root, *args, timeout):
        if args[:2] == ("worktree", "add"):
            add_timeouts.append(timeout)
        return real_git(repo_root, *args, timeout=timeout)

    monkeypatch.setattr(kbw, "_git", fake_git)
    monkeypatch.setenv("HERMES_KANBAN_WORKTREE_TIMEOUT", "123")

    kbw._ensure_git_worktree(repo, target, "wt/task-3")
    assert add_timeouts == [123]


def test_record_task_failure_keeps_4000_chars(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="spawn-fail diagnostics")
        conn.execute(
            "UPDATE tasks SET status='running' WHERE id = ?", (tid,),
        )
        conn.commit()

        kbd._record_task_failure(
            conn, tid, "x" * 5000,
            outcome="spawn_failed", release_claim=True, end_run=True,
        )
        row = conn.execute(
            "SELECT last_failure_error FROM tasks WHERE id = ?", (tid,),
        ).fetchone()
        assert len(row["last_failure_error"]) == 4000
