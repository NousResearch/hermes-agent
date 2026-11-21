"""A reused task branch/worktree is brought up to the fetched remote default branch.

A NEW task branch forks from ``origin/main``; the reuse paths (existing branch,
existing linked worktree) used to hand back whatever the branch was parked on.
Policy under test: a clean reused tree behind the base is rebased; a dirty tree
refuses the dispatch (work is never discarded); no remote means no change.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_workspace as kbw


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        [
            "git", "-C", str(cwd),
            "-c", "user.name=Test User",
            "-c", "user.email=test@example.com",
            "-c", "commit.gpgsign=false",
            *args,
        ],
        check=True, capture_output=True, text=True,
    ).stdout


def _commit(repo: Path, name: str) -> None:
    (repo / name).write_text(name, encoding="utf-8")
    _git(repo, "add", name)
    _git(repo, "commit", "-m", name)


def _behind(tree: Path, ref: str = "origin/main") -> int:
    """Commits on the remote's TRUE tip that ``tree`` lacks (fetches first, so it cannot be fooled by a stale ref)."""
    _git(tree, "fetch", "origin", "--quiet")
    return int(_git(tree, "rev-list", "--count", f"HEAD..{ref}").strip())


@pytest.fixture
def stale_clone(tmp_path):
    """Clone that has never fetched: a second clone pushed one more commit to origin/main."""
    origin = tmp_path / "origin.git"
    subprocess.run(["git", "init", "--bare", "-b", "main", str(origin)], check=True, capture_output=True)
    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", str(origin), str(clone)], check=True, capture_output=True)
    _git(clone, "checkout", "-b", "main")
    _commit(clone, "base.txt")
    _git(clone, "push", "-u", "origin", "main")
    _git(clone, "branch", "wt/existing")
    _git(clone, "branch", "wt/other")
    pusher = tmp_path / "pusher"
    subprocess.run(["git", "clone", str(origin), str(pusher)], check=True, capture_output=True)
    _commit(pusher, "ahead.txt")
    _git(pusher, "push", "origin", "main")
    return clone


def _resolve(branch: str, workspace_path: Path | None = None) -> Path:
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn,
            title="t",
            workspace_kind="worktree",
            workspace_path=str(workspace_path) if workspace_path else None,
            branch_name=branch,
        )
        task = kb.get_task(conn, tid)
    return kbw.resolve_workspace(task)


@pytest.fixture
def board_on(stale_clone, kanban_home):
    kb.write_board_metadata(kb.get_current_board(), default_workdir=str(stale_clone))
    return stale_clone


def _dispatch_anchored(branch: str) -> Path:
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="t", workspace_kind="worktree", branch_name=branch)
        task = kb.get_task(conn, tid)
    return kbw.resolve_workspace(task)


def test_new_branch_forks_from_remote_default(board_on):
    tree = _dispatch_anchored("wt/brand-new")
    assert _behind(tree) == 0


def test_existing_branch_without_tree_is_rebased(board_on):
    tree = _dispatch_anchored("wt/existing")
    assert _git(tree, "branch", "--show-current").strip() == "wt/existing"
    assert _behind(tree) == 0


def test_existing_tree_on_task_branch_is_rebased(board_on):
    tree = board_on / ".worktrees" / "mine"
    _git(board_on, "worktree", "add", str(tree), "wt/existing")
    _resolve("wt/existing", tree)
    assert _behind(tree) == 0


def test_dirty_tree_refuses_and_is_untouched(board_on, kanban_home):
    tree = board_on / ".worktrees" / "mine"
    _git(board_on, "worktree", "add", str(tree), "wt/existing")
    (tree / "scratch.txt").write_text("wip", encoding="utf-8")
    status_before = _git(tree, "status", "--porcelain")
    sha_before = _git(tree, "rev-parse", "wt/existing")

    with pytest.raises(RuntimeError) as exc:
        _resolve("wt/existing", tree)

    assert str(tree) in str(exc.value)
    assert f"git -C {tree} rebase origin/main" in str(exc.value)
    assert _git(tree, "status", "--porcelain") == status_before
    assert _git(tree, "rev-parse", "wt/existing") == sha_before


def test_conflicting_rebase_is_aborted_and_refused(board_on, kanban_home):
    tree = board_on / ".worktrees" / "mine"
    _git(board_on, "worktree", "add", str(tree), "wt/existing")
    # Same path origin/main added, different content: the rebase must conflict.
    (tree / "ahead.txt").write_text("mine", encoding="utf-8")
    _git(tree, "add", "ahead.txt")
    _git(tree, "commit", "-m", "mine")
    sha_before = _git(tree, "rev-parse", "HEAD")

    with pytest.raises(RuntimeError, match="conflicts"):
        _resolve("wt/existing", tree)

    assert _git(tree, "rev-parse", "HEAD") == sha_before
    assert _git(tree, "status", "--porcelain") == ""


def test_no_remote_still_dispatches(tmp_path, kanban_home):
    repo = tmp_path / "solo"
    repo.mkdir()
    subprocess.run(["git", "init", "-b", "main", str(repo)], check=True, capture_output=True)
    _commit(repo, "base.txt")
    _git(repo, "branch", "wt/existing")
    tree = repo / ".worktrees" / "mine"
    _git(repo, "worktree", "add", str(tree), "wt/existing")
    (tree / "scratch.txt").write_text("wip", encoding="utf-8")

    assert _resolve("wt/existing", tree) == tree.resolve()


def test_other_tasks_tree_is_not_rebased_or_refused(board_on, kanban_home):
    other = board_on / ".worktrees" / "other"
    _git(board_on, "worktree", "add", str(other), "wt/other")
    (other / "scratch.txt").write_text("wip", encoding="utf-8")
    sha_before = _git(other, "rev-parse", "HEAD")

    resolved = _resolve("wt/mine", other)

    assert resolved != other.resolve()
    assert _git(other, "rev-parse", "HEAD") == sha_before
    assert _behind(resolved) == 0


def test_reused_branch_converges_to_true_remote_tip_without_prior_fetch(board_on, kanban_home):
    tip = _git(board_on.parent / "pusher", "rev-parse", "HEAD").strip()
    # The dispatched clone has never fetched, so its own origin/main lags the tip.
    stale = subprocess.run(
        ["git", "-C", str(board_on), "cat-file", "-e", tip], capture_output=True, check=False,
    )
    assert stale.returncode != 0

    tree = _dispatch_anchored("wt/existing")

    subprocess.run(["git", "-C", str(tree), "merge-base", "--is-ancestor", tip, "HEAD"], check=True)
    assert _behind(tree) == 0
