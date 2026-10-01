"""A dispatched card worktree is cut from the repo's trunk — never from the
primary checkout's incidental HEAD — and the worker's opening context states
the base commit it was cut from.

t_46fcdf7c: the board's primary checkout sat on
``experimental/visual-stage-20260930`` (~110 commits off ``main``), so every
card worktree the dispatcher created inherited that branch. Three refs
(``experimental/visual-stage-20260930``, ``wt/t_8f2c540b``, ``wt/t_a90eff35``)
pointed at one commit, ``git branch --no-merged main`` double-counted, and a
worker's first ``git log -1`` reported a revision 110 commits stale unless it
checked ``main`` explicitly.

The fix has two halves, both pinned here:
* ``_ensure_git_worktree`` bases a new branch on the repo's trunk (``main`` /
  ``master`` — the branch landings and serving happen on), falling back to
  ``HEAD`` only when the repo has no trunk or the board pins another base;
* the worker's opening context reports the workspace HEAD, the base commit
  (fork point with the trunk) and the trunk divergence, so a stale workspace
  is visible before any work happens.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_workspace as kbw


def _git(*args: str, cwd: Path | str | None = None) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=None if cwd is None else str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )
    assert result.returncode == 0, f"git {' '.join(args)} failed: {result.stderr}"
    return result.stdout


def _rev(repo: Path | str, ref: str = "HEAD") -> str:
    return _git("-C", str(repo), "rev-parse", ref).strip()


def _divergence(repo: Path | str, trunk: str, rev: str = "HEAD") -> tuple[int, int]:
    """``(behind, ahead)`` of *rev* relative to *trunk*."""
    out = _git("-C", str(repo), "rev-list", "--left-right", "--count", f"{trunk}...{rev}")
    behind, ahead = (int(part) for part in out.split())
    return behind, ahead


def _is_ancestor(repo: Path | str, ancestor: str, descendant: str = "HEAD") -> bool:
    result = subprocess.run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor", ancestor, descendant],
        capture_output=True,
        text=True,
        timeout=60,
    )
    return result.returncode == 0


def _subject(repo: Path | str, rev: str = "HEAD") -> str:
    return _git("-C", str(repo), "log", "-1", "--format=%s", rev).strip()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def diverged_repo(tmp_path: Path) -> Path:
    """A repo whose checkout stands on a side branch while ``main`` has moved on.

    ``main``: init -> trunk work
    ``experimental`` (checked out): init -> experimental work

    So ``main`` is *not* an ancestor of the checkout's HEAD — the exact shape of
    the GOD board's primary checkout vs ``main``.
    """
    repo = tmp_path / "project"
    repo.mkdir()
    _git("init", "-q", "-b", "main", str(repo))
    _git("-C", str(repo), "config", "user.email", "t@example.com")
    _git("-C", str(repo), "config", "user.name", "t")
    (repo / "README.md").write_text("hello\n", encoding="utf-8")
    _git("-C", str(repo), "add", "README.md")
    _git("-C", str(repo), "commit", "-qm", "init")
    _git("-C", str(repo), "checkout", "-qb", "experimental")
    (repo / "experimental.txt").write_text("side branch work\n", encoding="utf-8")
    _git("-C", str(repo), "add", "experimental.txt")
    _git("-C", str(repo), "commit", "-qm", "experimental work")
    _git("-C", str(repo), "checkout", "-q", "main")
    (repo / "trunk.txt").write_text("trunk work\n", encoding="utf-8")
    _git("-C", str(repo), "add", "trunk.txt")
    _git("-C", str(repo), "commit", "-qm", "trunk work")
    _git("-C", str(repo), "checkout", "-q", "experimental")
    return repo


# ---------------------------------------------------------------------------
# Base ref: a fresh card worktree stands on the trunk
# ---------------------------------------------------------------------------


def test_card_worktree_is_cut_from_trunk_not_from_primary_head(diverged_repo: Path) -> None:
    """The defect: the worktree inherited the primary checkout's branch."""
    repo = diverged_repo
    target = repo / ".worktrees" / "t_aaaa0001"

    kbw._ensure_git_worktree(repo, target, "wt/t_aaaa0001")

    trunk, head = _rev(repo, "main"), _rev(repo, "experimental")
    assert trunk != head, "fixture must diverge"
    assert _rev(target) == trunk
    assert _is_ancestor(repo, "main", _rev(target))
    assert _divergence(target, "main") == (0, 0)
    # the branch-only commit must not be carried into the card workspace
    assert _subject(target) == "trunk work"
    assert not (target / "experimental.txt").exists()


def test_worktree_is_cut_from_trunk_when_the_checkout_is_already_on_it(
    diverged_repo: Path,
) -> None:
    """Unchanged outcome for the healthy case — trunk and HEAD agree."""
    repo = diverged_repo
    _git("-C", str(repo), "checkout", "-q", "main")
    target = repo / ".worktrees" / "t_aaaa0002"

    kbw._ensure_git_worktree(repo, target, "wt/t_aaaa0002")

    assert _rev(target) == _rev(repo, "main")
    assert _divergence(target, "main") == (0, 0)


def test_worktree_is_cut_from_trunk_when_the_checkout_head_is_detached(
    diverged_repo: Path,
) -> None:
    """A detached primary checkout has no branch to inherit — use the trunk."""
    repo = diverged_repo
    _git("-C", str(repo), "checkout", "-q", "--detach", "HEAD")
    target = repo / ".worktrees" / "t_aaaa0003"

    kbw._ensure_git_worktree(repo, target, "wt/t_aaaa0003")

    assert _rev(target) == _rev(repo, "main")
    assert _divergence(target, "main") == (0, 0)


def test_repo_without_a_trunk_keeps_branching_from_head(tmp_path: Path) -> None:
    """Fail-safe: no ``main``/``master`` anywhere means the old behaviour."""
    repo = tmp_path / "solo"
    repo.mkdir()
    _git("init", "-q", "-b", "dev", str(repo))
    _git("-C", str(repo), "config", "user.email", "t@example.com")
    _git("-C", str(repo), "config", "user.name", "t")
    (repo / "README.md").write_text("hello\n", encoding="utf-8")
    _git("-C", str(repo), "add", "README.md")
    _git("-C", str(repo), "commit", "-qm", "init")
    target = repo / ".worktrees" / "t_aaaa0004"

    kbw._ensure_git_worktree(repo, target, "wt/t_aaaa0004")

    assert _rev(target) == _rev(repo, "dev")


def test_board_metadata_can_pin_the_base_ref(kanban_home: Path, diverged_repo: Path) -> None:
    """An escape hatch for boards that deliberately stand on another branch."""
    repo = diverged_repo
    kb.create_board("pinned", default_workdir=str(repo))
    meta_path = kb.board_metadata_path("pinned")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["worktree_base"] = "experimental"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    target = repo / ".worktrees" / "t_aaaa0005"

    kbw._ensure_git_worktree(repo, target, "wt/t_aaaa0005", board="pinned")

    assert _rev(target) == _rev(repo, "experimental")


# ---------------------------------------------------------------------------
# Opening context: the base commit and the trunk divergence are stated
# ---------------------------------------------------------------------------


def _worktree_card(kanban_home: Path, repo: Path, task_id: str) -> str:
    """Create a worktree workspace for a real task row and return the task id."""
    with kbc.connect() as conn:
        task_id = kb.create_task(
            conn, title="card", assignee="default",
            workspace_kind="worktree", workspace_path=str(repo),
        )
        task = kb.get_task(conn, task_id)
        assert task is not None
        workspace = kbw.resolve_workspace(task)
        kbw.set_workspace_path(conn, task_id, str(workspace))
    return task_id


def test_opening_context_states_the_base_commit(kanban_home: Path, diverged_repo: Path) -> None:
    repo = diverged_repo
    task_id = _worktree_card(kanban_home, repo, "t_bbbb0001")
    workspace = repo / ".worktrees" / task_id
    assert _rev(workspace) == _rev(repo, "main")

    with kbc.connect() as conn:
        context = kb.build_worker_context(conn, task_id)

    assert "## Workspace base" in context
    base_line = next(
        line for line in context.splitlines() if line.startswith("Base commit")
    )
    assert _rev(repo, "main")[:12] in base_line
    assert "main" in base_line
    assert "0 behind / 0 ahead" in context


def test_opening_context_flags_a_workspace_that_is_behind_the_trunk(
    kanban_home: Path, diverged_repo: Path
) -> None:
    """A legacy (pre-fix) workspace must not look fresh."""
    repo = diverged_repo
    with kbc.connect() as conn:
        task_id = kb.create_task(
            conn, title="card", assignee="default",
            workspace_kind="worktree", workspace_path=str(repo),
        )
        stale = repo / ".worktrees" / task_id
        # cut the way the dispatcher did before the fix: from the primary HEAD
        _git("-C", str(repo), "worktree", "add", "-b", f"wt/{task_id}", str(stale), "HEAD")
        kbw.set_workspace_path(conn, task_id, str(stale))
        assert _rev(stale) == _rev(repo, "experimental")

        context = kb.build_worker_context(conn, task_id)

    assert "## Workspace base" in context
    behind, ahead = _divergence(stale, "main")
    assert behind and ahead, "fixture must have a diverged stale workspace"
    assert f"{behind} behind / {ahead} ahead" in context
    assert "⚠" in context
