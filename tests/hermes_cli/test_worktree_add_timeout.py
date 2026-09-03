"""Worktree creation gets a budget that survives disk contention.

``git worktree add`` is the one git call whose wall time scales with repo
size and with how many sibling agents are hitting the same disk. Every
other git call in these modules is a metadata operation that returns in
well under a second.

PR #90602 raised the budget in ``cli.py`` from 30s to 120s after measuring
113s wall for a 10k-file checkout under load. The three sibling call sites
that also materialize a worktree kept their original values: the Kanban
dispatcher at 60s, and the subagent and web-dashboard helpers at 30s. A
100k-file monorepo measures 34s warm-cache on an otherwise idle machine, so
those budgets kill creates that are progressing normally — the dispatcher
recorded three ``spawn_failed`` runs at 60-61s against one such repo.

These tests assert the observable contract: the subprocess that runs
``worktree add`` receives a budget of at least the measured-worst-case
120s, whichever module issues it.
"""

from __future__ import annotations

import subprocess

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import web_git
from tools import subagent_worktree

# The floor #90602 established from measurement. A budget at or above this
# covers the 113s loaded checkout it observed.
MIN_BUDGET_SECONDS = 120


class _Recorder:
    """Stand-in for ``subprocess.run`` that captures the timeout kwarg."""

    def __init__(self, returncode: int = 0) -> None:
        self.calls: list[tuple[list[str], float | None]] = []
        self._returncode = returncode

    def __call__(self, cmd, *args, **kwargs):
        self.calls.append((list(cmd), kwargs.get("timeout")))
        return subprocess.CompletedProcess(cmd, self._returncode, "", "")

    def timeout_for(self, needle: str) -> float:
        """Budget given to the first recorded call containing ``needle``."""
        for cmd, timeout in self.calls:
            if needle in cmd:
                assert timeout is not None, f"{needle!r} ran with no timeout at all"
                return timeout
        raise AssertionError(f"no recorded call contained {needle!r}: {self.calls}")


def test_kanban_dispatcher_worktree_add_budget(tmp_path, monkeypatch):
    """The dispatcher's worktree add survives a slow monorepo checkout."""
    rec = _Recorder()
    monkeypatch.setattr(kb.subprocess, "run", rec)
    monkeypatch.setattr(kb, "_git_common_dir", lambda _p: None)
    monkeypatch.setattr(kb, "_git_branch_exists", lambda _r, _b: False)

    kb._ensure_git_worktree(tmp_path / "repo", tmp_path / "wt", "wt/t_abc123")

    assert rec.timeout_for("worktree") >= MIN_BUDGET_SECONDS


def test_subagent_worktree_add_budget(tmp_path, monkeypatch):
    """The subagent helper's worktree add gets the same budget.

    Drives ``create_subagent_worktree`` so the budget under test is the one
    production picks, not one the test supplies.
    """
    rec = _Recorder()
    monkeypatch.setattr(subagent_worktree.subprocess, "run", rec)
    monkeypatch.setattr(subagent_worktree, "_ensure_gitignore_entry", lambda _r: None)
    monkeypatch.setattr(subagent_worktree, "resolve_repo_root", lambda _p: str(tmp_path))

    subagent_worktree.create_subagent_worktree(str(tmp_path), "sub-1")

    assert rec.timeout_for("worktree") >= MIN_BUDGET_SECONDS


def test_web_git_worktree_add_budget(tmp_path, monkeypatch):
    """The dashboard's worktree add gets the same budget.

    Drives ``worktree_add`` so the budget under test is production's own.
    """
    rec = _Recorder()
    monkeypatch.setattr(web_git.subprocess, "run", rec)
    monkeypatch.setattr(web_git, "_ensure_repo", lambda _c: None)
    monkeypatch.setattr(web_git, "_main_root", lambda _c: str(tmp_path))

    web_git.worktree_add(str(tmp_path), {"name": "scratch"})

    assert rec.timeout_for("worktree") >= MIN_BUDGET_SECONDS


@pytest.mark.parametrize(
    "module",
    [kb, subagent_worktree, web_git],
    ids=["kanban_db", "subagent_worktree", "web_git"],
)
def test_budget_is_a_named_constant(module):
    """The budget is a named constant, so all sites move together.

    Three of the four worktree-creating call sites drifted apart because
    each held its own literal; #90602 fixed ``cli.py`` alone and the others
    stayed at 30s and 60s.
    """
    assert getattr(module, "_WORKTREE_ADD_TIMEOUT", 0) >= MIN_BUDGET_SECONDS


def test_ordinary_git_calls_keep_the_short_budget():
    """Widening the worktree budget must not slow failure on metadata calls.

    A hung ``status``/``fetch`` should still fail fast; only the checkout
    that legitimately takes minutes gets the long budget.
    """
    assert web_git._GIT_TIMEOUT <= 60
    assert subagent_worktree._GIT_TIMEOUT <= 60


def test_web_git_metadata_call_is_not_widened(tmp_path, monkeypatch):
    """A sibling git call in the same module keeps the short budget.

    Guards the real risk of this change: raising the module-wide
    ``_GIT_TIMEOUT`` instead of giving ``worktree add`` its own budget would
    make every dashboard git call hang five times longer before failing.
    """
    rec = _Recorder()
    monkeypatch.setattr(web_git.subprocess, "run", rec)

    web_git._git(str(tmp_path), ["status", "--porcelain"])

    assert rec.timeout_for("status") <= 60
