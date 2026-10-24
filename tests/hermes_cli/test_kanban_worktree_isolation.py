"""Per-task worktree isolation for decompose siblings.

Decompose children used to inherit the root's literal ``workspace_path``,
so every sibling of a worktree-kind root pointed at the SAME checkout —
and ``_resolve_worktree_workspace``'s existing-checkout shortcut reused it
on whatever branch was there, letting sibling workers run concurrently in
one directory on one branch (cross-task provenance corruption, no lock).

Two-part fix under test:
- ``decompose_triage_task`` leaves worktree children's ``workspace_path``
  unset so each child materializes its own ``<repo>/.worktrees/<child-id>``.
- ``_resolve_worktree_workspace`` falls back to a fresh per-task worktree
  when the requested path is occupied by another task's branch (heals
  pre-existing rows that still carry a shared path).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_db_graph import decompose_triage_task
from hermes_cli import kanban_db_connect as kbc
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


def _add_worktree(repo: Path, target: Path, branch: str) -> Path:
    _git(repo, "worktree", "add", str(target), "-b", branch, "HEAD")
    return target


def test_decompose_worktree_children_get_own_workspace(kanban_home):
    with kbc.connect() as conn:
        root = kb.create_task(conn, title="build the feature", triage=True)
        conn.execute(
            "UPDATE tasks SET workspace_kind='worktree', "
            "workspace_path='/repo/.worktrees/root' WHERE id = ?",
            (root,),
        )
        conn.commit()

        child_ids = decompose_triage_task(
            conn,
            root,
            root_assignee="orchestrator",
            children=[
                {"title": "spec it", "assignee": "alice", "parents": []},
                {"title": "implement it", "assignee": "bob", "parents": [0]},
            ],
            author="decomposer",
        )
        assert child_ids is not None and len(child_ids) == 2

        for cid in child_ids:
            row = conn.execute(
                "SELECT workspace_kind, workspace_path FROM tasks WHERE id = ?",
                (cid,),
            ).fetchone()
            assert row["workspace_kind"] == "worktree"
            # Each child resolves its own <repo>/.worktrees/<child-id> at
            # dispatch; the root's literal path must never be shared.
            assert row["workspace_path"] is None




def test_resolve_worktree_falls_back_when_path_occupied(kanban_home, tmp_path):
    repo = _make_repo(tmp_path)
    occupied = _add_worktree(repo, repo / ".worktrees" / "sibling", "wt/sibling")

    with kbc.connect() as conn:
        tid = kb.create_task(
            conn,
            title="second sibling",
            workspace_kind="worktree",
            workspace_path=str(occupied),  # inherited shared/stale path
        )
        task = kb.get_task(conn, tid)

    workspace, branch = kbw._resolve_worktree_workspace(task)
    assert workspace == (repo / ".worktrees" / tid).resolve()
    assert branch == f"wt/{tid}"
    # The sibling's checkout is untouched, still on its own branch.
    assert (occupied / "README.md").exists()
    head = subprocess.run(
        ["git", "-C", str(occupied), "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert head == "wt/sibling"


@pytest.mark.parametrize('placement', ['external', 'canonical', 'root_collision', 'fallback_collision'])
def test_occupied_worktree_without_safe_fallback_is_refused(kanban_home, tmp_path, placement):
    repo = _make_repo(tmp_path)
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title='must not reuse another task', workspace_kind='worktree',
                             workspace_path=str(repo))
        target = (tmp_path / 'outside' / 'occupied' if placement == 'external'
                  else repo / '.worktrees' / tid)
        target.parent.mkdir(parents=True, exist_ok=True)
        _add_worktree(repo, target, 'wt/other-task')
        (target / 'README.md').write_text('other task active edits\n')
        (target / 'untracked').write_text('other task untracked\n')
        requested = target
        if placement == 'root_collision':
            requested = repo
        elif placement == 'fallback_collision':
            requested = _add_worktree(repo, repo / '.worktrees' / 'sibling', 'wt/sibling')
        kbw.set_workspace_path(conn, tid, str(requested))
        task = kb.get_task(conn, tid)
    with pytest.raises(ValueError, match='occupied'):
        kbw._resolve_worktree_workspace(task)
    assert (target / 'README.md').read_text() == 'other task active edits\n'
    assert (target / 'untracked').read_text() == 'other task untracked\n'
    branch = subprocess.run(['git', '-C', str(target), 'branch', '--show-current'],
                            check=True, capture_output=True, text=True).stdout.strip()
    assert branch == 'wt/other-task'
    assert (repo / 'README.md').read_text() == 'base\n'


@pytest.mark.parametrize('changed_branch', ['detached', 'switched'])
def test_owned_task_branch_recovery_preserves_work_and_requires_unblock(
        kanban_home, tmp_path, all_assignees_spawnable, changed_branch):
    from hermes_cli import kanban_db_dispatch as dispatch
    repo = _make_repo(tmp_path)
    spawned = []
    def spawn(task, workspace, board=None):
        spawned.append((task.id, workspace))
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title='resume owned task', assignee='fixture',
                             workspace_kind='worktree', workspace_path=str(repo))
        task = kb.get_task(conn, tid)
        tree, branch = kbw._resolve_worktree_workspace(task)
        kbw.set_workspace_path(conn, tid, str(tree))
        kbw.set_branch_name(conn, tid, branch)
        _git(tree, 'switch', '--detach' if changed_branch == 'detached' else '-c',
             'HEAD' if changed_branch == 'detached' else 'inspection-branch')
        (tree / 'README.md').write_text('owned staged work\n')
        _git(tree, 'add', 'README.md')
        (tree / 'untracked').write_text('owned untracked work\n')
        for expected_failures in (1, 2):
            result = dispatch.dispatch_once(conn, spawn_fn=spawn, failure_limit=2)
            task = kb.get_task(conn, tid)
            assert task.consecutive_failures == expected_failures
            assert task.status == ('ready' if expected_failures == 1 else 'blocked')
            assert not result.spawned and not spawned
        outcomes = conn.execute('SELECT outcome FROM task_runs WHERE task_id=? ORDER BY id', (tid,)).fetchall()
        assert [row['outcome'] for row in outcomes] == ['spawn_failed', 'gave_up']
        diagnostic = task.last_failure_error
        assert branch in diagnostic
        assert ('detached HEAD' if changed_branch == 'detached' else 'inspection-branch') in diagnostic
        assert (tree / 'README.md').read_text() == 'owned staged work\n'
        assert (tree / 'untracked').read_text() == 'owned untracked work\n'
        # An operator has verified ownership and no live worker; no reset/clean/force.
        _git(tree, 'switch', branch)
        still_blocked = dispatch.dispatch_once(conn, spawn_fn=spawn, failure_limit=2)
        assert not still_blocked.spawned and not spawned
        assert kb.get_task(conn, tid).status == 'blocked'
        assert kb.unblock_task(conn, tid)
        assert kb.get_task(conn, tid).consecutive_failures == 0
        result = dispatch.dispatch_once(conn, spawn_fn=spawn, failure_limit=2)
        assert len(result.spawned) == 1
        assert spawned == [(tid, str(tree))]
        assert (tree / 'README.md').read_text() == 'owned staged work\n'
        assert (tree / 'untracked').read_text() == 'owned untracked work\n'
        status = subprocess.run(['git', '-C', str(tree), 'status', '--porcelain'],
                                check=True, capture_output=True, text=True).stdout
        assert 'M  README.md' in status and '?? untracked' in status
        assert kb.get_task(conn, tid).status == 'running'


@pytest.mark.parametrize('occupant', ['plain_directory', 'detached', 'other_branch'])
def test_occupied_target_error_names_the_state_and_the_recovery(tmp_path, occupant):
    repo = _make_repo(tmp_path)
    target = repo / '.worktrees' / 't_occupied'
    if occupant == 'plain_directory':
        target.mkdir(parents=True)
        (target / 'leftover').write_text('keep me\n')
    else:
        target.parent.mkdir(parents=True)
        _add_worktree(repo, target, 'wt/someone-else')
        if occupant == 'detached':
            _git(target, 'switch', '--detach', 'HEAD')
    with pytest.raises(ValueError) as excinfo:
        kbw._ensure_git_worktree(repo, target, 'wt/t_occupied')
    message = str(excinfo.value)
    expected = {
        'plain_directory': ('not a linked worktree', 'Move or remove that directory'),
        'detached': ('detached HEAD', "switch it back to 'wt/t_occupied'", 'If it belongs to another task'),
        'other_branch': ("branch 'wt/someone-else'", "switch it back to 'wt/t_occupied'",
                         'If it belongs to another task'),
    }[occupant]
    for fragment in expected:
        assert fragment in message, message
    if occupant == 'plain_directory':
        assert (target / 'leftover').read_text() == 'keep me\n'
