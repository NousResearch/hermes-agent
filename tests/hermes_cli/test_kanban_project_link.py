"""Kanban <-> Projects integration: project-linked tasks get a deterministic
worktree path + branch instead of the random ``wt/<task-id>`` fallback — but
only when the project's ``primary_path`` can actually host one.

Regression for the creation-time gap: a card that inherited a project whose
``primary_path`` was not inside a git repo was silently upgraded to a
``worktree`` anyway, so it was created, dispatched, and then died at spawn
(``workspace: ... is not inside a git repo``) — twice, then ``gave_up``. The
upgrade is a default, so it must never break card creation.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import projects_db as pdb


@pytest.fixture
def kanban_conn(tmp_path):
    c = kbc.connect(db_path=tmp_path / "kanban.db")
    try:
        yield c
    finally:
        c.close()


def _git_init(repo):
    """A real git work tree — ``git init`` alone is what a worktree anchor needs.

    No commit: the predicate is "is this path inside a work tree", which an
    unborn branch satisfies, and no worktree is materialized at creation (only
    the path is recorded; dispatch creates it).
    """
    repo.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    return repo


def _make_project(tmp_path, name="Web App"):
    repo = _git_init(tmp_path / "webapp")
    with pdb.connect_closing() as pc:
        pid = pdb.create_project(pc, name=name, folders=[str(repo)])
        return pdb.get_project(pc, pid)


def _make_non_repo_project(tmp_path, name="Salvage"):
    """A project whose folder exists but is not inside a git work tree."""
    repo = tmp_path / "salvage-copy"
    repo.mkdir()
    with pdb.connect_closing() as pc:
        pid = pdb.create_project(pc, name=name, folders=[str(repo)])
        return pdb.get_project(pc, pid)


def test_project_linked_task_gets_deterministic_worktree_and_branch(kanban_conn, tmp_path):
    proj = _make_project(tmp_path)
    tid = kb.create_task(kanban_conn, title="Add login", project_id=proj.slug)
    task = kb.get_task(kanban_conn, tid)

    assert task.project_id == proj.id
    assert task.workspace_kind == "worktree"
    # Worktree dir anchored under the project's primary repo, keyed on task id.
    assert task.workspace_path == os.path.join(proj.primary_path, ".worktrees", tid)
    # Deterministic branch: <slug>/<task-id>-<title-slug>. NOT a random wt/...
    assert task.branch_name == f"{proj.slug}/{tid}-add-login"
    assert not task.branch_name.startswith("wt/")


def test_explicit_branch_overrides_project_default(kanban_conn, tmp_path):
    proj = _make_project(tmp_path)
    tid = kb.create_task(
        kanban_conn,
        title="x",
        project_id=proj.slug,
        workspace_kind="worktree",
        branch_name="feature/custom",
    )
    task = kb.get_task(kanban_conn, tid)
    assert task.branch_name == "feature/custom"


def test_unlinked_task_unchanged(kanban_conn):
    tid = kb.create_task(kanban_conn, title="plain")
    task = kb.get_task(kanban_conn, tid)

    assert task.project_id is None
    assert task.workspace_kind == "scratch"
    # No branch is persisted — the worker still owns the wt/<id> fallback for
    # genuinely ad-hoc worktree tasks, but unlinked scratch tasks have none.
    assert task.branch_name is None


def test_project_whose_path_is_not_a_git_repo_falls_back_to_scratch(kanban_conn, tmp_path):
    """A worktree upgrade that cannot be materialized must not strand the card:
    it is created as scratch, keeps its project link, and records why."""
    proj = _make_non_repo_project(tmp_path)

    tid = kb.create_task(kanban_conn, title="no repo here", project_id=proj.id)
    task = kb.get_task(kanban_conn, tid)

    assert task.workspace_kind == "scratch"
    assert task.workspace_path is None
    assert task.branch_name is None
    assert task.project_id == proj.id

    skipped = [e.payload for e in kb.list_events(kanban_conn, tid)
               if e.kind == "project_worktree_skipped"]
    assert len(skipped) == 1
    assert skipped[0]["reason"] == "primary_path_not_a_git_repo"
    assert proj.primary_path in skipped[0]["message"]
    assert "scratch" in skipped[0]["message"]


def test_project_inside_a_repo_subdir_still_gets_a_worktree(kanban_conn, tmp_path):
    """The check mirrors dispatch (nearest existing ancestor), so a project
    folder that does not exist yet but lives inside a repo is still a worktree."""
    repo = _git_init(tmp_path / "mono")
    folder = repo / "pkg" / "app"

    with pdb.connect_closing() as pc:
        pid = pdb.create_project(pc, name="Monorepo", folders=[str(folder)])
        proj = pdb.get_project(pc, pid)

    tid = kb.create_task(kanban_conn, title="subdir project", project_id=proj.id)
    task = kb.get_task(kanban_conn, tid)

    assert task.workspace_kind == "worktree"
    assert task.workspace_path == os.path.join(str(folder), ".worktrees", tid)


def test_explicit_worktree_on_non_repo_project_raises_at_creation(kanban_conn, tmp_path):
    """Asking for a worktree explicitly is not a default: refuse it at creation
    with the path named, instead of creating a card that cannot be dispatched."""
    proj = _make_non_repo_project(tmp_path)

    with pytest.raises(ValueError) as excinfo:
        kb.create_task(
            kanban_conn, title="bad anchor", project_id=proj.id, workspace_kind="worktree",
        )

    message = str(excinfo.value)
    assert str(proj.primary_path) in message
    assert "workspace_kind='scratch'" in message
    assert "inside a git repo" in message


# --- explicit workspace_path on a worktree task -----------------------------
# The refusal above used to list "pass an explicit workspace_path" as a remedy,
# but an explicit path overrides the project anchor at dispatch: a path outside
# any repo was accepted at creation and then died at spawn with "worktree path
# ... is not inside a git repo". Creation now applies dispatch's own rule.


def test_explicit_worktree_path_outside_a_repo_raises_at_creation(kanban_conn, tmp_path):
    """Following the old remedy ("pass an explicit workspace_path") must not
    produce a card that cannot be dispatched — even on a repo-backed project."""
    proj = _make_project(tmp_path)
    outside = tmp_path / "elsewhere" / "wt"

    with pytest.raises(ValueError) as excinfo:
        kb.create_task(
            kanban_conn, title="stranded", project_id=proj.id,
            workspace_kind="worktree", workspace_path=str(outside),
        )

    message = str(excinfo.value)
    assert str(outside) in message
    assert "not inside a git repo" in message
    assert "workspace_kind='scratch'" in message
    # Refused before the insert: no half-created card is left behind.
    assert kanban_conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0


def test_explicit_worktree_path_relative_is_refused_at_creation(kanban_conn, tmp_path):
    """Dispatch requires an absolute worktree path; refuse it at creation rather
    than resolving it against whatever CWD the create call happened to run in."""
    proj = _make_project(tmp_path)

    with pytest.raises(ValueError) as excinfo:
        kb.create_task(
            kanban_conn, title="relative", project_id=proj.id,
            workspace_kind="worktree", workspace_path=".worktrees/rel",
        )

    assert "is not absolute" in str(excinfo.value)
    assert kanban_conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0


def test_explicit_worktree_path_inside_a_repo_that_does_not_exist_yet_is_accepted(kanban_conn, tmp_path):
    """Legitimate case: dispatch materializes a not-yet-existing path inside a repo,
    so creation must keep accepting it."""
    repo = _git_init(tmp_path / "mono")
    with pdb.connect_closing() as pc:
        pid = pdb.create_project(pc, name="Monorepo", folders=[str(repo)])
        proj = pdb.get_project(pc, pid)
    target = repo / "pkg" / "app" / ".worktrees" / "future"
    assert not target.exists()

    tid = kb.create_task(
        kanban_conn, title="future wt", project_id=proj.id,
        workspace_kind="worktree", workspace_path=str(target),
    )
    task = kb.get_task(kanban_conn, tid)

    assert task.workspace_kind == "worktree"
    assert task.workspace_path == str(target)


def test_explicit_worktree_path_that_is_a_repo_root_is_accepted(kanban_conn, tmp_path):
    """Legitimate case: a repo root anchors ``<repo>/.worktrees/<id>`` at dispatch,
    so it must keep being accepted even though it is not *inside* a subdirectory."""
    repo = _git_init(tmp_path / "solo")
    (tmp_path / "salvage").mkdir()
    with pdb.connect_closing() as pc:
        # The project itself cannot anchor a worktree — acceptance must come from
        # the explicit path being a repo root, not from the project's own repo.
        pid = pdb.create_project(pc, name="Solo", folders=[str(tmp_path / "salvage")])
        proj = pdb.get_project(pc, pid)

    tid = kb.create_task(
        kanban_conn, title="repo root", project_id=proj.id,
        workspace_kind="worktree", workspace_path=str(repo),
    )
    task = kb.get_task(kanban_conn, tid)

    assert task.workspace_kind == "worktree"
    assert task.workspace_path == str(repo)


# --- cross-profile fallback: ``project_from_source_task`` -------------------
# A worker profile has its own projects.db, so a fan-out child can only learn its
# project's repo from the creator's *task row*. The recovered repo is ``<repo>``
# by construction, but the board is shared across hosts, so that path may not be
# a git repo here — the upgrade must then skip instead of stranding the child.


def _source_task_and_project(kanban_conn, tmp_path):
    proj = _make_project(tmp_path)
    src = kb.create_task(kanban_conn, title="root", project_id=proj.id)
    source = kb.get_task(kanban_conn, src)
    assert source.workspace_kind == "worktree"
    return proj, source


def _hide_project_from_projects_db(monkeypatch):
    """Simulate the cross-profile case: the project id is not in *this* profile's
    projects.db, so ``resolve_project_link`` must take the task-row fallback."""
    monkeypatch.setattr(pdb, "get_project", lambda *a, **k: None)


def test_source_task_fallback_upgrades_child_to_worktree_when_repo_exists(
    kanban_conn, tmp_path, monkeypatch,
):
    proj, source = _source_task_and_project(kanban_conn, tmp_path)
    _hide_project_from_projects_db(monkeypatch)

    tid = kb.create_task(
        kanban_conn, title="child", project_id=proj.id,
        project_source_task_id=source.id,
    )
    task = kb.get_task(kanban_conn, tid)

    assert task.project_id == proj.id
    assert task.workspace_kind == "worktree"
    assert task.workspace_path == os.path.join(
        str(Path(source.workspace_path).parent.parent), ".worktrees", tid)


def test_source_task_fallback_stays_scratch_when_recovered_repo_is_gone(
    kanban_conn, tmp_path, monkeypatch,
):
    """The recovered repo is not a repo on this host: do not upgrade to a worktree
    the dispatcher could not materialize — keep the card and record the skip."""
    proj, source = _source_task_and_project(kanban_conn, tmp_path)
    repo = Path(source.workspace_path).parent.parent
    assert (repo / ".git").exists()
    shutil.rmtree(repo)
    _hide_project_from_projects_db(monkeypatch)

    tid = kb.create_task(
        kanban_conn, title="child", project_id=proj.id,
        project_source_task_id=source.id,
    )
    task = kb.get_task(kanban_conn, tid)

    assert task.project_id == proj.id
    assert task.workspace_kind == "scratch"
    assert task.workspace_path is None
    skipped = [e.payload for e in kb.list_events(kanban_conn, tid)
               if e.kind == "project_worktree_skipped"]
    assert len(skipped) == 1
    assert skipped[0]["reason"] == "primary_path_not_a_git_repo"
    assert str(repo) in skipped[0]["message"]


# --- an explicit path that the *upgrade* would turn into a worktree ----------
# Both upgrade sites keep the caller's ``workspace_path`` while flipping
# ``scratch`` -> ``worktree``: the source-task recovery above and the project's
# own ``primary_path``. Creation therefore validated a path it would then treat
# as a worktree, accepted the card, and dispatch refused it (``not inside a git
# repo``). The rule now runs again after the upgrades — and every path dispatch
# can anchor is still accepted.


def test_source_task_recovery_with_explicit_non_repo_path_raises_at_creation(
    kanban_conn, tmp_path, monkeypatch,
):
    """The recovered repo upgrades the card to a worktree, so the caller's
    explicit path has to hold up as one: refuse it at creation, path named."""
    proj, source = _source_task_and_project(kanban_conn, tmp_path)
    _hide_project_from_projects_db(monkeypatch)
    outside = tmp_path / "elsewhere" / "wt"
    outside.mkdir(parents=True)

    with pytest.raises(ValueError) as excinfo:
        kb.create_task(
            kanban_conn, title="recovered stranded", project_id=proj.id,
            project_source_task_id=source.id,
            workspace_kind="scratch", workspace_path=str(outside),
        )

    message = str(excinfo.value)
    assert str(outside) in message
    assert "not inside a git repo" in message
    assert "workspace_kind='scratch'" in message
    # Refused before the insert: only the source task is on the board.
    assert kanban_conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 1
    assert kanban_conn.execute(
        "SELECT COUNT(*) FROM tasks WHERE title = ?", ("recovered stranded",),
    ).fetchone()[0] == 0


def test_project_scratch_with_explicit_non_repo_path_raises_at_creation(kanban_conn, tmp_path):
    """Same rule through the other upgrade site, with a *resolvable* project:
    reachable from the shipped surface, which exposes ``workspace_kind`` and
    ``workspace_path`` as independent properties."""
    proj = _make_project(tmp_path)
    outside = tmp_path / "elsewhere" / "wt"
    outside.mkdir(parents=True)

    with pytest.raises(ValueError) as excinfo:
        kb.create_task(
            kanban_conn, title="scratch stranded", project_id=proj.id,
            workspace_kind="scratch", workspace_path=str(outside),
        )

    message = str(excinfo.value)
    assert str(outside) in message
    assert "not inside a git repo" in message
    assert "workspace_kind='scratch'" in message
    # Nothing was created: the refusal happens before the insert.
    assert kanban_conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0


def test_source_task_recovery_with_explicit_in_repo_path_stays_a_worktree(
    kanban_conn, tmp_path, monkeypatch,
):
    """Control: an explicit path inside the recovered repo is exactly what the
    two checks must keep accepting — explicit path kept, kind upgraded."""
    proj, source = _source_task_and_project(kanban_conn, tmp_path)
    _hide_project_from_projects_db(monkeypatch)
    inside = Path(source.workspace_path).parent.parent / "pkg" / "wt"

    tid = kb.create_task(
        kanban_conn, title="recovered ok", project_id=proj.id,
        project_source_task_id=source.id,
        workspace_kind="scratch", workspace_path=str(inside),
    )
    task = kb.get_task(kanban_conn, tid)

    assert task.project_id == proj.id
    assert task.workspace_kind == "worktree"
    assert task.workspace_path == str(inside)


def test_project_scratch_with_explicit_in_repo_path_stays_a_worktree(kanban_conn, tmp_path):
    """Control for the project's own upgrade: inside the repo is a worktree,
    and the explicit path wins over ``<repo>/.worktrees/<id>``."""
    proj = _make_project(tmp_path)
    inside = Path(proj.primary_path) / "pkg" / "wt"

    tid = kb.create_task(
        kanban_conn, title="scratch ok", project_id=proj.id,
        workspace_kind="scratch", workspace_path=str(inside),
    )
    task = kb.get_task(kanban_conn, tid)

    assert task.project_id == proj.id
    assert task.workspace_kind == "worktree"
    assert task.workspace_path == str(inside)


# --- project-less tasks -----------------------------------------------------
# A card without a project is not exempt from the creation-time workspace rule:
# ``resolve_project_link`` used to return before validating anything when
# ``project_id`` was empty, so an explicit worktree path that dispatch could not
# anchor was accepted here and only died at spawn. Creation applies the same
# rule dispatch does, and still accepts every shape dispatch can resolve.


def test_projectless_worktree_with_non_repo_explicit_path_raises_at_creation(kanban_conn, tmp_path):
    """No project, explicit worktree path outside any repo: refuse at creation
    with the path named instead of creating a card that cannot be dispatched."""
    non_repo = tmp_path / "not-a-repo" / "sub"
    non_repo.mkdir(parents=True)

    with pytest.raises(ValueError) as excinfo:
        kb.create_task(
            kanban_conn, title="project-less stranded",
            workspace_kind="worktree", workspace_path=str(non_repo),
        )

    message = str(excinfo.value)
    assert str(non_repo) in message
    assert "not inside a git repo" in message
    assert "workspace_kind='scratch'" in message
    # Refused before the insert: no half-created card is left behind.
    assert kanban_conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0


def test_projectless_worktree_with_path_inside_a_repo_is_accepted(kanban_conn, tmp_path):
    """Legitimate project-less case: dispatch materializes a not-yet-existing
    path inside a repo, so the project-less check must keep accepting it."""
    repo = _git_init(tmp_path / "mono")
    target = repo / "wt" / "future"
    assert not target.exists()

    tid = kb.create_task(
        kanban_conn, title="future wt", workspace_kind="worktree",
        workspace_path=str(target),
    )
    task = kb.get_task(kanban_conn, tid)

    assert task.workspace_kind == "worktree"
    assert task.workspace_path == str(target)
    assert task.project_id is None


def test_projectless_worktree_without_a_path_is_accepted(kanban_conn):
    """Dispatch owns the ``wt/<task-id>`` fallback, so there is no explicit path
    to validate — pins against an over-refusal."""
    tid = kb.create_task(kanban_conn, title="ad hoc", workspace_kind="worktree")
    task = kb.get_task(kanban_conn, tid)

    assert task.workspace_kind == "worktree"
    assert task.workspace_path is None
    assert task.project_id is None


def test_projectless_dir_workspace_with_a_plain_directory_is_accepted(kanban_conn, tmp_path):
    """Only ``worktree`` needs a git anchor: a ``dir`` workspace is untouched."""
    plain = tmp_path / "plain-dir"
    plain.mkdir()

    tid = kb.create_task(
        kanban_conn, title="dir task", workspace_kind="dir",
        workspace_path=str(plain),
    )
    task = kb.get_task(kanban_conn, tid)

    assert task.workspace_kind == "dir"
    assert task.workspace_path == str(plain)
    assert task.project_id is None


# --- unresolvable project id with an explicit worktree path -----------------
# ``resolve_project_link`` validated an explicit worktree path only *after* the
# project lookup, and the unresolvable-id branch returned before it: a ghost
# ``project_id`` plus a path dispatch could not anchor was accepted here and
# died at spawn. The rule now runs before the lookup, so every lookup outcome
# is covered by the same refusal — and every path dispatch can resolve is
# still accepted.


def test_unresolvable_project_worktree_with_non_repo_path_raises_at_creation(
    kanban_conn, tmp_path,
):
    """A ghost project id must not smuggle a non-anchorable worktree path past
    creation: the path, not the project lookup, decides."""
    non_repo = tmp_path / "not-a-repo" / "sub"
    non_repo.mkdir(parents=True)

    with pytest.raises(ValueError) as excinfo:
        kb.create_task(
            kanban_conn, title="ghost project", project_id="ghost-project",
            workspace_kind="worktree", workspace_path=str(non_repo),
        )

    message = str(excinfo.value)
    assert str(non_repo) in message
    assert "not inside a git repo" in message
    assert "workspace_kind='scratch'" in message
    # Refused before the insert: no half-created card is left behind.
    assert kanban_conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0


def test_unresolvable_project_worktree_with_relative_path_raises_at_creation(
    kanban_conn, tmp_path,
):
    """Dispatch requires an absolute worktree path; a ghost project id does not
    exempt the card from that rule."""
    with pytest.raises(ValueError) as excinfo:
        kb.create_task(
            kanban_conn, title="ghost relative", project_id="ghost-project",
            workspace_kind="worktree", workspace_path=".worktrees/rel",
        )

    assert "is not absolute" in str(excinfo.value)
    assert kanban_conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0


def test_unresolvable_project_worktree_with_path_inside_a_repo_is_accepted(
    kanban_conn, tmp_path,
):
    """Legitimate case: the ghost id drops the link, and the explicit path inside
    a repo is what dispatch will use — keep accepting it."""
    repo = _git_init(tmp_path / "mono")
    target = repo / "wt" / "future"
    assert not target.exists()

    tid = kb.create_task(
        kanban_conn, title="ghost in repo", project_id="ghost-project",
        workspace_kind="worktree", workspace_path=str(target),
    )
    task = kb.get_task(kanban_conn, tid)

    assert task.project_id is None
    assert task.workspace_kind == "worktree"
    assert task.workspace_path == str(target)


def test_unresolvable_project_worktree_without_a_path_is_accepted(kanban_conn):
    """No explicit path, no project: dispatch owns the ``wt/<task-id>``
    fallback, so there is nothing for creation to validate."""
    tid = kb.create_task(
        kanban_conn, title="ghost no path", project_id="ghost-project",
        workspace_kind="worktree",
    )
    task = kb.get_task(kanban_conn, tid)

    assert task.project_id is None
    assert task.workspace_kind == "worktree"
    assert task.workspace_path is None
