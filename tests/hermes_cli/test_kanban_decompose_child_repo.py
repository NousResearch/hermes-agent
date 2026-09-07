"""A decomposed child must name the repo it is checked out in.

A ``workspace_kind=worktree`` child with no ``workspace_path`` is only
dispatchable when the board carries a ``default_workdir``. On a multi-repo
board there is none, so such a child used to be written anyway and then failed
at spawn, burned its retries and parked as ``blocked`` behind a message that
read like an infrastructure fault.

Inheriting the root's repo is not the fix: measured against the tasks that hit
this failure, the root's repo was the wrong checkout for 3 of the 17 whose
outcome is known, every one of them a fan-out that crossed repos.

A fresh board has ``default_workdir=None``, so these tests get the unanchored
case for free; the one test that needs an anchor sets it through the same call
the ``boards set-default-workdir`` command uses.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_decompose as kd


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _triage(conn, **kw):
    return kb.create_task(conn, title=kw.pop("title", "rough idea"), triage=True, **kw)


def _decompose(tid, children):
    with kbc.connect() as conn:
        return kb.decompose_triage_task(
            conn, tid, root_assignee="orchestrator", children=children, author="decomposer",
        )


def _anchor_the_board(path):
    kb.write_board_metadata(kb.read_board_metadata()["slug"], default_workdir=str(path))


# --------------------------------------------------------------------------
# The child carries the repo it named.
# --------------------------------------------------------------------------

def test_a_child_keeps_the_repo_it_named(kanban_home, tmp_path):
    repo = str(tmp_path / "code" / "frontend")
    with kbc.connect() as conn:
        tid = _triage(conn, title="ship a feature")

    child_ids = _decompose(tid, [{
        "title": "build the UI", "body": "", "assignee": "engineer", "parents": [],
        "workspace_kind": "worktree", "workspace_path": repo, "project_id": "p_front",
    }])

    with kbc.connect() as conn:
        child = kb.get_task(conn, child_ids[0])
    assert child.workspace_kind == "worktree"
    assert child.workspace_path == repo
    assert child.project_id == "p_front"


def test_siblings_may_sit_in_different_repos(kanban_home, tmp_path):
    """The whole point: one card fans out across checkouts."""
    front = str(tmp_path / "code" / "frontend")
    back = str(tmp_path / "code" / "backend")
    with kbc.connect() as conn:
        tid = _triage(conn, title="ship a feature end to end")

    child_ids = _decompose(tid, [
        {"title": "the screen", "body": "", "assignee": "ui", "parents": [],
         "workspace_kind": "worktree", "workspace_path": front, "project_id": "p_front"},
        {"title": "the endpoint", "body": "", "assignee": "api", "parents": [],
         "workspace_kind": "worktree", "workspace_path": back, "project_id": "p_back"},
    ])

    with kbc.connect() as conn:
        paths = {kb.get_task(conn, cid).workspace_path for cid in child_ids}
    assert paths == {front, back}


def test_a_child_does_not_inherit_the_root_repo(kanban_home, tmp_path):
    """Inheriting is the bug, not the fix — the child's own choice wins."""
    root_repo = str(tmp_path / "code" / "frontend")
    child_repo = str(tmp_path / "code" / "backend")
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="root in the frontend", triage=True,
            workspace_kind="worktree", workspace_path=root_repo,
        )

    child_ids = _decompose(tid, [{
        "title": "backend work", "body": "", "assignee": "api", "parents": [],
        "workspace_kind": "worktree", "workspace_path": child_repo, "project_id": "p_back",
    }])

    with kbc.connect() as conn:
        child = kb.get_task(conn, child_ids[0])
    assert child.workspace_path == child_repo
    assert child.workspace_path != root_repo


# --------------------------------------------------------------------------
# An unanchored worktree child is refused — loudly, and atomically.
# --------------------------------------------------------------------------

def test_a_worktree_child_with_no_repo_is_refused(kanban_home):
    with kbc.connect() as conn:
        tid = _triage(conn, title="ship a feature")

    with pytest.raises(ValueError, match="worktree task with no repo"):
        _decompose(tid, [{
            "title": "build it", "body": "", "assignee": "engineer", "parents": [],
            "workspace_kind": "worktree",
        }])


def test_the_refusal_leaves_no_half_built_graph(kanban_home, tmp_path):
    """The fan-out is one transaction: the good sibling must not survive."""
    with kbc.connect() as conn:
        tid = _triage(conn, title="ship a feature")
        before = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]

    with pytest.raises(ValueError):
        _decompose(tid, [
            {"title": "anchored", "body": "", "assignee": "a", "parents": [],
             "workspace_kind": "worktree", "workspace_path": str(tmp_path / "repo")},
            {"title": "unanchored", "body": "", "assignee": "b", "parents": [],
             "workspace_kind": "worktree"},
        ])

    with kbc.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == before
        assert kb.get_task(conn, tid).status == "triage"


def test_a_worktree_root_does_not_launder_its_children(kanban_home, tmp_path):
    """The child inherits the *kind* from the root, so it must still be refused."""
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="root in a repo", triage=True,
            workspace_kind="worktree", workspace_path=str(tmp_path / "code" / "frontend"),
        )

    with pytest.raises(ValueError, match="worktree task with no repo"):
        _decompose(tid, [{"title": "child", "body": "", "assignee": "a", "parents": []}])


# --------------------------------------------------------------------------
# Single-repo boards keep the upstream behaviour untouched.
# --------------------------------------------------------------------------

def test_a_board_default_workdir_still_anchors_the_child(kanban_home, tmp_path):
    _anchor_the_board(tmp_path / "srv" / "repo")
    with kbc.connect() as conn:
        tid = _triage(conn, title="ship a feature")

    child_ids = _decompose(tid, [{
        "title": "build it", "body": "", "assignee": "engineer", "parents": [],
        "workspace_kind": "worktree",
    }])

    with kbc.connect() as conn:
        child = kb.get_task(conn, child_ids[0])
    # Left unset on purpose: dispatch materializes a fresh worktree per child
    # under the board anchor, so siblings never share one checkout.
    assert child.workspace_path is None
    assert child.workspace_kind == "worktree"


def test_scratch_children_are_untouched(kanban_home):
    with kbc.connect() as conn:
        tid = _triage(conn, title="think about something")

    child_ids = _decompose(tid, [{"title": "think", "body": "", "assignee": "a", "parents": []}])

    with kbc.connect() as conn:
        child = kb.get_task(conn, child_ids[0])
    assert child.workspace_kind == "scratch"
    assert child.project_id is None


# --------------------------------------------------------------------------
# The decomposer maps the model's repo name onto real workspace fields.
# --------------------------------------------------------------------------

REPOS = [
    {"name": "vx-website-frontend", "path": "/code/website-frontend", "project_id": "p_a"},
    {"name": "vx-processes", "path": "/code/internal_processes", "project_id": "p_b"},
]


def test_a_named_repo_becomes_workspace_fields():
    assert kd._resolve_repo_choice("t_1", 0, "vx-processes", REPOS) == {
        "workspace_kind": "worktree",
        "workspace_path": "/code/internal_processes",
        "project_id": "p_b",
    }


def test_a_repo_may_be_named_by_path():
    assert kd._resolve_repo_choice("t_1", 0, "/code/website-frontend", REPOS)["project_id"] == "p_a"


@pytest.mark.parametrize("choice", [None, "", "   ", 7, {"name": "x"}])
def test_no_repo_named_leaves_the_child_alone(choice):
    assert kd._resolve_repo_choice("t_1", 0, choice, REPOS) == {}


def test_an_unknown_repo_is_dropped_not_guessed():
    """'website' is a prefix of a real entry — a fuzzy match would pick it."""
    assert kd._resolve_repo_choice("t_1", 0, "website", REPOS) == {}


def test_the_prompt_offers_the_registered_repos():
    rendered = kd._format_repos(REPOS)
    assert "vx-website-frontend: /code/website-frontend" in rendered
    assert "vx-processes: /code/internal_processes" in rendered


def test_the_prompt_says_so_when_nothing_is_registered():
    assert "none registered" in kd._format_repos([])
