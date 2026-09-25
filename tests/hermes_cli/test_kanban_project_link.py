"""Kanban <-> Projects integration: project-linked tasks get a deterministic
worktree path + branch instead of the random ``wt/<task-id>`` fallback."""

from __future__ import annotations

import json
import os
from typing import Optional

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


def _make_project(name="Web App", repo="/tmp/webapp"):
    with pdb.connect_closing() as pc:
        pid = pdb.create_project(pc, name=name, folders=[repo])
        return pdb.get_project(pc, pid)


def test_project_linked_task_gets_deterministic_worktree_and_branch(kanban_conn):
    proj = _make_project()
    tid = kb.create_task(kanban_conn, title="Add login", project_id=proj.slug)
    task = kb.get_task(kanban_conn, tid)

    assert task.project_id == proj.id
    assert task.workspace_kind == "worktree"
    # Worktree dir anchored under the project's primary repo, keyed on task id.
    assert task.workspace_path == os.path.join(proj.primary_path, ".worktrees", tid)
    # Deterministic branch: <slug>/<task-id>-<title-slug>. NOT a random wt/...
    assert task.branch_name == f"{proj.slug}/{tid}-add-login"
    assert not task.branch_name.startswith("wt/")


def test_explicit_branch_overrides_project_default(kanban_conn):
    proj = _make_project()
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


# ---------------------------------------------------------------------------
# A filer whose profile cannot resolve the project it names
#
# Projects live in a per-profile projects.db; the Kanban board is shared. So a
# project that is perfectly valid board-wide is unresolvable from the profile
# filing the card — a worker profile can hold no projects at all, and another
# may track the same repo under a different id. This used to end in a silent
# scratch card with no project link and no trace of what was asked for.
# ---------------------------------------------------------------------------

def _created_link(conn, tid: str) -> Optional[dict]:
    """The ``project_link`` report the card's ``created`` event carries, if any."""
    created = [e for e in kb.list_events(conn, tid) if e.kind == "created"]
    assert created, f"no created event for {tid}"
    return created[-1].payload.get("project_link")


@pytest.fixture
def cross_profile_filer(tmp_path, monkeypatch):
    """``(conn, owner_task, project_id, repo, homes)`` for the incident's shape.

    ``repo`` is the project's primary repo, seeded into home A's projects.db
    along with the project's first card. The caller then works as a second
    profile (home B): same board, a projects.db that has never heard of the
    project. ``homes`` exposes both HERMES_HOMEs for tests that need to file
    the same card as either profile.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    home_a, home_b = tmp_path / "home-a", tmp_path / "home-b"
    home_a.mkdir()
    home_b.mkdir()
    board = tmp_path / "kanban.db"
    # Pin the board: every surface (handler included) opens THIS file while
    # HERMES_HOME moves between the two profiles' stores.
    monkeypatch.setenv("HERMES_KANBAN_DB", str(board))

    monkeypatch.setenv("HERMES_HOME", str(home_a))
    with pdb.connect_closing() as pc:
        project_id = pdb.create_project(pc, name="Web App", folders=[str(repo)])
    conn = kbc.connect()
    try:
        owner = kb.create_task(conn, title="coordinator card", project_id=project_id)
        assert kb.get_task(conn, owner).project_id == project_id
        # The filer is a DIFFERENT profile: its own store, the same board.
        monkeypatch.setenv("HERMES_HOME", str(home_b))
        assert not (home_b / "projects.db").exists()
        yield conn, kb.get_task(conn, owner), project_id, repo, {"a": home_a, "b": home_b}
    finally:
        conn.close()


def test_filer_in_another_profile_keeps_the_project_it_names(cross_profile_filer):
    """The incident: the id is valid on the board and in the coordinator's
    store, and absent from the filing worker's — the card must keep its link."""
    conn, owner, project_id, repo, _homes = cross_profile_filer
    assert owner.workspace_path == os.path.join(str(repo), ".worktrees", owner.id)

    child = kb.create_task(
        conn, title="fan-out child", project_id=project_id,
        project_source_task_id=owner.id)

    task = kb.get_task(conn, child)
    assert task.project_id == project_id  # was None: the link was dropped
    assert task.workspace_kind == "worktree"
    assert task.workspace_path == os.path.join(str(repo), ".worktrees", child)
    link = _created_link(conn, child)
    assert link == {"requested": project_id, "resolution": "recovered_from_source_task",
                    "source_task": owner.id}


@pytest.mark.parametrize("source", [None, "t_not_a_card"])
def test_board_row_recovers_a_project_the_filer_is_not_working_in(
    cross_profile_filer, source,
):
    """No filer card (or one in another project): the board's canonical worktree
    row for that project is still the shared truth for where its repo is."""
    conn, owner, project_id, repo, _homes = cross_profile_filer

    child = kb.create_task(
        conn, title="child", project_id=project_id, project_source_task_id=source)

    task = kb.get_task(conn, child)
    assert task.project_id == project_id
    assert task.workspace_path == os.path.join(str(repo), ".worktrees", child)
    link = _created_link(conn, child)
    assert link is not None
    assert link["resolution"] == "recovered_from_board_task"
    assert link["source_task"] == owner.id


def test_unresolvable_project_request_is_recorded_not_silently_dropped(
    cross_profile_filer,
):
    """Nothing on the board carries the id: the card is created without the
    link (unchanged), and the dropped request is on the card for a reader."""
    conn, owner, _project_id, _repo, _homes = cross_profile_filer

    child = kb.create_task(
        conn, title="child", project_id="p_nowhere",
        project_source_task_id=owner.id)

    task = kb.get_task(conn, child)
    assert task.project_id is None
    assert task.workspace_kind == "scratch"
    link = _created_link(conn, child)
    assert link is not None
    assert link["requested"] == "p_nowhere"
    assert link["resolution"] == "dropped"
    assert "p_nowhere" in link["reason"]


@pytest.mark.parametrize("workspace_kind", [None, "worktree", "scratch"])
def test_recovered_link_behaves_like_a_locally_resolved_one(
    cross_profile_filer, monkeypatch, workspace_kind,
):
    """The recovered project must be indistinguishable from the one this
    profile's own store would have handed over — same workspace kind, same path
    convention under the repo, same branch — for every workspace request."""
    conn, owner, project_id, repo, homes = cross_profile_filer
    extra = {} if workspace_kind is None else {"workspace_kind": workspace_kind}

    monkeypatch.setenv("HERMES_HOME", str(homes["a"]))
    local = kb.get_task(conn, kb.create_task(
        conn, title="resolved locally", project_id=project_id, **extra))

    monkeypatch.setenv("HERMES_HOME", str(homes["b"]))
    recovered = kb.get_task(conn, kb.create_task(
        conn, title="resolved from the board", project_id=project_id,
        project_source_task_id=owner.id, **extra))

    assert local is not None and recovered is not None
    assert local.project_id == recovered.project_id == project_id
    assert recovered.workspace_kind == local.workspace_kind
    if recovered.workspace_kind == "worktree":
        assert recovered.workspace_path == os.path.join(str(repo), ".worktrees", recovered.id)
        slug = local.branch_name.split("/")[0]
        assert local.branch_name == f"{slug}/{local.id}-resolved-locally"
        assert recovered.branch_name == f"{slug}/{recovered.id}-resolved-from-the-board"
    else:
        assert (local.workspace_path, recovered.workspace_path) == (None, None)


def test_a_locally_resolved_project_is_not_reported(kanban_conn):
    """The ordinary create stays quiet: the report is for the asymmetric case."""
    proj = _make_project()
    tid = kb.create_task(kanban_conn, title="Add login", project_id=proj.slug)

    assert _created_link(kanban_conn, tid) is None


def test_worker_filing_a_card_keeps_the_project_through_the_tool(
    cross_profile_filer, monkeypatch,
):
    """End-to-end: the filing run's own card is the source the handler hands the
    DB, so an explicit ``project=`` survives a profile that cannot resolve it.
    A request nothing can resolve comes back reported instead of silent."""
    conn, owner, project_id, _repo, _homes = cross_profile_filer
    from tools import kanban_tools as kt

    monkeypatch.setenv("HERMES_PROFILE", "test-worker")
    monkeypatch.setenv("HERMES_KANBAN_TASK", owner.id)
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)

    filed = json.loads(kt._handle_create(
        {"title": "filed by the worker", "assignee": "peer", "project": project_id}))
    assert filed["ok"] is True, filed
    assert filed["project_id"] == project_id
    assert filed["workspace_kind"] == "worktree"
    assert filed["project_link"] == {
        "requested": project_id, "resolution": "recovered_from_source_task",
        "source_task": owner.id}
    assert kb.get_task(conn, filed["task_id"]).project_id == project_id

    dropped = json.loads(kt._handle_create(
        {"title": "filed with a stale id", "assignee": "peer", "project": "p_nowhere"}))
    assert dropped["ok"] is True, dropped
    assert dropped["project_id"] is None
    assert dropped["project_link"]["resolution"] == "dropped"
    assert dropped["project_link"]["requested"] == "p_nowhere"


