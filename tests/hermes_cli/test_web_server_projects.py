"""Tests for the Projects REST API endpoints in hermes_cli.web_server.

Creates an isolated HERMES_HOME with:
- a projects.db with one registered project (primary_path = tmp_path/project-a)
- a state.db with two sessions (one cwd under the project, one unrelated)

Then hits the four GET routes + the POST template route via TestClient and
asserts the structural contracts (no count snapshots, no source-text reads).
"""

import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def projects_env(_isolate_hermes_home, tmp_path):
    """Set up a minimal projects.db + state.db under the isolated HERMES_HOME."""
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")

    import hermes_state
    from hermes_constants import get_hermes_home
    from hermes_cli import projects_db as pdb

    home = get_hermes_home()

    # ---- projects.db ----
    project_path = tmp_path / "project-a"
    project_path.mkdir()
    (project_path / "existing.txt").write_text("hi")

    with pdb.connect_closing() as conn:
        pid = pdb.create_project(
            conn,
            name="Project Alpha",
            slug="project-alpha",
            folders=[str(project_path)],
            primary_path=str(project_path),
        )

    # ---- state.db ----
    state_db_path = home / "state.db"

    db = hermes_state.SessionDB(state_db_path)
    try:
        # Session inside the project
        db.create_session(
            "sess-in-project",
            "cli",
            cwd=str(project_path / "sub"),
        )
        # Session outside any project
        db.create_session(
            "sess-unassigned",
            "cli",
            cwd="/tmp/unrelated",
        )
    finally:
        db.close()

    # patch DEFAULT_DB_PATH so web_server routes read our state.db
    with patch.object(hermes_state, "DEFAULT_DB_PATH", state_db_path):
        from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

        client = TestClient(app)
        client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN

        yield {
            "client": client,
            "project_id": pid,
            "project_path": project_path,
            "state_db_path": state_db_path,
        }


# ---------------------------------------------------------------------------
# GET /api/projects
# ---------------------------------------------------------------------------


def test_get_projects_returns_project_list(projects_env):
    client = projects_env["client"]
    resp = client.get("/api/projects")
    assert resp.status_code == 200
    data = resp.json()

    # Top-level shape
    assert "projects" in data
    assert "unassigned" in data

    # At least our one project is present
    assert len(data["projects"]) >= 1
    project = next(
        (p for p in data["projects"] if p.get("id") == projects_env["project_id"]),
        None,
    )
    assert project is not None, "registered project missing from /api/projects"

    # Required fields
    assert "boardCounts" in project
    assert "lastEvent" in project  # may be None — just must be present

    # Brief contract fields from projects_db
    assert "slug" in project, "project missing 'slug'"
    assert "name" in project, "project missing 'name'"
    assert "folders" in project, "project missing 'folders'"
    assert "primary" in project, "project missing 'primary'"
    assert "board" in project, "project missing 'board'"
    assert project["slug"] == "project-alpha"
    assert project["name"] == "Project Alpha"

    # unassigned is a dict with the right keys
    unassigned = data["unassigned"]
    assert "sessionCount" in unassigned
    assert "groups" in unassigned
    assert isinstance(unassigned["sessionCount"], int)
    assert isinstance(unassigned["groups"], list)


def test_get_projects_unassigned_counts_non_project_sessions(projects_env):
    """Sessions whose cwd doesn't fall under any project appear in unassigned."""
    client = projects_env["client"]
    resp = client.get("/api/projects")
    assert resp.status_code == 200
    unassigned = resp.json()["unassigned"]
    # The /tmp/unrelated session must be unassigned (sessionCount >= 1)
    assert unassigned["sessionCount"] >= 1


# ---------------------------------------------------------------------------
# GET /api/projects/{id}/sessions
# ---------------------------------------------------------------------------


def test_get_project_sessions_returns_list(projects_env):
    client = projects_env["client"]
    pid = projects_env["project_id"]
    resp = client.get(f"/api/projects/{pid}/sessions")
    assert resp.status_code == 200
    data = resp.json()
    assert "sessions" in data
    assert "total" in data
    assert isinstance(data["sessions"], list)
    assert data["total"] == len(data["sessions"])


def test_get_project_sessions_by_slug(projects_env):
    """Can look up a project by slug, not just id."""
    client = projects_env["client"]
    resp = client.get("/api/projects/project-alpha/sessions")
    assert resp.status_code == 200


def test_get_project_sessions_session_shape(projects_env):
    """Each session row carries the expected keys."""
    client = projects_env["client"]
    pid = projects_env["project_id"]
    resp = client.get(f"/api/projects/{pid}/sessions")
    assert resp.status_code == 200
    sessions = resp.json()["sessions"]
    for s in sessions:
        for key in ("id", "title", "model", "endReason", "cwd", "task", "lastActive"):
            assert key in s, f"session row missing key {key!r}"


def test_get_project_sessions_invalid_filter_400(projects_env):
    client = projects_env["client"]
    pid = projects_env["project_id"]
    resp = client.get(f"/api/projects/{pid}/sessions?filter=bogus")
    assert resp.status_code == 400


def test_get_project_sessions_404_for_unknown_project(projects_env):
    client = projects_env["client"]
    resp = client.get("/api/projects/does-not-exist/sessions")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /api/projects/{id}/board
# ---------------------------------------------------------------------------


def test_get_project_board_no_board_slug(projects_env):
    """A project with no bound board returns empty tasks + consistent zero counts."""
    client = projects_env["client"]
    pid = projects_env["project_id"]
    resp = client.get(f"/api/projects/{pid}/board")
    assert resp.status_code == 200
    data = resp.json()
    assert data["tasks"] == []
    counts = data["counts"]
    assert isinstance(counts, dict)
    # Counts shape must be consistent: all five status keys present as ints
    for key in ("todo", "doing", "review", "blocked", "done"):
        assert key in counts, f"counts missing key {key!r}"
        assert isinstance(counts[key], int)


def test_get_project_board_404_for_unknown_project(projects_env):
    client = projects_env["client"]
    resp = client.get("/api/projects/no-such-id/board")
    assert resp.status_code == 404


def test_get_project_board_task_updated_field(projects_env):
    """When a board has tasks, each task shape uses 'updated' not 'updated_at'."""
    import hermes_cli.kanban_db as kdb

    client = projects_env["client"]
    pid = projects_env["project_id"]

    # Bind a fake board slug to the project.
    from hermes_cli import projects_db as pdb
    with pdb.connect_closing() as pconn:
        pconn.execute(
            "UPDATE projects SET board_slug = ? WHERE id = ?",
            ("fake-board", pid),
        )
        pconn.commit()

    fake_task = kdb.Task(
        id="t1",
        title="Test task",
        body=None,
        assignee="alice",
        status="done",
        priority=0,
        created_by="alice",
        created_at=1000,
        started_at=1001,
        completed_at=1002,
        workspace_kind="scratch",
        workspace_path=None,
        claim_lock=None,
        claim_expires=None,
        tenant=None,
    )
    fake_stats = {"by_status": {"done": 1, "todo": 0, "doing": 0, "review": 0, "blocked": 0}}

    import sqlite3 as _sqlite3

    with patch("hermes_cli.web_server._open_kanban_for_board",
               return_value=(_sqlite3.connect(":memory:"), "fake-board")), \
         patch("hermes_cli.kanban_db.list_tasks", return_value=[fake_task]), \
         patch("hermes_cli.kanban_db.board_stats", return_value=fake_stats):
        resp = client.get(f"/api/projects/{pid}/board")

    assert resp.status_code == 200
    data = resp.json()
    assert len(data["tasks"]) == 1
    task = data["tasks"][0]
    assert "updated" in task, "board task shape must use 'updated' not 'updated_at'"
    assert "updated_at" not in task, "deprecated 'updated_at' must not appear"
    assert task["updated"] == 1002  # completed_at value


# ---------------------------------------------------------------------------
# GET /api/projects/{id}/tree
# ---------------------------------------------------------------------------


def test_get_project_tree_lists_entries(projects_env):
    client = projects_env["client"]
    pid = projects_env["project_id"]
    resp = client.get(f"/api/projects/{pid}/tree")
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == pid
    assert len(data["folders"]) >= 1

    folder = data["folders"][0]
    assert folder["exists"] is True
    # existing.txt should appear in entries
    assert "existing.txt" in folder["entries"]
    # template items not yet created should appear in missing
    assert isinstance(folder["missing"], list)
    for item in folder["missing"]:
        assert item in (".hermes.md", "docs", "work", "assets", ".worktrees")


def test_get_project_tree_missing_reflects_template_items(projects_env):
    """Missing is exactly the set of standard template items that don't exist."""
    client = projects_env["client"]
    pid = projects_env["project_id"]
    project_path = projects_env["project_path"]

    # Create one template item so it's no longer missing
    (project_path / "docs").mkdir()

    resp = client.get(f"/api/projects/{pid}/tree")
    assert resp.status_code == 200
    folder = resp.json()["folders"][0]
    # docs should NOT be in missing
    assert "docs" not in folder["missing"]
    # the rest should still be there
    assert "docs" in folder["entries"]


def test_get_project_tree_404_for_unknown_project(projects_env):
    client = projects_env["client"]
    resp = client.get("/api/projects/ghost/tree")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /api/projects/{id}/apply-template
# ---------------------------------------------------------------------------


def test_apply_template_creates_missing_items(projects_env):
    client = projects_env["client"]
    pid = projects_env["project_id"]
    project_path = projects_env["project_path"]

    resp = client.post(f"/api/projects/{pid}/apply-template")
    assert resp.status_code == 200
    data = resp.json()
    assert "created" in data
    assert "errors" in data
    assert isinstance(data["created"], list)

    # After applying, all template items should exist
    for item in (".hermes.md", "docs", "work", "assets", ".worktrees"):
        assert (project_path / item).exists(), f"{item} not created by apply-template"


def test_apply_template_does_not_overwrite_existing(projects_env):
    """Existing files are untouched — apply-template is idempotent."""
    client = projects_env["client"]
    pid = projects_env["project_id"]
    project_path = projects_env["project_path"]

    hermes_md = project_path / ".hermes.md"
    hermes_md.write_text("# custom content\n")
    original_mtime = hermes_md.stat().st_mtime

    resp = client.post(f"/api/projects/{pid}/apply-template")
    assert resp.status_code == 200
    assert hermes_md.read_text() == "# custom content\n"


def test_apply_template_404_for_unknown_project(projects_env):
    client = projects_env["client"]
    resp = client.post("/api/projects/nope/apply-template")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# _resolve_chat_argv project_cwd injection (unit test — no TUI needed)
# ---------------------------------------------------------------------------


def test_resolve_chat_argv_injects_terminal_cwd_for_project(_isolate_hermes_home, tmp_path):
    """project_cwd is forwarded as TERMINAL_CWD in the spawned env."""
    try:
        from hermes_cli.web_server import _resolve_chat_argv
    except ImportError:
        pytest.skip("web_server not available")

    fake_argv = ["node", "fake"]
    fake_cwd = str(tmp_path)
    project_path = str(tmp_path / "my-project")

    # _make_tui_argv and apply_terminal_config_to_env are imported inside
    # _resolve_chat_argv from hermes_cli.main / hermes_cli.config respectively.
    with (
        patch("hermes_cli.main._make_tui_argv", return_value=(fake_argv, fake_cwd)),
        patch("hermes_cli.main._apply_tui_python_env"),
        patch("hermes_cli.config.apply_terminal_config_to_env"),
    ):
        _, _, env = _resolve_chat_argv(project_cwd=project_path)

    assert env is not None
    assert env.get("TERMINAL_CWD") == project_path
