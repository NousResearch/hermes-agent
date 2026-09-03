"""Tests for the `hermes project` CLI dispatch (hermes_cli/projects_cmd)."""

from __future__ import annotations

import argparse

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import projects_cmd
from hermes_cli import projects_db as pdb


def _run(argv):
    """Build the project subparser, parse argv, and dispatch. Returns rc."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    p = projects_cmd.build_parser(sub)
    p.set_defaults(func=projects_cmd.projects_command)
    args = parser.parse_args(["project", *argv])
    return projects_cmd.projects_command(args)


def test_create_list_show(capsys, tmp_path):
    assert _run(["create", "My App", str(tmp_path), "--use"]) == 0
    out = capsys.readouterr().out
    assert "Created project" in out

    with pdb.connect_closing() as conn:
        projects = pdb.list_projects(conn)
        assert len(projects) == 1
        assert projects[0].name == "My App"
        # --use set it active.
        assert pdb.get_active_id(conn) == projects[0].id

    assert _run(["list"]) == 0
    assert "my-app" in capsys.readouterr().out

    assert _run(["show", "my-app"]) == 0
    assert "My App" in capsys.readouterr().out




def test_rename_and_archive(tmp_path):
    _run(["create", "Old Name", str(tmp_path)])
    assert _run(["rename", "old-name", "New Name"]) == 0
    with pdb.connect_closing() as conn:
        assert pdb.get_project(conn, "old-name").name == "New Name"

    assert _run(["archive", "old-name"]) == 0
    with pdb.connect_closing() as conn:
        assert pdb.list_projects(conn) == []
        assert len(pdb.list_projects(conn, include_archived=True)) == 1

    assert _run(["restore", "old-name"]) == 0
    with pdb.connect_closing() as conn:
        assert len(pdb.list_projects(conn)) == 1


def test_bind_board_updates_both_sides_and_unbind_preserves_board(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    kb.create_board("widget", name="Widget Board", description="Keep me")
    board_conn = kb.connect(board="widget")
    try:
        existing_task = kb.create_task(board_conn, title="existing", board="widget")
    finally:
        board_conn.close()

    assert _run(["create", "Widget", str(repo)]) == 0
    assert _run(["bind-board", "widget", "widget"]) == 0

    with pdb.connect_closing() as conn:
        project = pdb.get_project(conn, "widget")
        assert project is not None
        assert project.board_slug == "widget"

    metadata = kb.read_board_metadata("widget")
    assert metadata["project_id"] == project.id
    assert metadata["default_workdir"] == str(repo)
    assert metadata["name"] == "Widget Board"
    assert metadata["description"] == "Keep me"

    board_conn = kb.connect(board="widget")
    try:
        inherited_task = kb.create_task(board_conn, title="inherited", board="widget")
        assert kb.get_task(board_conn, existing_task) is not None
        assert kb.get_task(board_conn, inherited_task).project_id == project.id
    finally:
        board_conn.close()

    assert _run(["bind-board", "widget"]) == 0
    with pdb.connect_closing() as conn:
        assert pdb.get_project(conn, "widget").board_slug is None

    metadata = kb.read_board_metadata("widget")
    assert metadata["project_id"] is None
    assert metadata["default_workdir"] == str(repo)
    assert metadata["name"] == "Widget Board"
    assert metadata["description"] == "Keep me"


def test_create_with_board_updates_reciprocal_metadata(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    kb.create_board("widget")

    assert _run(["create", "Widget", str(repo), "--board", "widget"]) == 0

    with pdb.connect_closing() as conn:
        project = pdb.get_project(conn, "widget")
        assert project is not None
        assert project.board_slug == "widget"

    metadata = kb.read_board_metadata("widget")
    assert metadata["project_id"] == project.id
    assert metadata["default_workdir"] == str(repo)




