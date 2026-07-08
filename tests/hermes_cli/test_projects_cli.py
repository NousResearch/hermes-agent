"""Tests for the `hermes project` CLI dispatch (hermes_cli/projects_cmd)."""

from __future__ import annotations

import argparse

import pytest

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


def test_add_remove_folder(tmp_path):
    _run(["create", "P", str(tmp_path / "a")])
    assert _run(["add-folder", "p", str(tmp_path / "b")]) == 0

    with pdb.connect_closing() as conn:
        proj = pdb.get_project(conn, "p")
        assert len(proj.folders) == 2

    assert _run(["remove-folder", "p", str(tmp_path / "b")]) == 0
    with pdb.connect_closing() as conn:
        assert len(pdb.get_project(conn, "p").folders) == 1


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


def test_use_clear(tmp_path):
    _run(["create", "P", str(tmp_path)])
    _run(["use", "p"])
    with pdb.connect_closing() as conn:
        assert pdb.get_active_id(conn) is not None

    _run(["use"])
    with pdb.connect_closing() as conn:
        assert pdb.get_active_id(conn) is None


def test_unknown_project_returns_error(capsys, tmp_path):
    assert _run(["show", "nope"]) == 1
    assert "no such project" in capsys.readouterr().err


def test_create_with_board_syncs_board_default_workdir(tmp_path):
    # `project create --board <b>` must sync the board's default_workdir to the
    # primary repo, exactly as `bind-board` does — otherwise the binding is
    # inert and cards on the board land in scratch dirs instead of worktrees.
    from hermes_cli import kanban_db as kb

    kb.create_board("appboard")
    repo = str(tmp_path / "repo")
    assert _run(["create", "My App", repo, "--primary", repo, "--board", "appboard"]) == 0

    assert kb.read_board_metadata("appboard").get("default_workdir") == repo


def test_create_without_board_does_not_touch_board_metadata(tmp_path):
    from hermes_cli import kanban_db as kb

    kb.create_board("plainboard")
    repo = str(tmp_path / "repo")
    assert _run(["create", "P", repo, "--primary", repo]) == 0
    assert kb.read_board_metadata("plainboard").get("default_workdir") is None
