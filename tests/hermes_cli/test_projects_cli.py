"""Tests for the `hermes project` CLI dispatch (hermes_cli/projects_cmd)."""

from __future__ import annotations

from types import SimpleNamespace

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


def test_bound_board_metadata_keeps_project_identity(monkeypatch):
    from hermes_cli import kanban_db as kb

    captured = {}
    monkeypatch.setattr(kb, "_normalize_board_slug", lambda value: value)
    monkeypatch.setattr(kb, "board_exists", lambda value: True)
    monkeypatch.setattr(
        kb,
        "write_board_metadata",
        lambda slug, **values: captured.update(slug=slug, **values),
    )

    projects_cmd._sync_board_default_workdir(
        SimpleNamespace(id="p_example", primary_path="/repo/example"),
        "project-board",
    )

    assert captured == {
        "slug": "project-board",
        "default_workdir": "/repo/example",
        "project_id": "p_example",
    }


