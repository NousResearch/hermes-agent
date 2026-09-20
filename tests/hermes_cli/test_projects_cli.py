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


def test_session_membership_verbs_resolve_prefix_and_lineage_root(capsys, tmp_path):
    from hermes_state import SessionDB

    _run(["create", "App", str(tmp_path)])
    capsys.readouterr()
    db = SessionDB()
    try:
        db.create_session("session-root", source="cli", cwd=str(tmp_path))
        db.end_session("session-root", "compression")
        db.create_session(
            "session-tip", source="cli", cwd=str(tmp_path),
            parent_session_id="session-root",
        )
    finally:
        db.close()

    assert _run(["assign", "session-t", "app"]) == 0
    assert "session-root" in capsys.readouterr().out
    with pdb.connect_closing() as conn:
        project = pdb.get_project(conn, "app")
        assert pdb.session_home_overrides(conn) == {"session-root": project.id}

    assert _run(["sessions", "app"]) == 0
    assert "session-root" in capsys.readouterr().out

    assert _run(["unfile", "session-t"]) == 0
    with pdb.connect_closing() as conn:
        assert pdb.session_home_overrides(conn) == {"session-root": None}

    assert _run(["release", "session-r"]) == 0
    with pdb.connect_closing() as conn:
        assert pdb.session_home_overrides(conn) == {}


def test_assign_rejects_unknown_session_and_project(capsys, tmp_path):
    from hermes_state import SessionDB

    _run(["create", "App", str(tmp_path)])
    capsys.readouterr()
    db = SessionDB()
    try:
        db.create_session("known-session", source="cli", cwd=str(tmp_path))
    finally:
        db.close()

    assert _run(["assign", "missing", "app"]) == 2
    assert "no unique session matches" in capsys.readouterr().err
    assert _run(["assign", "known", "missing-project"]) == 1
    assert "no such project" in capsys.readouterr().err



