"""Tests for the per-profile Projects store (hermes_cli/projects_db)."""

from __future__ import annotations

import os

import pytest

from hermes_cli import projects_db as pdb


@pytest.fixture
def conn(tmp_path):
    c = pdb.connect(db_path=tmp_path / "projects.db")
    try:
        yield c
    finally:
        c.close()






def test_discovery_policy_change_clears_only_discovered_rows(conn):
    project_id = pdb.create_project(conn, name="Explicit", folders=["/www/explicit"])
    pdb.record_discovered_repos(
        conn, [("/www/scanned", "scanned")], policy_key="policy-a"
    )

    assert pdb.reconcile_discovered_repos_policy(conn, "policy-b") is True
    assert pdb.list_discovered_repos(conn) == []
    assert pdb.get_project(conn, project_id) is not None
    assert pdb.get_discovery_policy_key(conn) == "policy-b"






def test_create_get_list(conn):
    pid = pdb.create_project(conn, name="Hermes Agent", folders=["/tmp/hermes"])
    proj = pdb.get_project(conn, pid)

    assert proj is not None
    assert proj.slug == "hermes-agent"
    assert proj.name == "Hermes Agent"
    # First folder becomes primary.
    assert proj.primary_path == "/tmp/hermes"
    assert [f.path for f in proj.folders] == ["/tmp/hermes"]
    assert proj.folders[0].is_primary is True

    # Lookup by slug too.
    assert pdb.get_project(conn, "hermes-agent").id == pid
    assert len(pdb.list_projects(conn)) == 1












def test_project_for_path_skips_archived(conn):
    pid = pdb.create_project(conn, name="P", folders=["/www/app"])
    pdb.archive_project(conn, pid)

    assert pdb.project_for_path(conn, "/www/app/src") is None
    # Archived hidden from the default list but visible with include_archived.
    assert pdb.list_projects(conn) == []
    assert len(pdb.list_projects(conn, include_archived=True)) == 1

    pdb.restore_project(conn, pid)
    assert pdb.project_for_path(conn, "/www/app/src").id == pid






def test_per_profile_isolation(tmp_path):
    # Two distinct DB paths stand in for two profiles' HERMES_HOME.
    a = pdb.connect(db_path=tmp_path / "a" / "projects.db")
    b = pdb.connect(db_path=tmp_path / "b" / "projects.db")
    try:
        pdb.create_project(a, name="Only In A", folders=["/a"])
        pdb.record_discovered_repos(a, [("/a/scanned", "scanned")])

        assert [p.slug for p in pdb.list_projects(a)] == ["only-in-a"]
        assert pdb.list_projects(b) == []
        assert [row["root"] for row in pdb.list_discovered_repos(a)] == [
            "/a/scanned"
        ]
        assert pdb.list_discovered_repos(b) == []
    finally:
        a.close()
        b.close()


def test_conversation_group_crud_and_session_assignment(conn):
    project_id = pdb.create_project(conn, name="SK 업무")

    first = pdb.create_conversation_group(conn, project_id, "ITO LLM Wiki")
    second = pdb.create_conversation_group(conn, project_id, "Tib/RV")

    assert [group.name for group in pdb.list_conversation_groups(conn, project_id)] == [
        "ITO LLM Wiki",
        "Tib/RV",
    ]

    assert pdb.update_conversation_group(conn, second, name="Tib / RV") is True
    pdb.reorder_conversation_groups(conn, project_id, [second, first])
    assert [group.name for group in pdb.list_conversation_groups(conn, project_id)] == [
        "Tib / RV",
        "ITO LLM Wiki",
    ]

    pdb.assign_session(conn, "session-1", project_id, second)
    assignment = pdb.get_session_assignments(conn)["session-1"]
    assert assignment.project_id == project_id
    assert assignment.group_id == second

    assert pdb.delete_conversation_group(conn, second) is True
    assignment = pdb.get_session_assignments(conn)["session-1"]
    assert assignment.project_id == project_id
    assert assignment.group_id is None


def test_session_assignment_can_move_to_ungrouped_without_changing_project(conn):
    project_id = pdb.create_project(conn, name="주식")
    group_id = pdb.create_conversation_group(conn, project_id, "SK하이닉스")
    pdb.assign_session(conn, "session-1", project_id, group_id)

    pdb.assign_session(conn, "session-1", project_id, None)

    assignment = pdb.get_session_assignments(conn)["session-1"]
    assert assignment.project_id == project_id
    assert assignment.group_id is None


