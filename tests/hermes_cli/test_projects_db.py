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


def test_conversation_binding_crud_roundtrip_and_serialization(conn):
    pid = pdb.create_project(conn, name="Support")

    binding = pdb.bind_conversation(
        conn,
        pid,
        platform="Telegram",
        chat_id="chat-1",
        thread_id="topic-7",
        alias="Launch topic",
    )
    project = pdb.get_project(conn, pid)

    assert binding.project_id == pid
    assert binding.platform == "telegram"
    assert binding.chat_id == "chat-1"
    assert binding.thread_id == "topic-7"
    assert binding.alias == "Launch topic"
    assert project.conversation_bindings == [binding]
    assert project.to_dict()["conversation_bindings"][0]["target_key"] == binding.target_key


def test_conversation_binding_threadless_normalizes_empty_thread(conn):
    pid = pdb.create_project(conn, name="DMs")

    bound = pdb.bind_conversation(conn, pid, platform="slack", chat_id="C123", thread_id="")
    same = pdb.get_conversation_binding(conn, platform="slack", chat_id="C123", thread_id=None)

    assert bound.thread_id is None
    assert same is not None
    assert same.target_key == bound.target_key


def test_conversation_binding_one_project_per_target_and_alias_update(conn):
    first = pdb.create_project(conn, name="First")
    second = pdb.create_project(conn, name="Second")

    pdb.bind_conversation(conn, first, platform="telegram", chat_id="chat", thread_id="thread", alias="Old")
    moved = pdb.bind_conversation(conn, second, platform="telegram", chat_id="chat", thread_id="thread", alias="New")
    updated = pdb.bind_conversation(conn, second, platform="telegram", chat_id="chat", thread_id="thread", alias="Updated")

    assert moved.project_id == second
    assert updated.alias == "Updated"
    assert pdb.list_conversation_bindings(conn, project_id=first) == []
    assert [b.project_id for b in pdb.list_conversation_bindings(conn)] == [second]


def test_conversation_binding_rejects_archived_project(conn):
    pid = pdb.create_project(conn, name="Archived")
    pdb.archive_project(conn, pid)

    with pytest.raises(ValueError, match="archived project"):
        pdb.bind_conversation(conn, pid, platform="telegram", chat_id="chat", thread_id="thread")

    assert pdb.list_conversation_bindings(conn) == []


def test_conversation_binding_unbind_and_cascade_delete(conn):
    pid = pdb.create_project(conn, name="Bound")
    pdb.bind_conversation(conn, pid, platform="telegram", chat_id="chat", thread_id=None)

    assert pdb.unbind_conversation(conn, platform="telegram", chat_id="chat", thread_id="missing") is False
    assert pdb.unbind_conversation(conn, platform="telegram", chat_id="chat", thread_id=None) is True
    assert pdb.list_conversation_bindings(conn) == []

    pdb.bind_conversation(conn, pid, platform="telegram", chat_id="chat", thread_id=None)
    pdb.delete_project(conn, pid)
    assert pdb.list_conversation_bindings(conn) == []


def test_active_pointer(conn):
    pid = pdb.create_project(conn, name="P")
    assert pdb.get_active_id(conn) is None

    pdb.set_active(conn, pid)
    assert pdb.get_active_id(conn) == pid

    pdb.set_active(conn, None)
    assert pdb.get_active_id(conn) is None




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


