"""Structured conversation provenance follows creator lineage, not dependency edges."""
import json
import sqlite3

import pytest

from hermes_cli import kanban_db as kb, kanban_db_connect as kbc
from hermes_cli.kanban_db_graph import decompose_triage_task


@pytest.mark.parametrize("surface", ["direct", "creator", "graph"])
def test_origin_round_trip_and_creator_inheritance(tmp_path, monkeypatch, surface):
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "board.db"))
    origin = {"profile": "default", "session_id": "conversation", "message_row_id": 42,
              "platform": "telegram", "chat_id": "group", "thread_id": "topic",
              "prompt_excerpt": "Prepare a release"}
    with kbc.connect() as conn:
        owner = kb.create_task(conn, title="owner", triage=True, session_id="conversation", origin=origin)
        if surface == "creator":
            tid = kb.create_task(conn, title="child", creator_task_id=owner)
        elif surface == "graph":
            tid = decompose_triage_task(conn, owner, root_assignee="default", children=[{"title": "child"}])[0]
        else:
            tid = owner
        task = kb.get_task(conn, tid)
        assert task.session_id == "conversation"
        assert task.origin == origin
        assert "Prepare a release" in kb.build_worker_context(conn, tid)
        assert kb.get_task(conn, kb.create_task(conn, title="legacy")).origin is None


def test_explicit_child_session_does_not_inherit_another_conversation_origin(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "board.db"))
    with kbc.connect() as conn:
        parent = kb.create_task(conn, title="parent", session_id="old", origin={"session_id": "old"})
        child = kb.create_task(conn, title="child", creator_task_id=parent, session_id="new")
        assert kb.get_task(conn, child).session_id == "new"
        assert kb.get_task(conn, child).origin is None


def test_legacy_migration_is_additive_and_backup_clone_keeps_origin(tmp_path, monkeypatch):
    path = tmp_path / "legacy.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(path))
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="legacy", session_id="conversation")
        conn.execute("ALTER TABLE tasks DROP COLUMN origin")
        conn.commit()
    kbc.init_db(path)
    kbc.init_db(path)
    with kbc.connect_closing(path) as conn:
        task = kb.get_task(conn, tid)
        assert (task.title, task.session_id, task.origin) == ("legacy", "conversation", None)
        originated = kb.create_task(conn, title="new", session_id="conversation",
                                    origin={"session_id": "conversation", "message_row_id": 42})
        with sqlite3.connect(tmp_path / "clone.db") as clone:
            conn.backup(clone)
    with kbc.connect_closing(tmp_path / "clone.db") as conn:
        assert kb.get_task(conn, originated).origin == {"session_id": "conversation", "message_row_id": 42}
        assert kb.get_task(conn, tid).origin is None


def test_secret_redacted_before_origin_is_persisted_or_rendered(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "board.db"))
    secret = "sk-proj-" + "a" * 40
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="safe origin", session_id="conversation",
                             origin={"session_id": "conversation", "prompt_excerpt": "Release using " + secret})
        raw = conn.execute("SELECT origin FROM tasks WHERE id = ?", (tid,)).fetchone()[0]
        task = kb.get_task(conn, tid)
        for value in (raw, json.dumps(task.origin), kb.build_worker_context(conn, tid)):
            assert secret not in value
            assert "Release using" in value


def test_board_export_scrubs_origin_alongside_session_and_transport_targets(tmp_path, monkeypatch):
    from hermes_cli.kanban_transfer import _scrub_local_state

    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "board.db"))
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="portable", session_id="local-session",
                             origin={"session_id": "local-session", "chat_id": "local-chat"})
        _scrub_local_state(conn)
        task = kb.get_task(conn, tid)
        assert task.session_id is None
        assert task.origin is None
