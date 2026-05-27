"""Contract tests for Kanban storage backends.

These tests intentionally exercise the existing SQLite-backed ``kanban_db``
module first.  Future storage backends should be wired into the ``store``
fixture and pass the same behavior without changing the assertions.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


def test_kanban_store_protocol_module_imports():
    from hermes_cli.kanban_store import (
        KanbanStore,
        KanbanStoreConnection,
        StoreCapabilities,
    )

    assert KanbanStore is not None
    assert KanbanStoreConnection is not None
    assert StoreCapabilities().backend == "unknown"


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an initialized default board."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_HOME", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def store(kanban_home):
    """Current backend under contract: the legacy SQLite facade."""
    return kb


def test_store_contract_create_list_show_roundtrip(store):
    with store.connect() as conn:
        task_id = store.create_task(
            conn,
            title="contract create",
            body="persist this body",
            assignee="alice",
            created_by="contract-test",
            priority=7,
        )
        shown = store.get_task(conn, task_id)
        listed = store.list_tasks(conn)

    assert shown is not None
    assert shown.id == task_id
    assert shown.title == "contract create"
    assert shown.body == "persist this body"
    assert shown.assignee == "alice"
    assert shown.created_by == "contract-test"
    assert shown.priority == 7
    assert shown.status == "ready"
    assert [task.id for task in listed] == [task_id]


def test_store_contract_parent_promotion_requires_completed_parent(store):
    with store.connect() as conn:
        parent_id = store.create_task(conn, title="parent")
        child_id = store.create_task(conn, title="child", parents=[parent_id])
        assert store.get_task(conn, child_id).status == "todo"

        promoted_before = store.recompute_ready(conn)
        assert promoted_before == 0
        assert store.get_task(conn, child_id).status == "todo"

        assert store.complete_task(conn, parent_id, result="done") is True
        child = store.get_task(conn, child_id)

    assert child.status == "ready"


def test_store_contract_claim_is_exclusive_and_creates_run_event(store):
    with store.connect() as conn:
        task_id = store.create_task(conn, title="claim me", assignee="alice")

        first = store.claim_task(conn, task_id, claimer="worker-a")
        second = store.claim_task(conn, task_id, claimer="worker-b")
        task = store.get_task(conn, task_id)
        events = store.list_events(conn, task_id)

    assert first is not None
    assert first.id == task_id
    assert second is None
    assert task.status == "running"
    assert task.claim_lock == "worker-a"
    assert task.current_run_id is not None
    claimed_events = [event for event in events if event.kind == "claimed"]
    assert len(claimed_events) == 1
    assert claimed_events[0].payload["lock"] == "worker-a"
    assert claimed_events[0].run_id == task.current_run_id


def test_store_contract_heartbeat_comment_and_complete_close_run(store):
    with store.connect() as conn:
        task_id = store.create_task(conn, title="finish me", assignee="alice")
        claimed = store.claim_task(conn, task_id, claimer="worker-a")
        assert claimed is not None

        assert store.heartbeat_claim(conn, task_id, claimer="worker-a") is True
        comment_id = store.add_comment(conn, task_id, "reviewer", "looks good")
        assert comment_id > 0
        assert store.complete_task(
            conn,
            task_id,
            result="ok",
            summary="completed by contract",
            expected_run_id=claimed.current_run_id,
        ) is True

        task = store.get_task(conn, task_id)
        comments = store.list_comments(conn, task_id)
        events = store.list_events(conn, task_id)

    assert task.status == "done"
    assert task.result == "ok"
    assert task.claim_lock is None
    assert task.claim_expires is None
    assert comments[-1].author == "reviewer"
    assert comments[-1].body == "looks good"
    assert "commented" in [event.kind for event in events]
    assert "completed" in [event.kind for event in events]


def test_store_contract_expired_claim_reclaims_to_ready(store):
    with store.connect() as conn:
        task_id = store.create_task(conn, title="reclaim me", assignee="alice")
        claimed = store.claim_task(conn, task_id, ttl_seconds=1, claimer="remote-host:123")
        assert claimed is not None
        with store.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET claim_expires = 1 WHERE id = ?",
                (task_id,),
            )
            conn.execute(
                "UPDATE task_runs SET claim_expires = 1 WHERE id = ?",
                (claimed.current_run_id,),
            )

        reclaimed = store.release_stale_claims(conn)
        task = store.get_task(conn, task_id)
        events = store.list_events(conn, task_id)

    assert reclaimed == 1
    assert task.status == "ready"
    assert task.claim_lock is None
    assert task.claim_expires is None
    reclaimed_events = [event for event in events if event.kind == "reclaimed"]
    assert len(reclaimed_events) == 1
    assert reclaimed_events[0].payload["stale_lock"] == "remote-host:123"
