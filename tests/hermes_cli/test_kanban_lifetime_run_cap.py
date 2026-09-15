"""Regression coverage for the Kanban lifetime worker-spawn cap."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "default")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _fake_spawn(_task, _workspace):
    return 12345


def _set_total_runs(conn, task_id, total_runs):
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET total_runs = ? WHERE id = ?", (total_runs, task_id))


def test_existing_database_migrates_total_runs_to_zero(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    path = kb.kanban_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    legacy = sqlite3.connect(path)
    # A minimal pre-column tasks table: connect() must add total_runs without
    # relying on the current SCHEMA_SQL's comments or formatting.
    legacy.executescript(
        """
        CREATE TABLE tasks (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            body TEXT,
            assignee TEXT,
            status TEXT NOT NULL,
            priority INTEGER DEFAULT 0,
            created_by TEXT,
            created_at INTEGER NOT NULL,
            started_at INTEGER,
            completed_at INTEGER,
            workspace_kind TEXT NOT NULL DEFAULT 'scratch',
            workspace_path TEXT,
            claim_lock TEXT,
            claim_expires INTEGER
        );
        """
    )
    legacy.execute(
        "INSERT INTO tasks (id, title, status, created_at) VALUES ('old', 'old', 'ready', 1)"
    )
    legacy.commit()
    legacy.close()

    with kbc.connect(path) as conn:
        assert conn.execute("SELECT total_runs FROM tasks WHERE id = 'old'").fetchone()[0] == 0


def test_claim_increments_total_runs_once_and_lost_claim_does_not(kanban_home):
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="task", assignee="default")
        claimed = kb.claim_task(conn, task_id)
        assert claimed is not None
        assert claimed.total_runs == 1
        assert kb.claim_task(conn, task_id) is None
        assert conn.execute("SELECT total_runs FROM tasks WHERE id = ?", (task_id,)).fetchone()[0] == 1


def test_total_runs_survives_unblock_reassign_and_ancestor_reopen(kanban_home):
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="task", assignee="alpha")
        _set_total_runs(conn, task_id, 4)
        assert kb.block_task(conn, task_id, reason="retry")
        assert kb.unblock_task(conn, task_id)
        assert kb.assign_task(conn, task_id, "beta")
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.total_runs == 4

        parent = kb.create_task(conn, title="parent", assignee="default")
        child = kb.create_task(conn, title="child", assignee="default")
        assert kb.link_tasks(conn, parent, child)
        _set_total_runs(conn, child, 4)
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status = 'ready' WHERE id IN (?, ?)", (parent, child))
        kb.invalidate_descendants_for_parent_reopen(conn, parent, author="test")
        reopened_child = kb.get_task(conn, child)
        assert reopened_child is not None
        assert reopened_child.total_runs == 4


def test_repeated_unblocks_cannot_bypass_lifetime_cap(kanban_home, all_assignees_spawnable):
    limit = 3
    spawns = []

    def spawn(task, _workspace):
        spawns.append(task.id)
        return 12345

    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="loop", assignee="default")
        for attempt in range(limit):
            result = kbd.dispatch_once(conn, spawn_fn=spawn, lifetime_run_limit=limit)
            assert [entry[0] for entry in result.spawned] == [task_id]
            task = kb.get_task(conn, task_id)
            assert task is not None
            assert task.total_runs == attempt + 1

            # Simulate the incident's worker block -> external unblock/re-spec
            # loop through public transitions. Alternate kinds so the separate
            # block-recurrence breaker does not mask the lifetime-cap invariant.
            assert kb.block_task(
                conn, task_id, reason="retry", kind=("transient" if attempt % 2 else "needs_input"),
            )
            assert kb.unblock_task(conn, task_id)

        result = kbd.dispatch_once(conn, spawn_fn=spawn, lifetime_run_limit=limit)
        assert result.spawned == []
        assert result.run_capped == [task_id]
        assert spawns == [task_id] * limit
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == "triage"
        event = next(event for event in kb.list_events(conn, task_id) if event.kind == "lifetime_run_capped")
        assert event.payload is not None
        assert event.payload["total_runs"] == limit
        assert event.payload["limit"] == limit


def test_zero_lifetime_limit_allows_spawn(kanban_home, all_assignees_spawnable):
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="legacy", assignee="default")
        _set_total_runs(conn, task_id, 1000)
        result = kbd.dispatch_once(conn, spawn_fn=_fake_spawn, lifetime_run_limit=0)
        assert result.run_capped == []
        assert [entry[0] for entry in result.spawned] == [task_id]


def test_gateway_dispatch_settings_preserve_zero_disable_value():
    from gateway.kanban_watchers_dispatcher import _resolve_dispatcher_settings

    settings = _resolve_dispatcher_settings({"lifetime_run_limit": 0}, kbd)
    assert settings.lifetime_run_limit == 0


def test_cap_warning_names_task_and_total_runs(kanban_home, all_assignees_spawnable, caplog):
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="warn", assignee="default")
        _set_total_runs(conn, task_id, 3)
        with caplog.at_level("WARNING"):
            kbd.dispatch_once(conn, spawn_fn=_fake_spawn, lifetime_run_limit=3)
    assert any(task_id in record.message and "3" in record.message for record in caplog.records)
