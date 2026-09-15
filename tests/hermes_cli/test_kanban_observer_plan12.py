from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli.plugins import VALID_HOOKS, get_plugin_manager


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def hooks():
    manager = get_plugin_manager()
    saved = {key: list(value) for key, value in manager._hooks.items()}
    events: list[tuple[str, dict]] = []
    for name in (
        "on_kanban_task_created",
        "kanban_task_completed",
        "kanban_task_blocked",
        "on_kanban_worker_exited",
    ):
        manager._hooks.setdefault(name, []).append(
            lambda _name=name, **kwargs: events.append((_name, kwargs))
        )
    try:
        yield events
    finally:
        manager._hooks = saved


def test_create_task_fires_created_observer_after_commit(kanban_home, hooks):
    conn = kbc.connect()
    try:
        task_id = kb.create_task(
            conn, title="created", assignee="worker", project_id="project-1",
        )
        assert conn.execute("SELECT id FROM tasks WHERE id = ?", (task_id,)).fetchone()
    finally:
        conn.close()

    created = [payload for name, payload in hooks if name == "on_kanban_task_created"]
    assert len(created) == 1
    assert created[0]["task_id"] == task_id
    assert created[0]["assignee"] == "worker"
    assert "project_id" in created[0]
    assert created[0]["workspace_kind"] == "scratch"
    assert created[0]["run_id"] is None




def test_plan12_hook_names_are_registered():
    assert {
        "kanban_task_claimed",
        "kanban_task_completed",
        "kanban_task_blocked",
        "on_kanban_task_created",
        "on_kanban_worker_spawned",
        "on_kanban_worker_exited",
        "on_kanban_worker_stale_claim",
        "on_kanban_task_updated",
        "on_kanban_dispatch_tick",
    } <= VALID_HOOKS


def test_lifecycle_observers_fire_in_creation_claim_spawn_completion_order(
    kanban_home, hooks, all_assignees_spawnable,
):
    manager = get_plugin_manager()
    manager._hooks.setdefault("kanban_task_claimed", []).append(
        lambda **_kwargs: hooks.append(("kanban_task_claimed", _kwargs))
    )
    manager._hooks.setdefault("on_kanban_worker_spawned", []).append(
        lambda **_kwargs: hooks.append(("on_kanban_worker_spawned", _kwargs))
    )
    conn = kbc.connect()
    try:
        task_id = kb.create_task(conn, title="ordered", assignee="worker")
        result = kbd.dispatch_once(conn, spawn_fn=lambda *_args, **_kwargs: 4242)
        assert any(row[0] == task_id for row in result.spawned)
        assert kb.complete_task(conn, task_id, summary="done")
    finally:
        conn.close()

    names = [name for name, payload in hooks if payload.get("task_id") == task_id]
    assert names[:4] == [
        "on_kanban_task_created",
        "kanban_task_claimed",
        "on_kanban_worker_spawned",
        "kanban_task_completed",
    ]


def test_archive_task_reports_archived_completion(kanban_home, hooks):
    conn = kbc.connect()
    try:
        task_id = kb.create_task(conn, title="archive", assignee="worker")
        assert kb.archive_task(conn, task_id)
    finally:
        conn.close()

    completed = [payload for name, payload in hooks if name == "kanban_task_completed"]
    assert len(completed) == 1
    assert completed[0]["task_id"] == task_id
    assert completed[0]["outcome"] == "archived"



def test_scope_block_kind_routes_through_existing_block_observer(kanban_home, hooks):
    conn = kbc.connect()
    try:
        task_id = kb.create_task(conn, title="scope", assignee="worker")
        assert kb.block_task(conn, task_id, kind="scope", reason="needs /outside")
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.block_kind == "scope"
    finally:
        conn.close()

    blocked = [payload for name, payload in hooks if name == "kanban_task_blocked"]
    assert len(blocked) == 1
    assert blocked[0]["reason"] == "needs /outside"


def test_timeout_reports_worker_exit_observer(kanban_home, hooks, monkeypatch):
    conn = kbc.connect()
    try:
        task_id = kb.create_task(
            conn, title="timeout", assignee="worker", max_runtime_seconds=1,
        )
        assert kb.claim_task(conn, task_id)
        conn.execute(
            "UPDATE task_runs SET started_at = ? WHERE id = (SELECT current_run_id FROM tasks WHERE id = ?)",
            (1, task_id),
        )
        conn.execute(
            "UPDATE tasks SET worker_pid = ?, worker_started_at = ?, claim_lock = ? WHERE id = ?",
            (12345, 1, kb._host_prefix() + "worker", task_id),
        )
        conn.commit()
        monkeypatch.setattr(kbd.time, "time", lambda: 100)
        monkeypatch.setattr(kbd, "_pid_recycled", lambda *_args: False)
        monkeypatch.setattr(kbd, "_worker_alive", lambda *_args: False)
        monkeypatch.setattr(kb, "_pid_alive", lambda *_args: False)
        assert task_id in kbd.enforce_max_runtime(conn, signal_fn=lambda *_args: None)
    finally:
        conn.close()

    exits = [payload for name, payload in hooks if name == "on_kanban_worker_exited"]
    assert len(exits) == 1
    assert exits[0]["task_id"] == task_id
    assert exits[0]["exit_kind"] == "timed_out"
    assert exits[0]["outcome"] == "timed_out"
