"""Configured Kanban spawn guards are a fail-closed admission boundary."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def test_configured_spawn_guard_wraps_custom_dispatch_spawn(monkeypatch):
    calls = []
    task = object()

    def native(*args, **kwargs):
        calls.append((args, kwargs))
        return 4242

    def guard(task, workspace, board, native_spawn):
        assert task is not None
        return native_spawn(task, workspace, board=board)

    monkeypatch.setattr(kb, "_load_configured_spawn_guard", lambda: guard)
    assert kb._spawn_with_guard(task, "/tmp/workspace", "board", native) == 4242
    assert len(calls) == 1


def test_configured_spawn_guard_failure_never_calls_native_spawn(monkeypatch):
    called = []

    def native(*args, **kwargs):
        called.append(True)
        return 4242

    def guard(*args, **kwargs):
        raise RuntimeError("governor unavailable")

    monkeypatch.setattr(kb, "_load_configured_spawn_guard", lambda: guard)
    with pytest.raises(kb.SpawnAdmissionError, match="governor unavailable"):
        kb._spawn_with_guard(object(), "/tmp/workspace", "board", native)
    assert called == []


def test_guard_deferral_releases_claim_without_creating_a_phantom_worker(
    kanban_home, all_assignees_spawnable, monkeypatch,
):
    """Healthy policy denial returns a task to ready without failure accounting."""
    native_spawn = Mock(return_value=4242)

    def guard(*_args, **_kwargs):
        raise kb.SpawnAdmissionDeferred("provider reserve policy")

    monkeypatch.setattr(kb, "_load_configured_spawn_guard", lambda: guard)
    conn = kb.connect()
    try:
        task_id = kb.create_task(conn, title="defer safely", assignee="alice")
        result = kb.dispatch_once(conn, spawn_fn=native_spawn)
        task = kb.get_task(conn, task_id)
        run = kb.latest_run(conn, task_id)
    finally:
        conn.close()

    native_spawn.assert_not_called()
    assert result.spawned == []
    assert task is not None
    assert task.status == "ready"
    assert task.claim_lock is None
    assert task.worker_pid is None
    assert task.current_run_id is None
    assert task.consecutive_failures == 0
    assert run is not None
    assert run.outcome == "deferred"
    assert run.error == "provider reserve policy"
