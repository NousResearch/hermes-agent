"""Gateway auto-decompose must leave needs_input escalations untouched."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gateway.run import GatewayRunner
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_decompose as decomp


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    return home


def _escalate_needs_input(conn, task_id: str) -> None:
    assert kb.block_task(conn, task_id, reason="answer required", kind="needs_input")
    assert kb.unblock_task(conn, task_id)
    assert kb.block_task(conn, task_id, reason="answer required", kind="needs_input")


def test_gateway_tick_does_not_redispatch_escalated_needs_input(
    kanban_home: Path,
    tmp_path: Path,
) -> None:
    with kb.connect_closing() as conn:
        task_id = kb.create_task(
            conn,
            title="wait for operator",
            assignee="worker",
            workspace_kind="worktree",
            workspace_path=str(tmp_path / "repo" / "checkout"),
            branch_name="feature/wait-for-operator",
        )
        _escalate_needs_input(conn, task_id)
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == "triage"

    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running = True
    runner._kanban_dispatcher_lock_handle = None
    sleep_calls: list[float] = []
    real_sleep = asyncio.sleep

    async def fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)
        if len(sleep_calls) >= 2:
            runner._running = False
        await real_sleep(0)

    async def fake_to_thread(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    config = {
        "kanban": {
            "dispatch_in_gateway": True,
            "auto_decompose": True,
            "auto_decompose_per_tick": 3,
            "dispatch_interval_seconds": 1,
            "reconcile_orphans": False,
        }
    }
    spawn = MagicMock(name="spawn")
    decompose = MagicMock(name="decompose")

    with (
        patch("hermes_cli.config.load_config", return_value=config),
        patch(
            "gateway.kanban_watchers._acquire_singleton_lock",
            return_value=(object(), "held"),
        ),
        patch("gateway.kanban_watchers._release_singleton_lock"),
        patch(
            "gateway.kanban_watchers._kanban_dispatch_allowed",
            return_value=True,
        ),
        patch.object(kb, "_default_spawn", spawn),
        patch.object(decomp, "decompose_task", decompose),
        patch("asyncio.sleep", side_effect=fake_sleep),
        patch("asyncio.to_thread", side_effect=fake_to_thread),
    ):
        asyncio.run(runner._kanban_dispatcher_watcher())

    decompose.assert_not_called()
    spawn.assert_not_called()
    with kb.connect_closing() as conn:
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == "triage"
        assert kb.child_ids(conn, task_id) == []
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 1
