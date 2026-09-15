"""Board-level automation hold regressions for Kanban dispatch safety."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_WORKTREE = Path(__file__).resolve().parents[2]
if str(_WORKTREE) not in sys.path:
    sys.path.insert(0, str(_WORKTREE))

from gateway.kanban_watchers_dispatcher import _DispatcherSettings, _KanbanDispatcher
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def fresh_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    for var in (
        "HERMES_KANBAN_DB",
        "HERMES_KANBAN_WORKSPACES_ROOT",
        "HERMES_KANBAN_HOME",
        "HERMES_KANBAN_BOARD",
    ):
        monkeypatch.delenv(var, raising=False)
    try:
        import hermes_constants
        hermes_constants._cached_default_hermes_root = None  # type: ignore[attr-defined]
    except Exception:
        pass
    kb._INITIALIZED_PATHS.clear()
    return home


def test_board_automation_hold_metadata_round_trip(fresh_home):
    kb.create_board("held", name="Held Board", description="before")

    meta = kb.set_board_automation_hold("held", reason="SOQ hold", set_by="test")

    hold = meta["automation_hold"]
    assert hold["enabled"] is True
    assert hold["reason"] == "SOQ hold"
    assert hold["set_by"] == "test"
    assert hold["allow_reclaim"] is False
    assert isinstance(hold["set_at"], int)
    reread = kb.read_board_metadata("held")
    assert reread["name"] == "Held Board"
    assert reread["description"] == "before"
    assert kb.board_automation_held("held") is True

    kb.clear_board_automation_hold("held")

    assert kb.read_board_metadata("held")["automation_hold"] is None
    assert kb.board_automation_held("held") is False


def test_dispatch_once_skips_held_board_before_claim_or_spawn(fresh_home, monkeypatch):
    kb.create_board("held")
    kb.set_board_automation_hold("held", reason="safety", set_by="test")
    monkeypatch.setattr(kbd, "_profile_exists_fn", lambda: (lambda _name: True))
    calls = []

    with kbc.connect(board="held") as conn:
        tid = kb.create_task(conn, title="must not spawn", assignee="default")
        res = kbd.dispatch_once(
            conn,
            board="held",
            spawn_fn=lambda task, workspace, board=None: calls.append(task.id) or 123,
        )
        task = kb.get_task(conn, tid)
        events = [
            row["kind"]
            for row in conn.execute(
                "SELECT kind FROM task_events WHERE task_id = ? ORDER BY id", (tid,)
            )
        ]

    assert res.skipped_board_hold is True
    assert "safety" in (res.skipped_board_hold_reason or "")
    assert calls == []
    assert task.status == "ready"
    assert "claimed" not in events
    assert "spawned" not in events


def test_held_board_does_not_recompute_ready_or_promote_todo_tasks(fresh_home, monkeypatch):
    kb.create_board("held")
    kb.set_board_automation_hold("held", reason="freeze", set_by="test")
    monkeypatch.setattr(kbd, "_profile_exists_fn", lambda: (lambda _name: True))

    with kbc.connect(board="held") as conn:
        tid = kb.create_task(conn, title="would be promoted", assignee="default")
        conn.execute("UPDATE tasks SET status = 'todo' WHERE id = ?", (tid,))
        conn.commit()
        res = kbd.dispatch_once(conn, board="held", spawn_fn=lambda *_args, **_kw: 123)
        task = kb.get_task(conn, tid)
        promoted = conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id = ? AND kind = 'promoted'",
            (tid,),
        ).fetchone()[0]

    assert res.skipped_board_hold is True
    assert task.status == "todo"
    assert promoted == 0


def test_ready_nonempty_ignores_held_boards_for_health_telemetry(fresh_home, monkeypatch):
    kb.create_board("held")
    kb.set_board_automation_hold("held", reason="freeze", set_by="test")
    monkeypatch.setattr(kbd, "_profile_exists_fn", lambda: (lambda _name: True))
    with kbc.connect(board="held") as conn:
        kb.create_task(conn, title="ready but held", assignee="default")

    settings = _DispatcherSettings(
        interval=1.0,
        max_spawn=None,
        max_in_progress=None,
        failure_limit=2,
        stale_timeout_seconds=0,
        reconcile_orphans=True,
        default_assignee=None,
        max_in_progress_per_profile=None,
    )
    dispatcher = _KanbanDispatcher(kb, settings)

    assert dispatcher.ready_nonempty() is False


def test_auto_decompose_tick_skips_held_board(fresh_home, monkeypatch):
    kb.create_board("held")
    kb.set_board_automation_hold("held", reason="freeze", set_by="test")
    with kbc.connect(board="held") as conn:
        tid = kb.create_task(conn, title="triage", assignee="default")
        conn.execute("UPDATE tasks SET status = 'triage' WHERE id = ?", (tid,))
        conn.commit()

    calls = []

    def fail_if_called(*_args, **_kwargs):
        calls.append(True)
        raise AssertionError("held board should not be decomposed")

    from hermes_cli import kanban_decompose
    monkeypatch.setattr(kanban_decompose, "decompose_task", fail_if_called)

    settings = _DispatcherSettings(
        interval=1.0,
        max_spawn=None,
        max_in_progress=None,
        failure_limit=2,
        stale_timeout_seconds=0,
        reconcile_orphans=True,
        default_assignee=None,
        max_in_progress_per_profile=None,
    )
    dispatcher = _KanbanDispatcher(kb, settings)

    assert dispatcher.auto_decompose_tick(10) == 0
    assert calls == []
    with kbc.connect(board="held") as conn:
        assert kb.get_task(conn, tid).status == "triage"
