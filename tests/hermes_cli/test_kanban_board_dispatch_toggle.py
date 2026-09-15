"""F5: enforce board.json's ``dispatch_enabled`` at the dispatch_once() choke point.

PR #11 added ``dispatch_enabled`` / ``auto_decompose_enabled`` to board.json plus
desktop UI toggles, but wired zero backend enforcement. This is a narrowing-only
toggle: default is enabled/True so old boards (or boards created before this
field existed) with no ``dispatch_enabled`` key in board.json keep dispatching
exactly as before. Only an explicit ``False`` in board.json must stop new spawns
on that board's next tick — it must never retroactively kill an already-running
task (intake gate only, not an execution kill switch).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _fake_spawn_factory(spawns: list):
    def fake_spawn(task, workspace, board=None):
        spawns.append((task.id, board))
        return 42
    return fake_spawn


def test_disabled_board_ready_task_stays_unclaimed(kanban_home, all_assignees_spawnable):
    """One board flagged dispatch_enabled=False, one board left at the default:
    the disabled board's ready task must not spawn; the other board's must."""
    kb.create_board("blocked")
    kb.write_board_metadata("blocked", dispatch_enabled=False)
    kb.create_board("open")  # no dispatch_enabled key at all — default True

    spawns: list = []
    with kbc.connect(board="blocked") as conn:
        blocked_id = kb.create_task(conn, title="stuck", assignee="alice")
        res_blocked = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn_factory(spawns), board="blocked",
        )
    with kbc.connect(board="open") as conn:
        open_id = kb.create_task(conn, title="goes", assignee="alice")
        res_open = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn_factory(spawns), board="open",
        )

    assert not res_blocked.spawned
    assert res_blocked.board_dispatch_disabled is True
    assert not any(tid == blocked_id for tid, _b in spawns)

    assert len(res_open.spawned) == 1
    assert res_open.board_dispatch_disabled is False
    assert any(tid == open_id for tid, _b in spawns)

    # The blocked task must still be sitting in 'ready', unclaimed.
    with kbc.connect(board="blocked") as conn:
        row = conn.execute("SELECT status, claim_lock FROM tasks WHERE id = ?", (blocked_id,)).fetchone()
        assert row["status"] == "ready"
        assert row["claim_lock"] is None


def test_missing_board_json_field_keeps_dispatching_by_default(kanban_home, all_assignees_spawnable):
    """No board.json / no dispatch_enabled key at all -> old boards keep working."""
    spawns: list = []
    with kbc.connect() as conn:  # default board — never had write_board_metadata called
        tid = kb.create_task(conn, title="old-board-task", assignee="alice")
        res = kbd.dispatch_once(conn, spawn_fn=_fake_spawn_factory(spawns))

    assert len(res.spawned) == 1
    assert res.board_dispatch_disabled is False
    assert any(t == tid for t, _b in spawns)


def test_flag_flipped_on_reenables_on_next_tick(kanban_home, all_assignees_spawnable):
    """The flag is read fresh every tick, not cached across ticks."""
    kb.create_board("flippy")
    kb.write_board_metadata("flippy", dispatch_enabled=False)

    spawns: list = []
    with kbc.connect(board="flippy") as conn:
        tid = kb.create_task(conn, title="queued", assignee="alice")
        res1 = kbd.dispatch_once(conn, spawn_fn=_fake_spawn_factory(spawns), board="flippy")
    assert not res1.spawned
    assert res1.board_dispatch_disabled is True
    assert not spawns

    # Flip the flag on; the SAME already-queued task must spawn on the NEXT tick.
    kb.write_board_metadata("flippy", dispatch_enabled=True)
    with kbc.connect(board="flippy") as conn:
        res2 = kbd.dispatch_once(conn, spawn_fn=_fake_spawn_factory(spawns), board="flippy")
    assert len(res2.spawned) == 1
    assert res2.board_dispatch_disabled is False
    assert any(t == tid for t, _b in spawns)


def test_disabling_mid_flight_does_not_kill_running_task(kanban_home, all_assignees_spawnable):
    """Intake gate only: an already-running task on a board must survive the
    board being disabled afterwards — dispatch_once() must not touch it."""
    kb.create_board("midflight")
    with kbc.connect(board="midflight") as conn:
        tid = kb.create_task(conn, title="already-running", assignee="alice")
        assert kb.claim_task(conn, tid) is not None
        row = conn.execute("SELECT status FROM tasks WHERE id = ?", (tid,)).fetchone()
        assert row["status"] == "running"

    # Disable the board AFTER the task started running.
    kb.write_board_metadata("midflight", dispatch_enabled=False)

    spawns: list = []
    with kbc.connect(board="midflight") as conn:
        res = kbd.dispatch_once(conn, spawn_fn=_fake_spawn_factory(spawns), board="midflight")

    assert res.board_dispatch_disabled is True
    with kbc.connect(board="midflight") as conn:
        row = conn.execute("SELECT status FROM tasks WHERE id = ?", (tid,)).fetchone()
        # Still running — the intake gate never touched an in-flight worker.
        assert row["status"] == "running"
