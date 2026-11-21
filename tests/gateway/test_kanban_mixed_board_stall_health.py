"""End-to-end (real sqlite boards, real dispatcher tick) regression test for
the #DRE-292 mixed-board health-telemetry gap.

``test_kanban_dispatch_capacity_health.py`` already proves the pure helper
``any_genuine_stall`` is correct against constructed ``DispatchResult``
objects. That is necessary but not sufficient: the embedded gateway loop
(``gateway/kanban_watchers.py``) wires real per-board ``tick_once()``
results and ``board_ready_flags()`` into it. This test drives the actual
``_KanbanDispatcher`` against two real boards sharing one tick — one at its
own ``max_spawn`` cap (healthy load), one with real ready work whose spawn
genuinely fails (a stand-in for a broken PATH/venv/credentials) — and
asserts the combination the embedded loop would see is still flagged as a
genuine stall, exactly as the pure-function test predicts.
"""
import os
import time

from gateway.kanban_watchers_dispatcher import _KanbanDispatcher, _resolve_dispatcher_settings
from hermes_cli import kanban_db as kb, kanban_db_connect as kbc, kanban_db_dispatch as kbd
from hermes_cli.config import atomic_config_write, load_config_readonly


def _spawn_fn(task, workspace, board=None):
    """Board 'full-board' spawns fine; board 'broken-board' fails every time
    (stand-in for a broken PATH/venv/credentials spawn-time failure)."""
    if board == "broken-board":
        raise RuntimeError("simulated broken spawn environment")
    return None


def test_mixed_full_and_broken_board_stall_is_not_hidden(tmp_path, monkeypatch):
    from hermes_cli import profiles
    monkeypatch.setattr(profiles, "profile_exists", lambda name: True)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = tmp_path / "config.yaml"
    # max_spawn=1 is a PER-BOARD cap (``count_running_tasks(conn)`` only
    # counts the connected board's own rows), unlike ``max_in_progress``
    # which is host-wide across every board. "full-board" is saturated by
    # its own pre-existing running task while "broken-board" still has
    # free per-board budget (host-wide total stays well under any cap).
    atomic_config_write(config, {"kanban": {"max_spawn": 1}})

    kb.create_board("full-board")
    kb.create_board("broken-board")

    with kbc.connect(board="full-board") as conn:
        running_id = kb.create_task(conn, title="already running", assignee="alice")
        kb.create_task(conn, title="queued behind the cap", assignee="alice")
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status = 'running', claim_lock = 'test-claim', "
                "claim_expires = ?, worker_pid = ?, worker_started_at = ? WHERE id = ?",
                (int(time.time()) + 3600, os.getpid(), int(time.time()), running_id),
            )
        assert kbd.count_running_tasks(conn) == 1

    with kbc.connect(board="broken-board") as conn:
        kb.create_task(conn, title="would spawn but the env is broken", assignee="bob")

    monkeypatch.setattr(kbd, "_default_spawn", _spawn_fn)

    settings = _resolve_dispatcher_settings(load_config_readonly()["kanban"], kb)
    dispatcher = _KanbanDispatcher(kb, settings)

    results = dispatcher.tick_once()
    board_ready_flags = dispatcher.board_ready_flags()
    by_slug = dict(results)

    # Sanity on the per-board facts this test is built to exercise.
    assert by_slug["full-board"].capacity_full is True
    assert by_slug["full-board"].spawned == []
    assert board_ready_flags["full-board"] is True  # still has a queued, unspawned task

    assert by_slug["broken-board"].capacity_full is False
    assert by_slug["broken-board"].spawned == []
    assert board_ready_flags["broken-board"] is True

    # The pre-fix bug: any_capacity_full() folds every board into one any(),
    # so the saturated full-board would have wrongly excused this tick.
    assert kbd.any_capacity_full(res for _slug, res in results) is True

    # The fix under review: judged per board, "broken-board"'s unexplained
    # zero-spawn must still be visible even though "full-board" is at cap.
    stuck = kbd.any_genuine_stall(
        (board_ready_flags.get(slug, False), res) for slug, res in results
    )
    assert stuck is True
