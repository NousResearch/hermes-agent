"""Invariant: dispatch never starts a new run while a prior worker is alive.

A closed run (review_requested / timed_out) keeps its worker_pid on the
task_runs row, but tasks.worker_pid is cleared so the next tick can claim
the card again. reap_terminal_workers leaves that pid alone for
TERMINAL_WORKER_REAP_GRACE_SECONDS. Without a spawn-time liveness check,
two workers for one card write the same directory — the dir-workspace
collision observed on t_52986b30.

These tests must fail on origin/main (RED) and pass once dispatch defers
while a closed-run pid is still alive.
"""

from __future__ import annotations

import subprocess
import sys
import time
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
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _sleeper():
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(120)"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(0.2)
    return proc


def _dir_card(conn, tmp_path, *, assignee="builder"):
    ws = tmp_path / "shared-dir"
    ws.mkdir()
    tid = kb.create_task(
        conn,
        title="dir card",
        assignee=assignee,
        workspace_kind="dir",
        workspace_path=str(ws),
    )
    return tid, ws


def test_review_handoff_does_not_spawn_alongside_live_implementer(
    kanban_home, tmp_path, all_assignees_spawnable,
):
    """request_review closes the run; the implementer pid can still be alive.

    The next dispatch tick must not start a reviewer (or anyone) for that
    card until the prior process is gone. Two live workers sharing a dir
    workspace is the defect.
    """
    proc = _sleeper()
    spawned: list[int] = []

    def spawn(task, workspace):
        spawned.append(int(task.worker_pid or 0))
        return 4242

    try:
        with kbc.connect() as conn:
            tid, _ws = _dir_card(conn, tmp_path, assignee="builder")
            claimed = kb.claim_task(conn, tid, claimer=kb._claimer_id())
            assert claimed is not None
            kbd._set_worker_pid(conn, tid, proc.pid)
            run_id = kb._current_run_id(conn, tid)
            assert kb.request_review(
                conn, tid,
                summary="ready for review",
                reviewer="reviewer",
                expected_run_id=run_id,
            ) is True
            task = kb.get_task(conn, tid)
            assert task is not None and task.status == "review"
            assert proc.poll() is None

            result = kbd.dispatch_once(conn, spawn_fn=spawn)

            assert proc.poll() is None, "prior worker must still be alive at the spawn decision"
            assert result.spawned == [], (
                f"spawned a second worker for {tid} while pid {proc.pid} was still alive: "
                f"{result.spawned}"
            )
            assert spawned == []
            assert (tid, "prior_worker_alive") in result.respawn_guarded
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def test_timed_out_card_does_not_respawn_while_worker_still_alive(
    kanban_home, tmp_path, all_assignees_spawnable,
):
    """A timed_out run requeues the card to ready while its worker may still
    be dying. The next tick must not spawn a sibling into the same dir."""
    proc = _sleeper()
    spawned_ids: list[str] = []

    def spawn(task, workspace):
        spawned_ids.append(task.id)
        return 4343

    try:
        with kbc.connect() as conn:
            tid, _ws = _dir_card(conn, tmp_path, assignee="builder")
            claimed = kb.claim_task(conn, tid, claimer=kb._claimer_id())
            assert claimed is not None
            kbd._set_worker_pid(conn, tid, proc.pid)
            run_id = kb._current_run_id(conn, tid)
            with kb.write_txn(conn):
                conn.execute(
                    "UPDATE tasks SET status = 'ready', claim_lock = NULL, "
                    "claim_expires = NULL, worker_pid = NULL, "
                    "worker_started_at = NULL, last_heartbeat_at = NULL "
                    "WHERE id = ?",
                    (tid,),
                )
                kb._end_run(conn, tid, outcome="timed_out", status="timed_out")
            assert kb._current_run_id(conn, tid) is None
            run = conn.execute(
                "SELECT worker_pid FROM task_runs WHERE id = ?", (run_id,),
            ).fetchone()
            assert run["worker_pid"] == proc.pid
            assert proc.poll() is None

            result = kbd.dispatch_once(conn, spawn_fn=spawn)

            assert proc.poll() is None
            assert tid not in spawned_ids
            assert result.spawned == []
            assert (tid, "prior_worker_alive") in result.respawn_guarded

            proc.kill()
            proc.wait()
            after = kbd.dispatch_once(conn, spawn_fn=spawn)
            assert tid in [t for t, _a, _w in after.spawned]
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
