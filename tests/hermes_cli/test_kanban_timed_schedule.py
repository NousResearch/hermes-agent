"""Timed follow-ups share the CLI, worker handoff and dispatcher contracts (#154911)."""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from threading import Barrier, Event

import pytest

from hermes_cli import kanban as cli
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli.kanban_parser import build_parser
from tools import kanban_tools  # Register the real tool handlers.
from tools.registry import registry


@pytest.fixture
def board(tmp_path, monkeypatch, all_assignees_spawnable):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return kb.kanban_db_path()


def _schedule_cli(task_id, *args):
    parser = argparse.ArgumentParser()
    build_parser(parser.add_subparsers())
    return cli.kanban_command(parser.parse_args(["kanban", "schedule", task_id, *args]))


def test_worker_schedule_retimes_durably_and_dispatches_once(board, monkeypatch):
    """A real worker handoff closes its run; the last CLI schedule survives reopen."""
    at = "2040-01-01T09:00:00.500000+09:00"
    due = datetime.fromisoformat(at).timestamp()
    with kbc.connect_closing(board) as conn:
        tid = kb.create_task(conn, title="follow up", assignee="worker")
        claimed = kb.claim_task(conn, tid)
        with monkeypatch.context() as worker:
            worker.setenv("HERMES_KANBAN_TASK", tid)
            worker.setenv("HERMES_KANBAN_RUN_ID", str(claimed.current_run_id))
            out = json.loads(registry.get_entry("kanban_schedule").handler({"at": at, "reason": "check again"}))
        assert out["ok"] is True, out
        assert out["status"] == "scheduled"
        run = kb.latest_run(conn, tid)
        assert (run.outcome, run.summary) == ("scheduled", "check again")
        assert run.ended_at is not None
        assert kb.get_task(conn, tid).current_run_id is None
        # A stale worker cannot replace the new schedule after its run ended.
        assert not kb.schedule_task(conn, tid, at=at, expected_run_id=claimed.current_run_id)
    assert _schedule_cli(tid, "--at", "2040-01-02T00:00:00.500000Z") == 0
    monkeypatch.setattr(kbd.time, "time", lambda: due)
    spawned = []
    with kbc.connect_closing(board) as conn:
        assert kbd.dispatch_once(conn, spawn_fn=lambda t, _ws: spawned.append(t.id)).spawned == []
        assert kb.get_task(conn, tid).status == "scheduled"
        monkeypatch.setattr(kbd.time, "time", lambda: due + 86400 - 0.001)
        assert kbd.dispatch_once(conn, spawn_fn=lambda t, _ws: spawned.append(t.id)).spawned == []
        assert kb.get_task(conn, tid).status == "scheduled"
    monkeypatch.setattr(kbd.time, "time", lambda: due + 86400)
    # Independent connections racing the public dispatcher, as on a restart.
    barrier = Barrier(2)

    def tick():
        with kbc.connect_closing(board) as conn:
            barrier.wait(timeout=10)
            return kbd.dispatch_once(conn, spawn_fn=lambda t, _ws: spawned.append(t.id))

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: tick(), range(2)))
    assert sum(len(r.spawned) for r in results) == 1
    assert spawned == [tid]
    with kbc.connect_closing(board) as conn:
        assert kbd.dispatch_once(conn, spawn_fn=lambda t, _ws: spawned.append(t.id)).spawned == []
        assert kb.get_task(conn, tid).scheduled_at is None
        assert len([e for e in kb.list_events(conn, tid) if e.kind == "schedule_elapsed"]) == 1


@pytest.mark.parametrize("at", ["tomorrow", "2040-01-01", "2040-01-01T12:00:00", ""])
def test_invalid_time_changes_nothing_on_either_surface(board, at):
    with kbc.connect_closing(board) as conn:
        tid = kb.create_task(conn, title="untouched", assignee="worker")
        before = kb.list_events(conn, tid)
        out = json.loads(registry.get_entry("kanban_schedule").handler({"task_id": tid, "at": at}))
        assert "error" in out
        assert _schedule_cli(tid, "--at", at) != 0
        assert kb.get_task(conn, tid).status == "ready"
        assert kb.list_events(conn, tid) == before


def test_due_schedule_keeps_dependency_capacity_and_legacy_gates(board, monkeypatch):
    due = "2040-01-01T00:00:00Z"
    monkeypatch.setattr(kbd.time, "time", lambda: datetime.fromisoformat(due).timestamp())
    with kbc.connect_closing(board) as conn:
        parent = kb.create_task(conn, title="dependency", triage=True)
        child = kb.create_task(conn, title="dependent follow-up", assignee="worker", parents=[parent])
        legacy = kb.create_task(conn, title="manual wake", assignee="worker")
        assert kb.schedule_task(conn, child, at=due)
        assert kb.schedule_task(conn, legacy, reason=f"SCHEDULED_UNTIL={due}")
        kbd.dispatch_once(conn, dry_run=True)
        assert kb.get_task(conn, child).status == "scheduled"
        spawned = []
        kbd.dispatch_once(conn, max_spawn=0, spawn_fn=lambda t, _ws: spawned.append(t.id))
        assert kb.get_task(conn, child).status == "todo"
        assert kb.get_task(conn, legacy).status == "scheduled"
        assert not spawned
        kb.archive_task(conn, parent)
        kbd.dispatch_once(conn, max_spawn=0, spawn_fn=lambda t, _ws: spawned.append(t.id))
        assert kb.get_task(conn, child).status == "ready"
        assert not spawned
        kbd.dispatch_once(conn, max_spawn=1, spawn_fn=lambda t, _ws: spawned.append(t.id))
        assert spawned == [child]


def test_manual_parking_cancels_previous_timer(board, monkeypatch):
    with kbc.connect_closing(board) as conn:
        tid = kb.create_task(conn, title="rescheduled", assignee="worker")
        assert kb.schedule_task(conn, tid, at="2040-01-01T00:00:00Z")
        assert kb.schedule_task(conn, tid, reason="wait for operator")
        monkeypatch.setattr(kbd.time, "time", lambda: datetime.fromisoformat("2041-01-01T00:00:00Z").timestamp())
        kbd.dispatch_once(conn, max_spawn=0)
        assert kb.get_task(conn, tid).status == "scheduled"
        assert kb.get_task(conn, tid).scheduled_at is None
        assert kb.unblock_task(conn, tid)
        assert kb.get_task(conn, tid).status == "ready"


def test_legacy_board_migrates_without_turning_reason_into_timer(board):
    with kbc.connect_closing(board) as conn:
        tid = kb.create_task(conn, title="legacy parking", assignee="worker")
        assert kb.schedule_task(conn, tid, reason="SCHEDULED_UNTIL=2000-01-01T00:00:00Z")
        conn.execute("ALTER TABLE tasks DROP COLUMN scheduled_at")
    kb._INITIALIZED_PATHS.discard(str(board.resolve()))
    with kbc.connect_closing(board) as conn:
        kbd.dispatch_once(conn, max_spawn=0)
        task = kb.get_task(conn, tid)
        assert (task.status, task.scheduled_at) == ("scheduled", None)
        assert kb.schedule_task(conn, tid, at="2000-01-01T00:00:00Z")
        kbd.dispatch_once(conn, max_spawn=0)
        assert kb.get_task(conn, tid).status == "ready"


@pytest.mark.parametrize("allowed", ["[]", "[worker]"])
def test_due_schedule_respects_dispatch_profile_allowlist(board, allowed):
    (board.parent / "config.yaml").write_text(
        f"kanban:\n  dispatch_profiles: {allowed}\n", encoding="utf-8",
    )
    with kbc.connect_closing(board) as conn:
        tid = kb.create_task(conn, title="not a dispatch profile", assignee="chat-only")
        assert kb.schedule_task(conn, tid, at="2000-01-01T00:00:00Z")
        spawned = []
        result = kbd.dispatch_once(conn, spawn_fn=lambda t, _ws: spawned.append(t.id))
        assert not spawned
        assert result.skipped_nonspawnable == [tid]
        task = kb.get_task(conn, tid)
        assert (task.status, task.assignee, task.current_run_id) == ("ready", "chat-only", None)


def test_reschedule_wins_before_wake_transaction(board, monkeypatch):
    from hermes_cli.kanban_db_schedule import wake_due_scheduled

    with kbc.connect_closing(board) as conn:
        tid = kb.create_task(conn, title="racing deadline", assignee="worker")
        assert kb.schedule_task(conn, tid, at="2000-01-01T00:00:00Z")
    waiting, rescheduled = Event(), Event()
    real_txn = kb.write_txn

    def wake():
        with kbc.connect_closing(board) as conn:
            @contextmanager
            def pause_before_lock(c, **kw):
                if c is conn:
                    waiting.set()
                    assert rescheduled.wait(timeout=10)
                with real_txn(c, **kw) as txn:
                    yield txn

            monkeypatch.setattr(kb, "write_txn", pause_before_lock)
            wake_due_scheduled(conn)

    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(wake)
        try:
            assert waiting.wait(timeout=10)
            assert _schedule_cli(tid, "--at", "2099-01-01T00:00:00Z") == 0
        finally:
            rescheduled.set()
        pending.result(timeout=10)
    with kbc.connect_closing(board) as conn:
        assert kb.get_task(conn, tid).status == "scheduled"
        assert not any(e.kind == "schedule_elapsed" for e in kb.list_events(conn, tid))
