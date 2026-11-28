"""A worker parks its card on its own background process; the dispatcher reopens it.

Parks go through the registered ``kanban_schedule`` handler and wakes through
``dispatch_once``, against real background processes started the way a managed
worker starts them (an output log plus the shell trap that records the exit code).
"""

from __future__ import annotations

import json
import os
import signal
import time
from datetime import UTC, datetime

import pytest

from tests.waiting import wait_until

pytestmark = pytest.mark.platforms("posix")


@pytest.fixture
def worker_env(monkeypatch, tmp_path):
    """An isolated board with one claimed task, seen from that task's worker."""
    from pathlib import Path as _Path

    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "test-worker")
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    monkeypatch.setattr(_Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="wait-on-process", assignee="test-worker")
        kb.claim_task(conn, tid)
        run_id = kb._current_run_id(conn, tid)
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
    return tid


@pytest.fixture
def spawn(monkeypatch, tmp_path):
    """Start a background process as a managed worker does; kill leftovers at teardown."""
    from agent import runtime_session_store
    from tools.process_registry import process_registry

    monkeypatch.setattr(runtime_session_store, "_worker_process", True)
    started = []

    def _spawn(command: str):
        session = process_registry.spawn_local(command, cwd=str(tmp_path))
        started.append(session.id)
        return session

    yield _spawn
    for sid in started:
        process_registry.kill_process(sid, source="test")


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, UTC).isoformat()


def _park(session, *, until: float | None = None) -> dict:
    from tools import kanban_tools  # registers the kanban tools
    from tools.registry import registry

    args = {"reason": "waiting on a job", "wait_for_process": session.id,
            "until": _iso(until if until is not None else time.time() + 3600)}
    return json.loads(registry.get_entry("kanban_schedule").handler(args))


def _tick(spawned: list) -> None:
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd

    def record_spawn(task, workspace, board=None):
        spawned.append(task.id)

    with kbc.connect() as conn:
        kbd.dispatch_once(conn, spawn_fn=record_spawn)


def _status(tid: str) -> str:
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    with kbc.connect() as conn:
        return kb.get_task(conn, tid).status


def _events(tid: str, kind: str) -> list:
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    with kbc.connect() as conn:
        return [e for e in kb.list_events(conn, tid) if e.kind == kind]


def _comments(tid: str) -> list:
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    with kbc.connect() as conn:
        return kb.list_comments(conn, tid)


def _exit_recorded(session) -> bool:
    from tools.process_registry_output_log import recorded_exit_code

    return recorded_exit_code(session.output_log) is not None


@pytest.mark.parametrize("code", [0, 3])
def test_task_wakes_when_awaited_process_exits(worker_env, spawn, all_assignees_spawnable, code):
    session = spawn(f"sleep 2; exit {code}")
    assert _park(session)["status"] == "scheduled"
    wait_until(lambda: _exit_recorded(session), timeout=15, what="the exit file")
    spawned: list = []
    _tick(spawned)

    assert worker_env in spawned
    [comment] = [c for c in _comments(worker_env) if c.author == "dispatcher"]
    assert f"code {code}" in comment.body and session.id in comment.body
    [resolved] = _events(worker_env, "wait_resolved")
    assert (resolved.payload["outcome"], resolved.payload["exit_code"]) == ("exited", code)


def test_process_that_exits_during_the_sweep_is_not_reported_lost(worker_env, spawn, monkeypatch, tmp_path,
                                                                  all_assignees_spawnable):
    """A sweep whose exit-file read lands just before the job records its code must not call the job lost."""
    import tools.process_registry_output_log as output_log
    from hermes_cli import kanban_db as kb

    release = tmp_path / "release"
    session = spawn(f"until [ -e {release} ]; do sleep 0.1; done; exit 5")
    assert _park(session)["status"] == "scheduled"
    real_read, stale_reads = output_log.recorded_exit_code, []

    def read_then_let_the_job_finish(log):
        code = real_read(log)
        if not stale_reads and code is None:
            stale_reads.append(log)
            release.touch()
            wait_until(lambda: real_read(log) is not None and not kb._pid_alive(session.pid),
                       timeout=15, what="the job to record its exit code and go away")
        return code

    monkeypatch.setattr(output_log, "recorded_exit_code", read_then_let_the_job_finish)
    spawned: list = []
    _tick(spawned)
    _tick(spawned)

    assert stale_reads, "the sweep never read the exit file while the job was running"
    assert worker_env in spawned
    [resolved] = _events(worker_env, "wait_resolved")
    assert (resolved.payload["outcome"], resolved.payload["exit_code"]) == ("exited", 5)


def test_task_wakes_when_awaited_process_dies_without_exit_code(worker_env, spawn, all_assignees_spawnable):
    session = spawn("sleep 300")
    assert _park(session)["status"] == "scheduled"
    os.killpg(os.getpgid(session.pid), signal.SIGKILL)
    wait_until(lambda: session.exited, timeout=15, what="the killed process to be reaped")
    spawned: list = []
    _tick(spawned)

    assert worker_env in spawned
    [resolved] = _events(worker_env, "wait_resolved")
    assert (resolved.payload["outcome"], resolved.payload["exit_code"]) == ("lost", None)


def test_task_wakes_at_deadline_and_leaves_process_running(worker_env, spawn, all_assignees_spawnable):
    session = spawn("sleep 300")
    until = int(time.time()) + 2
    assert _park(session, until=until)["status"] == "scheduled"
    spawned: list = []
    _tick(spawned)
    assert _status(worker_env) == "scheduled" and spawned == []

    wait_until(lambda: time.time() >= until, timeout=10, what="the deadline")
    _tick(spawned)

    assert worker_env in spawned
    [resolved] = _events(worker_env, "wait_resolved")
    assert resolved.payload["outcome"] == "deadline"
    assert not session.exited


def test_task_stays_parked_while_parking_worker_is_alive(worker_env, spawn, all_assignees_spawnable):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli.kanban_db_dispatch import _process_fingerprint

    run_id = int(os.environ["HERMES_KANBAN_RUN_ID"])
    with kbc.connect() as conn, kbc.write_txn(conn):
        kb._append_event(conn, worker_env, "worker_bound",
                         {"pid": os.getpid(), "started_at": _process_fingerprint(os.getpid())},
                         run_id=run_id)
    session = spawn("sleep 2; exit 0")
    assert _park(session)["status"] == "scheduled"
    wait_until(lambda: _exit_recorded(session), timeout=15, what="the exit file")
    spawned: list = []
    _tick(spawned)
    assert _status(worker_env) == "scheduled" and spawned == []

    with kbc.connect() as conn, kbc.write_txn(conn):
        kb._append_event(conn, worker_env, "worker_result", {"exit_code": 0}, run_id=run_id)
    _tick(spawned)

    assert worker_env in spawned


def test_later_plain_schedule_ignores_earlier_process_wait(worker_env, spawn, monkeypatch, all_assignees_spawnable):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from tools import kanban_tools  # registers the kanban tools
    from tools.process_registry import process_registry
    from tools.registry import registry

    session = spawn("sleep 300")
    assert _park(session)["status"] == "scheduled"
    with kbc.connect() as conn:
        assert kb.unblock_task(conn, worker_env)
        kb.claim_task(conn, worker_env)
        monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(kb._current_run_id(conn, worker_env)))
    out = json.loads(registry.get_entry("kanban_schedule").handler({"reason": "parked by hand"}))
    assert out["status"] == "scheduled"
    process_registry.kill_process(session.id, source="test")
    wait_until(lambda: session.exited, timeout=15, what="the first process to end")
    spawned: list = []
    _tick(spawned)

    assert _status(worker_env) == "scheduled" and spawned == []
    assert _events(worker_env, "wait_resolved") == []


def test_wake_with_open_parent_lands_in_todo(worker_env, spawn, all_assignees_spawnable):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    session = spawn("sleep 2; exit 0")
    assert _park(session)["status"] == "scheduled"
    with kbc.connect() as conn:
        parent = kb.create_task(conn, title="upstream", assignee="test-worker")
        kb.claim_task(conn, parent)
        kb.link_tasks(conn, parent, worker_env)
    wait_until(lambda: _exit_recorded(session), timeout=15, what="the exit file")
    spawned: list = []
    _tick(spawned)

    assert _status(worker_env) == "todo"
    assert worker_env not in spawned
    [resolved] = _events(worker_env, "wait_resolved")
    assert resolved.payload["outcome"] == "exited"
