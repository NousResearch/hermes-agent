"""A terminal reaper must not orphan the native owner-side interpreter.

The reaper clears mutable run PID fields after killing the submitter. A watcher
whose first tick comes afterwards still needs the same run's original identity.
"""
from contextlib import closing
import os
from pathlib import Path
import subprocess
import sys

import pytest

from hermes_cli import kanban_db as kb, kanban_db_dispatch as dispatch
from hermes_cli.kanban_db_connect import connect


@pytest.fixture
def owned_run(tmp_path, monkeypatch):
    home = tmp_path / "state"
    home.mkdir()
    user = tmp_path / "user"
    user.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HOME", str(user))
    monkeypatch.setattr(Path, "home", lambda: user)
    kb.init_db()
    with closing(connect()) as conn:
        task_id = kb.create_task(conn, title="Owned watcher", assignee="default")
        task = kb.claim_task(conn, task_id)
        yield conn, task_id, task.current_run_id


def _observer(path, run_id, *, expect_alive=False):
    # A real interpreter runs the production watcher: its own os._exit cannot
    # take the test runner down. Only the observation notification is wrapped.
    script = """
import sys, threading
from pathlib import Path
from gateway import session_kanban
from hermes_cli import kanban_db_dispatch as dispatch
if sys.argv[3] == 'alive':
    observed = threading.Event()
    original = dispatch._worker_alive
    def alive(pid, fingerprint):
        result = original(pid, fingerprint)
        if result:
            observed.set()
        return result
    dispatch._worker_alive = alive
session_kanban._watch_submitter(Path(sys.argv[1]), int(sys.argv[2]))
if sys.argv[3] == 'alive':
    assert observed.wait(5), 'watcher lost the live original submitter'
    session_kanban._TURNS_DONE.set()
else:
    threading.Event().wait(15)
"""
    return subprocess.Popen([sys.executable, "-c", script, str(path), str(run_id),
                             "alive" if expect_alive else "dead"],
                            env=dict(os.environ), stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def _cleanup(process):
    if process.poll() is None:
        process.kill()
    process.wait(timeout=5)


@pytest.mark.platforms("linux", "macos")
@pytest.mark.parametrize("registered", [False, True])
def test_reaped_submitter_still_ends_the_bound_interpreter(owned_run, monkeypatch, registered):
    conn, task_id, run_id = owned_run
    submitter = subprocess.Popen([sys.executable, "-c", "import threading; threading.Event().wait(30)"],
                                 stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    observer = None
    try:
        if registered:
            assert dispatch.adopt_worker_pid(conn, task_id, run_id, submitter.pid)
        else:
            dispatch._set_worker_pid(conn, task_id, submitter.pid, run_id=run_id)
        assert kb.block_task(conn, task_id, reason="operator Stop", kind="needs_input", expected_run_id=run_id)
        monkeypatch.setattr(dispatch, "TERMINAL_WORKER_REAP_GRACE_SECONDS", 0)
        assert dispatch.reap_terminal_workers(conn) == [task_id]
        submitter.wait(timeout=5)
        assert conn.execute("SELECT worker_pid,worker_started_at FROM task_runs WHERE id=?", (run_id,)).fetchone()[:] == (None, None)
        # Start AFTER the reaper cleared the row: no scheduling luck or cached
        # prior observation can make this pass.
        observer = _observer(kb.kanban_db_path(), run_id)
        assert observer.wait(timeout=5) == 1
    finally:
        if observer is not None:
            _cleanup(observer)
        _cleanup(submitter)


@pytest.mark.platforms("linux", "macos")
def test_cleared_handle_does_not_end_work_beside_a_live_original_submitter(owned_run):
    conn, task_id, run_id = owned_run
    submitter = subprocess.Popen([sys.executable, "-c", "import threading; threading.Event().wait(30)"],
                                 stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    observer = None
    try:
        dispatch._set_worker_pid(conn, task_id, submitter.pid, run_id=run_id)
        assert kb.block_task(conn, task_id, reason="terminal task with live submitter", kind="needs_input")
        # A cleared mutable row is not itself proof that the original process
        # ended: retain the immutable fingerprint and observe the real process.
        with kb.write_txn(conn):
            conn.execute("UPDATE task_runs SET worker_pid=NULL,worker_started_at=NULL WHERE id=?", (run_id,))
        observer = _observer(kb.kanban_db_path(), run_id, expect_alive=True)
        stdout, stderr = observer.communicate(timeout=8)
        assert observer.returncode == 0, stdout + stderr
        assert submitter.poll() is None
    finally:
        if observer is not None:
            _cleanup(observer)
        _cleanup(submitter)
