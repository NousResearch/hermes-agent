"""AgentOps #275: timeout is not cleanup when an owned worker survives."""
import subprocess
import sys
import time

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as dispatch


@pytest.mark.parametrize('signal_result', ['denied', 'delivered_but_alive', 'unavailable'])
def test_timeout_retains_run_until_owned_cleanup_is_verified(tmp_path, monkeypatch, signal_result):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    ready = tmp_path / 'ready'
    child = subprocess.Popen([sys.executable, '-c',
        'import pathlib,sys,time; pathlib.Path(sys.argv[1]).touch(); time.sleep(120)', str(ready)])
    conn = kbc.connect(tmp_path / 'kanban.db')
    try:
        wait_until = time.monotonic() + 10
        while not ready.exists() and time.monotonic() < wait_until:
            time.sleep(0.05)
        assert ready.exists() and child.poll() is None
        task_id = kb.create_task(conn, title='timeout ownership', assignee='worker', max_runtime_seconds=1)
        original = kb.claim_task(conn, task_id)
        dispatch._set_worker_pid(conn, task_id, child.pid)
        fingerprint = conn.execute('SELECT worker_started_at FROM tasks WHERE id=?',
                                   (task_id,)).fetchone()['worker_started_at']
        assert fingerprint != dispatch.UNVERIFIED_WORKER_FINGERPRINT
        assert dispatch._worker_alive(child.pid, fingerprint)
        time.sleep(2)

        def signal(pid, sig):
            assert pid == child.pid
            if signal_result == 'denied':
                raise PermissionError('injected denied termination')
            # Returning successfully proves delivery only, not process exit.

        with monkeypatch.context() as fault:
            if signal_result == 'unavailable':
                fault.setattr(dispatch, '_kill_fn', lambda _: None)
            assert dispatch.enforce_max_runtime(conn, signal_fn=signal) == []
        held = kb.get_task(conn, task_id)
        assert child.poll() is None
        assert held.status == 'running' and held.current_run_id == original.current_run_id
        assert held.worker_pid == child.pid and kb.claim_task(conn, task_id) is None
        events = kb.list_events(conn, task_id)
        assert any(e.kind == 'reclaim_deferred' and e.payload['reason'] == 'max_runtime_worker_alive'
                   for e in events)
        assert not any(e.kind == 'timed_out' for e in events)

        child.terminate()
        child.wait(timeout=10)
        assert dispatch.enforce_max_runtime(conn) == [task_id]
        assert dispatch.enforce_max_runtime(conn) == []
        after = kb.get_task(conn, task_id)
        assert after.worker_pid is None and after.current_run_id is None
        assert len([e for e in kb.list_events(conn, task_id) if e.kind == 'timed_out']) == 1
        runs = conn.execute('SELECT * FROM task_runs WHERE task_id=?', (task_id,)).fetchall()
        assert len(runs) == 1 and runs[0]['id'] == original.current_run_id
        assert runs[0]['outcome'] == 'timed_out'
    finally:
        if child.poll() is None:
            child.terminate()
        child.wait(timeout=10)
        conn.close()
