"""AgentOps #275: timeout is not cleanup when an owned worker survives."""
import subprocess
import sys
import time

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as dispatch


@pytest.mark.parametrize('signal_result', ['denied', 'delivered_but_alive', 'unavailable', 'identity_lost_after_term',
    pytest.param('status_probe_unavailable', marks=pytest.mark.macos_only)])
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

        signals = []
        def signal(pid, sig):
            assert pid == child.pid
            signals.append(sig)
            if signal_result == 'denied':
                raise PermissionError('injected denied termination')
            if signal_result == 'identity_lost_after_term':
                fault.setattr(dispatch, '_process_fingerprint', lambda _: None)
            # Returning successfully proves delivery only, not process exit.

        with monkeypatch.context() as fault:
            if signal_result == 'unavailable':
                fault.setattr(dispatch, '_kill_fn', lambda _: None)
            if signal_result == 'status_probe_unavailable':
                real_run = subprocess.run
                def unreadable_status(args, *a, **kw):
                    if args[:3] == ['ps', '-o', 'stat=']:
                        return subprocess.CompletedProcess(args, 1, stdout='')
                    return real_run(args, *a, **kw)
                fault.setattr(dispatch.subprocess, 'run', unreadable_status)
            assert dispatch.enforce_max_runtime(conn, signal_fn=signal) == []
        if signal_result == 'identity_lost_after_term':
            assert len(signals) == 1  # No escalation against an unidentified live process.
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


@pytest.mark.parametrize('witness', ['boot_and_start', 'legacy_start'])
@pytest.mark.parametrize('cleanup', ['timeout', 'stale', 'expired_claim', 'crash', 'terminal'])
def test_unreadable_identity_retains_worker_across_automatic_cleanup(tmp_path, monkeypatch, witness, cleanup):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    from gateway import status
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])
    conn = kbc.connect(tmp_path / 'kanban.db')
    try:
        task_id = kb.create_task(conn, title='uncertain cleanup identity', assignee='worker', max_runtime_seconds=1)
        original = kb.claim_task(conn, task_id)
        with monkeypatch.context() as older:
            if witness == 'legacy_start':
                older.setattr(dispatch, '_process_fingerprint', status.get_process_start_time)
            dispatch._set_worker_pid(conn, task_id, child.pid)
        before = conn.execute('SELECT * FROM task_runs WHERE id=?', (original.current_run_id,)).fetchone()
        assert before['worker_started_at'] not in (None, dispatch.UNVERIFIED_WORKER_FINGERPRINT)
        assert dispatch._worker_alive(child.pid, before['worker_started_at'])
        if cleanup == 'terminal':
            assert kb.complete_task(conn, task_id, result='done', expected_run_id=original.current_run_id)
        signals = []
        def record_signal(pid, sig):
            signals.append((pid, sig))
        later = time.time() + 7200
        with monkeypatch.context() as fault:
            fault.setattr(dispatch.time, 'time', lambda: later)
            if witness == 'legacy_start':
                fault.setattr(status, 'get_process_start_time', lambda _: None)
            else:
                fault.setattr(dispatch, '_process_fingerprint', lambda _: None)
            fault.setattr(dispatch, '_kill_fn', lambda _: record_signal)
            actions = {
                'timeout': lambda: dispatch.enforce_max_runtime(conn),
                'stale': lambda: dispatch.detect_stale_running(conn, stale_timeout_seconds=1),
                'expired_claim': lambda: kb.release_stale_claims(conn),
                'crash': lambda: dispatch.detect_crashed_workers(conn),
                'terminal': lambda: dispatch.reap_terminal_workers(conn),
            }
            assert not actions[cleanup]()
        assert signals == [] and child.poll() is None
        after = conn.execute('SELECT * FROM task_runs WHERE id=?', (original.current_run_id,)).fetchone()
        assert (after['worker_pid'], after['worker_started_at']) == (before['worker_pid'], before['worker_started_at'])
        if cleanup != 'terminal':
            held = kb.get_task(conn, task_id)
            assert held.status == 'running' and held.current_run_id == original.current_run_id
            assert held.worker_pid == child.pid and kb.claim_task(conn, task_id) is None
    finally:
        child.terminate()
        child.wait(timeout=10)
        conn.close()
