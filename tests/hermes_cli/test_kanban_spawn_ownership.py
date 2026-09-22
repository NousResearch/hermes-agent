"""Uncertain launch ownership needs original process evidence, not expiry."""
from pathlib import Path
import subprocess
import sys

import pytest

from hermes_cli import kanban_db as kb, kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as dispatch, kanban_spawn_ownership as ownership


@pytest.mark.parametrize('receipt', ['alive', 'gone', 'unreadable', 'mismatch', 'remote', 'none', 'pid', 'workspace_error'])
def test_only_original_worker_exit_resolves_uncertain_launch(tmp_path, monkeypatch, all_assignees_spawnable, receipt):
    home = tmp_path/'hermes'
    home.mkdir()
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setenv('HERMES_KANBAN_HOME', str(home))
    monkeypatch.setenv('HERMES_KANBAN_WORKSPACES_ROOT', str(tmp_path/'workspaces'))
    monkeypatch.setenv('HERMES_KANBAN_CRASH_GRACE_SECONDS', '0')
    kb.init_db()
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])
    calls = []
    try:
        with kbc.connect_closing() as conn:
            task_id = kb.create_task(conn, title='receipt probe', assignee='fixture')
            def spawn(task, workspace, *, board=None):
                calls.append(task.current_run_id)
                return None if receipt == 'none' else child.pid
            if receipt == 'workspace_error':
                from hermes_cli import kanban_db_workspace as workspace
                def fail(*args, **kwargs):
                    raise OSError('workspace unavailable before callback')
                monkeypatch.setattr(workspace, 'resolve_workspace', fail)
            result = dispatch.dispatch_once(conn, spawn_fn=spawn, max_in_progress=1)
            task = kb.get_task(conn, task_id)
            if receipt == 'workspace_error':
                assert calls == [] and not result.spawned
                assert task.status == 'ready' and task.current_run_id is None
                assert not ownership.pending(conn, task_id)
                return
            assert len(calls) == len(result.spawned) == 1
            assert not ownership.pending(conn, task_id)
            if receipt in {'none', 'pid'}:
                assert kb.complete_task(conn, task_id, expected_run_id=task.current_run_id)
                return
            ownership.uncertain(conn, task, RuntimeError('receipt interrupted'))
            if receipt == 'gone':
                child.terminate()
                child.wait(timeout=10)
            elif receipt == 'unreadable':
                monkeypatch.setattr(dispatch, '_process_fingerprint', lambda pid: None)
            elif receipt == 'mismatch':
                with kb.write_txn(conn):
                    conn.execute('UPDATE task_runs SET worker_started_at=? WHERE id=?', ('other', task.current_run_id))
            elif receipt == 'remote':
                with kb.write_txn(conn):
                    conn.execute('UPDATE tasks SET claim_lock=? WHERE id=?', ('other-host:1', task_id))
                    conn.execute('UPDATE task_runs SET claim_lock=? WHERE id=?', ('other-host:1', task.current_run_id))
        with kbc.connect_closing() as conn:
            assert ownership.pending(conn, task_id)
            assert ownership.reconcile(conn, task_id) is (receipt == 'gone')
            assert ownership.pending(conn, task_id) is (receipt != 'gone')
            if receipt == 'gone':
                assert kb.reclaim_task(conn, task_id)
                assert kb.get_task(conn, task_id).current_run_id is None
            else:
                assert not kb.reclaim_task(conn, task_id)
                assert kb.get_task(conn, task_id).current_run_id == task.current_run_id
    finally:
        if child.poll() is None:
            child.terminate()
        child.wait(timeout=10)
