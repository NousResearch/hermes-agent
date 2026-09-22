"""Launch ownership is durable run state; pruning audit history is not cleanup."""
from pathlib import Path
import subprocess
import sys
import time

import pytest

from hermes_cli import kanban_db as kb, kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as dispatch, kanban_spawn_ownership as ownership


@pytest.fixture
def board_home(tmp_path, monkeypatch, all_assignees_spawnable):
    home = tmp_path/'hermes'
    home.mkdir()
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setenv('HERMES_KANBAN_HOME', str(home))
    monkeypatch.setenv('HERMES_KANBAN_WORKSPACES_ROOT', str(tmp_path/'workspaces'))
    kb.init_db()
    return home


@pytest.mark.parametrize('receipt', ['lost', 'known', 'unverified', 'settled', 'none'])
@pytest.mark.parametrize('retention', ['full', 'partial'])
@pytest.mark.parametrize('cap', ['global', 'profile'])
def test_retention_neither_releases_unresolved_owner_nor_restores_settled_owner(
        board_home, monkeypatch, receipt, retention, cap):
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])
    original = []
    base_time = time.time()
    options = {'max_in_progress': 1} if cap == 'global' else {'max_in_progress': 3, 'max_in_progress_per_profile': 1}
    if receipt == 'unverified':
        monkeypatch.setattr(dispatch, '_process_fingerprint', lambda pid: None)
    try:
        with kbc.connect_closing() as conn:
            task_id = kb.create_task(conn, title='original owner', assignee='fixture')
            with monkeypatch.context() as launch_clock:
                launch_clock.setattr(kb.time, 'time', lambda: base_time)
                def spawn(task, workspace, *, board=None):
                    original.append(task.current_run_id)
                    # Attempt ages out while terminal/receipt events remain recent.
                    if retention == 'partial':
                        launch_clock.setattr(kb.time, 'time', lambda: base_time+2*24*3600)
                    with kbc.connect_closing() as other:
                        assert kb.complete_task(other, task.id, result='finished', expected_run_id=task.current_run_id)
                    if receipt == 'lost':
                        raise TypeError('child launched but receipt lost')
                    if receipt == 'none':
                        child.terminate()
                        child.wait(timeout=10)
                        return None
                    return child.pid
                dispatch.dispatch_once(conn, spawn_fn=spawn, **options)
                if receipt == 'settled':
                    child.terminate()
                    child.wait(timeout=10)
                    assert ownership.reconcile(conn, task_id)
            unresolved = receipt not in {'settled', 'none'}
            assert ownership.pending(conn, task_id) is unresolved
            with monkeypatch.context() as gc_clock:
                gc_clock.setattr(kb.time, 'time', lambda: base_time+31*24*3600)
                assert kb.gc_events(conn) > 0
            remaining = conn.execute('SELECT COUNT(*) FROM task_events WHERE task_id=?', (task_id,)).fetchone()[0]
            assert (remaining == 0) is (retention == 'full')
        with kbc.connect_closing() as conn:
            assert ownership.pending(conn, task_id) is unresolved
            assert dispatch.count_running_tasks(conn) == int(unresolved)
            kb.create_task(conn, title='following work', assignee='fixture')
            calls = []
            dispatch.dispatch_once(conn, spawn_fn=lambda *a, **k: calls.append(True), **options)
            assert bool(calls) is not unresolved
            assert kb.get_task(conn, task_id).status == 'done'
            if unresolved:
                assert child.poll() is None
                with pytest.raises(RuntimeError, match='spawn ownership is unresolved'):
                    kb.delete_task(conn, task_id)
                child.terminate()
                child.wait(timeout=10)
                # A lost PID cannot be guessed from expired audit history.
                assert ownership.reconcile(conn, task_id) is (receipt != 'lost')
                assert ownership.pending(conn, task_id) is (receipt == 'lost')
            assert len(original) == 1
    finally:
        if child.poll() is None:
            child.terminate()
        child.wait(timeout=10)


def test_additive_schema_upgrade_preserves_legacy_runs_and_new_ownership(board_home):
    with kbc.connect_closing() as conn:
        legacy = kb.create_task(conn, title='legacy run', assignee='fixture')
        claimed = kb.claim_task(conn, legacy)
        assert kb.complete_task(conn, legacy, result='old result', expected_run_id=claimed.current_run_id)
        before = dict(conn.execute('SELECT * FROM task_runs WHERE id=?', (claimed.current_run_id,)).fetchone())
        before.pop('spawn_state')
        # Construct the pre-change schema, without changing historical run data.
        with kb.write_txn(conn):
            conn.execute('ALTER TABLE task_runs DROP COLUMN spawn_state')
    # Explicit init performs upgrade; same-process connect caches prior initialization.
    kb.init_db()
    with kbc.connect_closing() as conn:
        restored = dict(conn.execute('SELECT * FROM task_runs WHERE id=?', (claimed.current_run_id,)).fetchone())
        assert restored.pop('spawn_state') is None
        assert restored == before
        task_id = kb.create_task(conn, title='new interrupted owner', assignee='fixture')
        task = kb.claim_task(conn, task_id)
        ownership.begin(conn, task)
        assert kb.complete_task(conn, task_id, expected_run_id=task.current_run_id)
        ownership.uncertain(conn, task, RuntimeError('receipt lost'))
    # Re-running init/upgrade must not overwrite an existing ownership state.
    kb.init_db()
    with kbc.connect_closing() as conn:
        assert ownership.pending(conn, task_id)
        assert dispatch.count_running_tasks(conn) == 1
        assert kb.get_task(conn, legacy).result == 'old result'
