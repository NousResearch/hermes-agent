"""A spawn error cannot authorize a second process under the same original run."""
from pathlib import Path
import subprocess
import sys

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.mark.parametrize('error', [TypeError, ValueError, KeyboardInterrupt])
@pytest.mark.parametrize('cap', ['global', 'profile'])
def test_dispatch_does_not_repeat_callback_after_process_was_started(tmp_path, monkeypatch, all_assignees_spawnable, error, cap):
    home = tmp_path/'hermes'
    home.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setenv('HERMES_KANBAN_HOME', str(home))
    monkeypatch.setenv('HERMES_KANBAN_WORKSPACES_ROOT', str(tmp_path/'workspaces'))
    monkeypatch.setenv('HERMES_KANBAN_CRASH_GRACE_SECONDS', '0')
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    kb.init_db()
    children = []
    original = []
    options = {'max_in_progress': 1} if cap == 'global' else {'max_in_progress': 3, 'max_in_progress_per_profile': 1}
    def spawn(task, workspace, *, board=None):
        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])
        children.append(child)
        original.append((task.id, task.current_run_id, task.claim_lock))
        raise error('post-spawn receipt failed')
    try:
        with kbc.connect_closing() as conn:
            kb.create_task(conn, title='One-shot spawn probe', assignee='fixture', max_runtime_seconds=1)
            kb.create_task(conn, title='Other work must wait', assignee='fixture')
            if error is KeyboardInterrupt:
                with pytest.raises(KeyboardInterrupt):
                    kbd.dispatch_once(conn, spawn_fn=spawn, **options)
            else:
                assert not kbd.dispatch_once(conn, spawn_fn=spawn, **options).spawned
        # Reopen the actual SQLite file: no process-local state can enforce this.
        with kbc.connect_closing() as conn:
            task_id, run_id, claim = original[0]
            with kb.write_txn(conn):
                conn.execute('UPDATE tasks SET claim_expires=1,started_at=1,last_heartbeat_at=1 WHERE id=?', (task_id,))
                conn.execute('UPDATE task_runs SET claim_expires=1,started_at=1 WHERE id=?', (run_id,))
            assert kb.release_stale_claims(conn) == 0
            assert kbd.detect_stale_running(conn, stale_timeout_seconds=1) == []
            assert kbd.reconcile_orphaned_running(conn) == []
            assert kbd.enforce_max_runtime(conn) == []
            assert kb.reclaim_task(conn, task_id) is False
            for transition in [lambda: kb.complete_task(conn, task_id, force=True),
                               lambda: kb.block_task(conn, task_id, reason='operator'),
                               lambda: kb.archive_task(conn, task_id),
                               lambda: kb.unblock_task(conn, task_id),
                               lambda: kb.reopen_review_task(conn, task_id)]:
                with pytest.raises(RuntimeError, match='spawn ownership is unresolved'):
                    transition()
            assert not kbd.dispatch_once(conn, spawn_fn=spawn, **options).spawned
            task = kb.get_task(conn, task_id)
            assert (task.status, task.current_run_id, task.claim_lock) == ('running', run_id, claim)
            assert len(conn.execute('SELECT id FROM task_runs WHERE task_id=?', (task_id,)).fetchall()) == 1
            assert len(children) == 1
            assert children[0].poll() is None
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
            child.wait(timeout=10)


@pytest.mark.parametrize('shape', ['legacy', 'board', 'opaque_type', 'opaque_value'])
def test_spawn_signature_compatibility_invokes_once(shape):
    calls = []
    task = object()
    def legacy(value, workspace):
        calls.append((value, workspace))
        return 123
    def board(value, workspace, *, board=None):
        calls.append((value, workspace, board))
        return 123
    class Opaque:
        @property
        def __signature__(self):
            raise (TypeError if shape == 'opaque_type' else ValueError)('signature unavailable')
        def __call__(self, value, workspace):
            return legacy(value, workspace)
    callback = legacy if shape == 'legacy' else board if shape == 'board' else Opaque()
    assert kbd._call_spawn_fn(callback, task, '/workspace', 'selected-board') == 123
    assert calls == [(task, '/workspace', 'selected-board') if shape == 'board' else (task, '/workspace')]
