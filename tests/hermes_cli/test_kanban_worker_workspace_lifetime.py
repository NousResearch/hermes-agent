"""Completion must not invalidate a physical worker's working directory."""
import subprocess
import sys
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as dispatch


def test_completed_scratch_survives_until_worker_exits(tmp_path, monkeypatch):
    home = tmp_path / 'home'
    home.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(home))
    for key in ('HERMES_KANBAN_DB', 'HERMES_KANBAN_BOARD'):
        monkeypatch.delenv(key, raising=False)
    kb.init_db()
    conn = kbc.connect()
    process = None
    try:
        task_id = kb.create_task(conn, title='Synthetic review', assignee='default', workspace_kind='scratch')
        task = kb.claim_task(conn, task_id)
        assert task is not None
        workspace = home / 'kanban' / 'workspaces' / task_id
        workspace.mkdir(parents=True)
        conn.execute('UPDATE tasks SET workspace_path=? WHERE id=?', (str(workspace), task_id))
        conn.commit()
        process = subprocess.Popen(
            [sys.executable, '-c', 'import os,sys; print("ready",flush=True); sys.stdin.readline(); print(os.getcwd(),flush=True)'],
            cwd=workspace, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        assert process.stdout.readline().strip() == 'ready'
        dispatch._set_worker_pid(conn, task_id, process.pid)
        assert kb.complete_task(conn, task_id, summary='Saved findings', expected_run_id=task.current_run_id,
                                fire_lifecycle_hook=False)
        assert workspace.is_dir(), 'logical completion removed the physical worker cwd'
        stdout, stderr = process.communicate('\n', timeout=10)
        assert process.returncode == 0, stderr
        assert Path(stdout.strip()) == workspace
        # The ordinary reaper must perform deferred cleanup, including after reopening the DB.
        conn.close()
        conn = kbc.connect()
        conn.execute('UPDATE task_runs SET ended_at=0 WHERE task_id=?', (task_id,))
        conn.commit()
        dispatch.reap_terminal_workers(conn)
        assert not workspace.exists()
        assert kb.latest_run(conn, task_id).summary == 'Saved findings'
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.communicate(timeout=10)
        conn.close()


@pytest.mark.parametrize('nested', [False, True])
def test_terminal_sibling_keeps_shared_scratch_until_its_worker_exits(tmp_path, monkeypatch, nested):
    home = tmp_path / 'home'
    home.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.delenv('HERMES_KANBAN_DB', raising=False)
    monkeypatch.delenv('HERMES_KANBAN_BOARD', raising=False)
    kb.init_db()
    conn = kbc.connect()
    process = None
    try:
        first = kb.create_task(conn, title='First', assignee='default', workspace_kind='scratch')
        second = kb.create_task(conn, title='Second', assignee='default', workspace_kind='scratch')
        run = kb.claim_task(conn, first)
        shared = home / 'kanban' / 'workspaces' / first
        worker_path = shared / 'nested' if nested else shared
        worker_path.mkdir(parents=True)
        conn.execute('UPDATE tasks SET workspace_path=? WHERE id=?', (str(worker_path), first))
        conn.execute('UPDATE tasks SET workspace_path=? WHERE id=?', (str(shared), second))
        conn.commit()
        process = subprocess.Popen([sys.executable, '-c', 'import sys; print("ready",flush=True); sys.stdin.readline()'],
                                   cwd=worker_path, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        assert process.stdout.readline().strip() == 'ready'
        dispatch._set_worker_pid(conn, first, process.pid)
        assert kb.complete_task(conn, first, summary='First done', expected_run_id=run.current_run_id,
                                fire_lifecycle_hook=False)
        assert kb.complete_task(conn, second, summary='Second done', fire_lifecycle_hook=False)
        assert shared.is_dir()
    finally:
        if process is not None and process.poll() is None:
            process.communicate('\n', timeout=10)
        conn.close()


def test_reaper_retries_cleanup_after_pid_clear_and_interruption(tmp_path, monkeypatch):
    from hermes_cli import kanban_db_workspace as workspace_module

    home = tmp_path / 'home'
    home.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.delenv('HERMES_KANBAN_DB', raising=False)
    monkeypatch.delenv('HERMES_KANBAN_BOARD', raising=False)
    kb.init_db()
    conn = kbc.connect()
    try:
        task_id = kb.create_task(conn, title='Interrupted cleanup', assignee='default', workspace_kind='scratch')
        run = kb.claim_task(conn, task_id)
        scratch = home / 'kanban' / 'workspaces' / task_id
        scratch.mkdir(parents=True)
        conn.execute('UPDATE tasks SET workspace_path=? WHERE id=?', (str(scratch), task_id))
        conn.commit()
        dispatch._set_worker_pid(conn, task_id, 99999999)
        assert kb.complete_task(conn, task_id, summary='Saved', expected_run_id=run.current_run_id,
                                fire_lifecycle_hook=False)
        conn.execute('UPDATE task_runs SET ended_at=0 WHERE task_id=?', (task_id,))
        conn.commit()
        original = workspace_module._cleanup_workspace
        def interrupted(*args):
            raise OSError('synthetic cleanup interruption after PID clear')
        monkeypatch.setattr(workspace_module, '_cleanup_workspace', interrupted)
        dispatch.reap_terminal_workers(conn)
        assert scratch.is_dir()
        monkeypatch.setattr(workspace_module, '_cleanup_workspace', original)
        conn.close()
        conn = kbc.connect()
        dispatch.reap_terminal_workers(conn)
        assert not scratch.exists()
    finally:
        conn.close()
