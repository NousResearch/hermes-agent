"""Whole-execution cleanup is a kernel child-owner receipt, never root exit."""
import json
import os
from pathlib import Path
import signal
import sys
import time

import pytest

from hermes_cli import kanban_db as kb, kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as dispatch, kanban_spawn_ownership as ownership
from hermes_cli import kanban_execution_scope as scopes


@pytest.fixture
def board(tmp_path, monkeypatch, all_assignees_spawnable):
    home = tmp_path/'hermes'
    home.mkdir()
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setenv('HERMES_KANBAN_HOME', str(home))
    monkeypatch.setenv('HERMES_KANBAN_WORKSPACES_ROOT', str(tmp_path/'workspaces'))
    kb.init_db()
    return home


def wait_for(check, timeout=15):
    until = time.monotonic()+timeout
    while time.monotonic() < until:
        value = check()
        if value:
            return value
        time.sleep(.02)
    pytest.fail('bounded execution-scope observation timed out')


def test_original_scope_cannot_replay_or_settle_from_root_exit_after_restart(board):
    with kbc.connect_closing() as conn:
        task_id = kb.create_task(conn, title='original scope', assignee='fixture', max_runtime_seconds=60)
        task = kb.claim_task(conn, task_id)
        ownership.begin(conn, task)
        assert not scopes.settled(conn, task.current_run_id)  # legacy root is not whole execution
        scope = scopes.prepare(conn, task)
        with pytest.raises(RuntimeError, match='fresh original'):
            scopes.prepare(conn, task, deadline=scope['deadline']+60)
        assert scopes.read(conn, task.current_run_id) == scope
        assert not ownership.cleanup_verified(conn, task_id, task.current_run_id, None, None)
        assert kb.complete_task(conn, task_id, expected_run_id=task.current_run_id)
        kb.gc_events(conn, older_than_seconds=-1)
    with kbc.connect_closing() as conn:
        assert scopes.read(conn, task.current_run_id) == scope
        assert not ownership.reconcile(conn, task_id)
        assert ownership.pending(conn, task_id)
        assert dispatch.count_running_tasks(conn) == 1
        assert kb.claim_task(conn, task_id) is None
        with pytest.raises(RuntimeError, match='unresolved'):
            kb.delete_task(conn, task_id)


@pytest.mark.linux_only
@pytest.mark.parametrize('case', ['root_exit', 'detached', 'deadline', 'delayed', 'supervisor_loss',
                                 'lost_receipt', 'manual_reclaim', 'stale', 'terminal_reaper',
                                 'billing', 'auth', 'rate_limit'])
def test_native_supervisor_owns_descendants_and_cutoff_through_cleanup(board, tmp_path, monkeypatch, case):
    provider_case = case in {'billing', 'auth', 'rate_limit'}
    monkeypatch.setattr(kb, '_resolve_crash_grace_seconds', lambda: 0)
    workspace = tmp_path/'work'
    workspace.mkdir()
    publication, root_release, child_release = [workspace/name for name in ('children.json','root-release','child-release')]
    child_source = '''import os,signal,sys,time
from pathlib import Path
signal.signal(signal.SIGTERM, signal.SIG_IGN)
os.environ.clear()
until=time.monotonic()+25
while not Path(sys.argv[1]).exists() and time.monotonic()<until: time.sleep(.02)
'''
    root_source = '''import json,os,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,sys.argv[6])
if sys.argv[5] in {'billing','auth','rate_limit'}:
    # A real descendant inheriting the task/run/claim cannot impersonate its worker.
    bad="import sys;sys.path.insert(0,sys.argv[1]);from hermes_cli.kanban_worker_failure import report_provider_failure;report_provider_failure({'failure_reason':'model_not_found'})"
    subprocess.run([sys.executable,'-c',bad,sys.argv[6]],check=True,timeout=10)
child=subprocess.Popen([sys.executable,'-c',sys.argv[1],sys.argv[4]],start_new_session=sys.argv[5]=='detached')
out=Path(sys.argv[2]);stage=out.with_suffix('.pending')
stage.write_text(json.dumps({'root':os.getpid(),'child':child.pid}));stage.replace(out)
until=time.monotonic()+25
while not Path(sys.argv[3]).exists() and time.monotonic()<until: time.sleep(.02)
if sys.argv[5] in {'billing','auth','rate_limit'}:
    from hermes_cli.kanban_worker_failure import report_provider_failure
    report_provider_failure({'failure_reason':sys.argv[5]})
'''

    command = [sys.executable,'-c',root_source,child_source,str(publication),str(root_release),str(child_release),case,str(Path(__file__).resolve().parents[2])]
    monkeypatch.setattr(dispatch, '_worker_argv', lambda *args: command)
    launched = []
    real_popen = dispatch.subprocess.Popen
    def popen(argv, **kwargs):
        if 'kanban_execution_supervisor.py' in str(argv):
            if case == 'delayed':
                time.sleep(7)  # Cross the frozen cutoff after prepare but before exec.
            proc = real_popen(argv, **kwargs)
            launched.append(proc)
            return proc
        return real_popen(argv, **kwargs)
    monkeypatch.setattr(dispatch.subprocess, 'Popen', popen)
    child = None
    try:
        with kbc.connect_closing() as conn:
            task_id = kb.create_task(conn, title='scope subprocess', assignee='fixture',
                workspace_kind='dir', workspace_path=str(workspace),
                max_runtime_seconds=6 if case in {'deadline','delayed'} else 20)
            dispatch.dispatch_once(conn, max_in_progress=1)
            task = kb.get_task(conn, task_id)
            assert launched, task.last_failure_error
            original = scopes.read(conn, task.current_run_id)
            assert original and ownership.pending(conn, task_id)
            if case == 'delayed':
                wait_for(lambda: scopes.settled(conn, task.current_run_id))
                assert not publication.exists()
                assert scopes.read(conn, task.current_run_id)['reason'] == 'deadline_before_launch'
            else:
                wait_for(publication.exists)
                pids = json.loads(publication.read_text())
                child = pids['child']
                child_identity = dispatch._process_fingerprint(child)
                assert child_identity
                # A completed result is not a cleanup proof. Descendants even
                # discard task env and detach; subreaping keeps kernel ownership.
                if not provider_case and case not in {'manual_reclaim','stale'}:
                    assert kb.complete_task(conn, task_id, expected_run_id=task.current_run_id)
                if provider_case:
                    scope = scopes.read(conn, task.current_run_id)
                    assert scope['worker_pid'] == pids['root'] != task.worker_pid
                    assert scope['supervisor_pid'] == task.worker_pid
                    assert not [e for e in kb.list_events(conn, task_id) if e.kind == 'provider_failure']
                assert not ownership.reconcile(conn, task_id)
                if case == 'supervisor_loss':
                    launched[0].kill()
                    launched[0].wait(timeout=5)
                    root_release.touch(); child_release.touch()
                    wait_for(lambda: not dispatch._pid_alive(child))
                    assert not scopes.settled(conn, task.current_run_id)
                    assert not ownership.reconcile(conn, task_id)
                    assert dispatch.count_running_tasks(conn) == 1
                else:
                    if case == 'lost_receipt':
                        with kb.write_txn(conn):
                            conn.execute('UPDATE task_runs SET worker_pid=NULL,worker_started_at=NULL,spawn_state=\'uncertain\' WHERE id=?', (task.current_run_id,))
                            conn.execute('UPDATE tasks SET worker_pid=NULL,worker_started_at=NULL WHERE id=?', (task_id,))
                    if case == 'manual_reclaim':
                        assert not kb.reclaim_task(conn, task_id)
                    elif case == 'stale':
                        with kb.write_txn(conn):
                            conn.execute('UPDATE task_runs SET started_at=? WHERE id=?', (int(time.time())-7200,task.current_run_id))
                            conn.execute('UPDATE tasks SET last_heartbeat_at=? WHERE id=?', (int(time.time())-7200,task_id))
                        assert dispatch.detect_stale_running(conn, stale_timeout_seconds=60) == []
                    elif case == 'terminal_reaper':
                        with kb.write_txn(conn):
                            conn.execute('UPDATE task_runs SET ended_at=? WHERE id=?', (int(time.time())-180,task.current_run_id))
                        assert dispatch.reap_terminal_workers(conn) == []
                    elif case != 'deadline':
                        root_release.touch()
                    wait_for(lambda: scopes.settled(conn, task.current_run_id))
                    assert dispatch._worker_identity(child, child_identity) in {'gone','foreign'}
                    assert ownership.reconcile(conn, task_id)
                    if case in {'manual_reclaim','stale'}:
                        assert kb.reclaim_task(conn, task_id)
                    if not provider_case:
                        assert dispatch.count_running_tasks(conn) == 0
                    if case == 'deadline':
                        assert scopes.read(conn, task.current_run_id)['reason'] == 'deadline'
            assert scopes.read(conn, task.current_run_id)['deadline'] == original['deadline']
            if not provider_case:
                kb.gc_events(conn, older_than_seconds=-1)
        with kbc.connect_closing() as conn:
            assert scopes.settled(conn, task.current_run_id) is (case != 'supervisor_loss')
            if provider_case:
                # New connection and no in-memory child returncode: normal restart reclaim.
                launched[0].wait(timeout=10)
                from hermes_cli.kanban_worker_failure import provider_verdict
                verdict = provider_verdict(conn, task_id, task.worker_pid)
                assert verdict and verdict['reason'] == case and verdict['pid'] == pids['root']
                assert len([e for e in kb.list_events(conn, task_id) if e.kind == 'provider_failure']) == 1
                dispatch.detect_crashed_workers(conn)
                assert kb.get_task(conn, task_id).status == ('ready' if case == 'rate_limit' else 'blocked')
                assert kb.get_task(conn, task_id).consecutive_failures == 0
                run = conn.execute('SELECT * FROM task_runs WHERE id=?', (task.current_run_id,)).fetchone()
                assert json.loads(run['metadata'])['provider_failure'] == verdict
                assert dispatch.count_running_tasks(conn) == 0

    finally:
        # Cooperative test-owned release also works after reparenting; never
        # signal an orphan using guessed ancestry or an unverified PID.
        root_release.touch(); child_release.touch()
        for proc in launched:
            if proc.poll() is None:
                proc.terminate()
            proc.wait(timeout=10)
        if child:
            wait_for(lambda: not dispatch._pid_alive(child), timeout=10)
