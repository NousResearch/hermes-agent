"""Original worker results and process cleanup remain separate across launch receipts."""
from pathlib import Path
import json
import subprocess
import sys
import time
import traceback

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
    monkeypatch.setattr(dispatch, 'review_dispatch_enabled', lambda: True)
    kb.init_db()
    return home


@pytest.mark.parametrize('transition', ['complete', 'block', 'review', 'changes', 'schedule'])
@pytest.mark.parametrize('receipt', ['live_pid', 'dead_pid', 'none', 'error'])
@pytest.mark.parametrize('cap', ['global', 'profile'])
def test_original_terminal_result_survives_launch_receipt_without_releasing_live_capacity(
        board_home, tmp_path, monkeypatch, transition, receipt, cap):
    children = []
    launches = []
    hooks = []
    callback_errors = []
    monkeypatch.setattr(kb, '_kanban_observer_consumed', lambda name: True)
    monkeypatch.setattr(kb, '_fire_kanban_lifecycle_hook', lambda *args, **kwargs: hooks.append((args, kwargs)))
    expected_status = {'complete': 'done', 'block': 'blocked', 'review': 'review', 'changes': 'ready', 'schedule': 'scheduled'}[transition]
    options = {'max_in_progress': 1} if cap == 'global' else {'max_in_progress': 3, 'max_in_progress_per_profile': 1}
    output = tmp_path/'worker-result.json'
    source = '''import json,sys
from pathlib import Path
from hermes_cli import kanban_db as kb, kanban_db_connect as kbc
with kbc.connect_closing() as conn:
    task,run,action,path=sys.argv[1:]
    kwargs={'expected_run_id':int(run)}
    if action=='complete': result=kb.complete_task(conn,task,result='finished',**kwargs)
    elif action=='block': result=kb.block_task(conn,task,reason='needs input',**kwargs)
    elif action=='review': result=kb.request_review(conn,task,summary='finished',reviewer='reviewer',**kwargs)
    elif action=='schedule': result=kb.schedule_task(conn,task,reason='wait',**kwargs)
    else: result=kb.request_changes(conn,task,reason='fix finding',**kwargs)[0]
    staged=Path(path).with_suffix('.pending')
    staged.write_text(json.dumps({'accepted':result}))
    staged.replace(path)
sys.stdin.read(1)
'''
    def spawn(task, workspace, *, board=None):
        launches.append((task.id, task.current_run_id, task.claim_lock))
        child = subprocess.Popen([sys.executable, '-c', source, task.id, str(task.current_run_id), transition, str(output)],
                                 stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        children.append(child)
        until = time.monotonic()+10
        while not output.exists() and child.poll() is None and time.monotonic() < until:
            time.sleep(.01)
        assert output.exists(), 'worker did not publish its result before the launch receipt'
        assert json.loads(output.read_text()) == {'accepted': True}
        # The callback is still in progress but the task has already transitioned.
        with kbc.connect_closing() as other:
            assert kb.get_task(other, task.id).status == expected_status
            assert ownership.pending(other, task.id)
            assert dispatch.count_running_tasks(other) == 1
            assert kb.claim_task(other, task.id) is None
            assert kb.claim_review_task(other, task.id) is None
            with pytest.raises(RuntimeError, match='spawn ownership is unresolved'):
                kb.delete_task(other, task.id)
        if receipt in {'none', 'dead_pid'}:
            child.communicate(input='x', timeout=10)
            assert child.returncode == 0
        if receipt == 'error':
            raise TypeError('late receipt lost after accepted terminal result')
        return None if receipt == 'none' else child.pid
    def spawn_with_diagnostics(task, workspace, *, board=None):
        try:
            return spawn(task, workspace, board=board)
        except Exception as exc:
            callback_errors.append({'type': type(exc).__name__, 'error': str(exc),
                                    'traceback': traceback.format_exc(),
                                    'worker_output': output.read_text() if output.exists() else None})
            raise
    try:
        with kbc.connect_closing() as conn:
            task_id = kb.create_task(conn, title='early result', assignee='fixture', priority=10)
            if transition == 'changes':
                assert kb.request_review(conn, task_id, summary='implementation', reviewer='fixture')
            first = dispatch.dispatch_once(conn, spawn_fn=spawn_with_diagnostics, **options)
            expected_error = receipt == 'error'
            if callback_errors and (not expected_error or callback_errors[-1]['type'] != 'TypeError'
                                    or callback_errors[-1]['error'] != 'late receipt lost after accepted terminal result'):
                pytest.fail(json.dumps(callback_errors, indent=2))
            if bool(first.spawned) is expected_error:
                evidence = {'callback_errors': callback_errors,
                            'task': dict(conn.execute('SELECT * FROM tasks WHERE id=?', (task_id,)).fetchone()),
                            'runs': [dict(r) for r in conn.execute('SELECT * FROM task_runs WHERE task_id=?', (task_id,))],
                            'events': [dict(r) for r in conn.execute('SELECT * FROM task_events WHERE task_id=?', (task_id,))]}
                pytest.fail(json.dumps(evidence, indent=2))
            run_id = launches[0][1]
            task = kb.get_task(conn, task_id)
            run = conn.execute('SELECT * FROM task_runs WHERE id=?', (run_id,)).fetchone()
            assert task.status == expected_status and task.current_run_id is None
            assert task.worker_pid is None
            assert run['ended_at'] is not None and run['claim_lock'] == launches[0][2]
            assert bool(first.spawned) is (receipt != 'error')
            spawned_hooks = [kwargs for args, kwargs in hooks if args[0] == 'on_kanban_worker_spawned']
            assert [hook['run_id'] for hook in spawned_hooks] == ([] if receipt == 'error' else [run_id])
            if receipt.endswith('pid'):
                assert run['worker_pid'] == children[0].pid
            kb.create_task(conn, title='next work', assignee='fixture')
        # Reopening must recover terminal occupancy from the run, not the task pointer.
        with kbc.connect_closing() as conn:
            if receipt in {'live_pid', 'error'}:
                called = []
                dispatch.dispatch_once(conn, spawn_fn=lambda *a, **k: called.append(True), **options)
                assert called == []
                assert ownership.pending(conn, task_id)
                assert dispatch.count_running_tasks(conn) == 1
                assert children[0].poll() is None
            for child in children:
                if child.poll() is None:
                    child.communicate(input='x', timeout=10)
            assert ownership.reconcile(conn, task_id) is (receipt != 'error')
            assert dispatch.count_running_tasks(conn) == (1 if receipt == 'error' else 0)
            assert kb.get_task(conn, task_id).status == expected_status
            assert len(launches) == 1
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
            child.communicate(timeout=10)


def test_terminal_reaper_settles_original_run_before_erasing_process_identity(board_home, monkeypatch):
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])
    monkeypatch.setattr(dispatch, 'TERMINAL_WORKER_REAP_GRACE_SECONDS', 0)
    try:
        with kbc.connect_closing() as conn:
            task_id = kb.create_task(conn, title='terminal cleanup', assignee='fixture')
            dispatch.dispatch_once(conn, spawn_fn=lambda *a, **k: child.pid)
            task = kb.get_task(conn, task_id)
            assert kb.complete_task(conn, task_id, expected_run_id=task.current_run_id)
            assert dispatch.count_running_tasks(conn) == 1
            assert ownership.pending(conn, task_id)
            dispatch.reap_terminal_workers(conn)
            child.wait(timeout=10)
            assert not ownership.pending(conn, task_id)
            assert dispatch.count_running_tasks(conn) == 0
            run = conn.execute('SELECT * FROM task_runs WHERE id=?', (task.current_run_id,)).fetchone()
            assert run['worker_pid'] is None
            assert kb.get_task(conn, task_id).status == 'done'
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=10)
