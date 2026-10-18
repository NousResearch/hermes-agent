"""Kernel process identity survives clock correction without relaxing worker fences."""
from contextlib import closing
import io
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
from types import SimpleNamespace

import psutil
import pytest

from hermes_state import SessionDB
from hermes_state_runtime import RuntimeStoreError, admit_session_input, begin_runtime_epoch, claim_session_input


@pytest.fixture
def worker(tmp_path):
    root = Path(__file__).resolve().parents[2]
    home = tmp_path / 'user'
    home.mkdir()
    env = {'PATH': os.environ['PATH'], 'HOME': str(home), 'HERMES_HOME': str(home / 'state'),
           'TMPDIR': str(tmp_path), 'PYTHONPATH': str(root)}
    code = ('import sys; from pathlib import Path; import agent.managed_worker as m; '
            f'assert Path(m.__file__).resolve() == Path({str(root / "agent/managed_worker.py")!r}); '
            'hello = m.hello(); assert "agent.context_compressor" not in sys.modules; '
            'assert "hermes_cli.config" not in sys.modules; '
            'sys.stdout.buffer.write(m.encode_frame({"type": "hello", **hello})); '
            'sys.stdout.buffer.flush(); sys.stdin.read()')
    child = subprocess.Popen([sys.executable, '-c', code], cwd=root, env=env,
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        from agent.managed_worker import read_frame
        yield child, read_frame(child.stdout)
    finally:
        child.stdin.close()
        child.wait(timeout=10)
        child.stdout.close()
        child.stderr.close()


def _authority(tmp_path, db):
    from gateway.session_authority import SessionAuthority
    runner = SimpleNamespace(_draining=False, config=SimpleNamespace(multiplex_profiles=False))
    db.create_session('owned', 'cli')
    authority = SessionAuthority(runner, profile_id=str(tmp_path), instance_id='owner', db=db,
                                 epoch=begin_runtime_epoch(db, instance_id='owner'))
    admit_session_input(db, epoch=authority.epoch, principal_id='human', session_id='owned', request_id='input', payload={})
    return authority, claim_session_input(db, epoch=authority.epoch, session_id='owned')


def _shift_clock(monkeypatch):
    original = psutil.Process.create_time
    monkeypatch.setattr(psutil.Process, 'create_time', lambda self: original(self) + 3600)


@pytest.mark.platforms('linux')
@pytest.mark.parametrize('shift_at', ['admission', 'persistence'])
@pytest.mark.asyncio
async def test_clock_shift_preserves_reservation_persistence_and_recovery(tmp_path, monkeypatch, worker, shift_at):
    from agent.managed_worker import encode_frame, read_frame
    from gateway.session_contract import Principal, SessionRef
    from gateway.session_worker import worker_request
    from gateway.session_worker_reservation import reserve_admission_worker
    from hermes_cli.kanban_owner_recovery import owner_reclaim_paused
    child, hello = worker
    with closing(SessionDB(tmp_path / 'state.db')) as db:
        authority, row = _authority(tmp_path, db)
        actor = Principal('human', str(tmp_path), frozenset({'worker:adopt'}), 'ticket')
        connection = SimpleNamespace(authority=authority, actor=actor)
        ref = SessionRef(str(tmp_path), 'owned')
        if shift_at == 'admission':
            _shift_clock(monkeypatch)
        scope = reserve_admission_worker(authority, admission_id=row['admission_id'], process=child,
                                         principal_id='human', hello=hello)
        identity = {k: v for k, v in scope.items() if k != 'epoch'}
        from tui_gateway.contracts.canonical import WorkerScopeParams
        assert WorkerScopeParams.model_validate(identity).birth == scope['birth']
        await worker_request(connection, ref, identity, operation='adopt')
        if shift_at == 'persistence':
            _shift_clock(monkeypatch)

        async def persist(sequence, operation, payload):
            # The same JSON frame boundary as managed worker requests; birth is opaque on wire.
            params = read_frame(io.BytesIO(encode_frame(scope | dict(sequence=sequence, operation=operation, payload=payload))))
            return await worker_request(connection, ref, params, operation='persist')

        assert (await persist(1, 'session.title', {'title': 'Clock-stable worker', 'source': 'user'}))['value']
        appended = await persist(2, 'transcript.append', {'messages': [{'role': 'assistant', 'content': 'durable after clock shift'}]})
        assert appended['count'] == 1
        # Dispatch is a second identity consumer, inside the real receipt transaction.
        from tools.async_delegation import _initialize_schema
        db._execute_write(_initialize_schema)
        await persist(3, 'delegation.dispatch', dict(delegation_id='clock-child', task={'goal': 'inert'},
            dispatched_at=1, parent_session_id='owned', session_key='', origin_ui_session_id='', origin_session_id=''))
        assert db._read_one('SELECT owner_pid,owner_execution_id FROM async_delegations')[:] == (child.pid, scope['execution_id'])

        # A started admission's recovery fence must still recognize the original interpreter.
        with closing(sqlite3.connect(':memory:')) as board:
            board.row_factory = sqlite3.Row
            board.executescript('CREATE TABLE tasks(id,current_run_id); CREATE TABLE task_events(id,task_id,run_id,kind,payload); '
                                "INSERT INTO tasks VALUES('card','run');")
            binding = dict(db=str(tmp_path / 'state.db'), session_id='owned', request_id='input', pid=child.pid, birth=scope['birth'])
            board.execute("INSERT INTO task_events VALUES(1,'card','run','owner_admitted',?)", (json.dumps(binding),))
            assert not owner_reclaim_paused(board, 'card')
        assert (await persist(4, 'execution.finish', {}))['status'] == 'terminal'
        assert db.get_session('owned')['title'] == 'Clock-stable worker'
        assert db.get_messages('owned')[0]['content'] == 'durable after clock shift'
        assert db._read_one('SELECT status FROM worker_executions')[0] == 'terminal'
        # Launcher cleanup uses the same tagged identity even after the clock shifted.
        from gateway.session_managed_worker import ManagedWorker
        handle = SimpleNamespace(worker=(child.pid, scope['birth']), process=SimpleNamespace(pid=-1))
        ManagedWorker._signal_worker(handle, kill=False)
        assert child.wait(timeout=10) < 0


@pytest.mark.platforms('linux')
@pytest.mark.parametrize('fault', ['ticks', 'boot', 'malformed', 'unreadable', 'boot-unreadable', 'legacy-drift',
                                  'dead', 'zombie', 'user', 'access', 'capability', 'profile', 'principal',
                                  'secret', 'generation', 'epoch', 'ancestry'])
@pytest.mark.asyncio
async def test_identity_drift_fix_keeps_rejections(tmp_path, monkeypatch, worker, fault):
    from gateway.session_contract import Principal, SessionRef
    from gateway.session_worker import worker_request
    from gateway.session_worker_reservation import reserve_admission_worker
    child, hello = worker
    with closing(SessionDB(tmp_path / 'state.db')) as db:
        authority, row = _authority(tmp_path, db)
        if fault == 'ancestry':
            with pytest.raises(RuntimeStoreError, match='permission_denied'):
                reserve_admission_worker(authority, admission_id=row['admission_id'], process=child,
                    principal_id='human', hello={**hello, 'pid': os.getppid()})
            return
        if fault == 'legacy-drift':
            hello['birth'] = psutil.Process(child.pid).create_time()
        scope = reserve_admission_worker(authority, admission_id=row['admission_id'], process=child,
                                         principal_id='human', hello=hello)
        actor = Principal('human', str(tmp_path), frozenset({'worker:adopt'}), 'ticket')
        connection = SimpleNamespace(authority=authority, actor=actor)
        ref = SessionRef(str(tmp_path), 'owned')
        identity = {k: v for k, v in scope.items() if k != 'epoch'}
        await worker_request(connection, ref, identity, operation='adopt')
        params = scope | dict(sequence=1, operation='session.title', payload={'title': 'must not persist', 'source': 'user'})
        import hermes_process_identity

        def stop():
            child.terminate()
            child.wait(timeout=10)

        def denied(self):
            raise psutil.AccessDenied(child.pid)

        original_read = Path.read_text

        def unreadable_boot(path, *args, **kwargs):
            if str(path) == '/proc/sys/kernel/random/boot_id':
                raise PermissionError('unreadable boot ID')
            return original_read(path, *args, **kwargs)

        original_user = psutil.Process.username
        damage = {
            'ticks': (lambda: params.update(birth=params['birth'].rsplit(':', 1)[0] + ':' + str(int(params['birth'].rsplit(':', 1)[1]) + 1)), 'permission_denied'),
            'boot': (lambda: params.update(birth='linux-v1:00000000-0000-0000-0000-000000000000:' + params['birth'].rsplit(':', 1)[1]), 'permission_denied'),
            'malformed': (lambda: params.update(birth='linux-v99:unknown:1'), 'invalid_params'),
            'unreadable': (lambda: monkeypatch.setattr(hermes_process_identity, '_proc_start_ticks', lambda pid: None), 'permission_denied'),
            'boot-unreadable': (lambda: monkeypatch.setattr(Path, 'read_text', unreadable_boot), 'permission_denied'),
            'legacy-drift': (lambda: _shift_clock(monkeypatch), 'permission_denied'),
            'dead': (stop, 'worker_not_live'),
            'zombie': (lambda: monkeypatch.setattr(psutil.Process, 'status', lambda self: psutil.STATUS_ZOMBIE), 'permission_denied'),
            'user': (lambda: monkeypatch.setattr(psutil.Process, 'username', lambda self: 'foreign-user' if self.pid == child.pid else original_user(self)), 'permission_denied'),
            'access': (lambda: monkeypatch.setattr(psutil.Process, 'username', denied), 'worker_not_live'),
            'capability': (lambda: setattr(connection, 'actor', Principal('human', str(tmp_path), frozenset(), 'ticket')), 'permission_denied'),
            'profile': (lambda: params.update(profile_id='foreign-profile'), 'profile_mismatch'),
            'principal': (lambda: setattr(connection, 'actor', Principal('other', str(tmp_path), frozenset({'worker:adopt'}), 'ticket')), 'permission_denied'),
            'secret': (lambda: params.update(secret='foreign-secret-000000'), 'permission_denied'),
            'generation': (lambda: params.update(generation=params['generation'] + 1), 'stale_generation'),
            'epoch': (lambda: params.update(epoch='foreign-epoch'), 'stale_epoch'),
        }
        mutate, error = damage[fault]
        mutate()
        with pytest.raises(RuntimeStoreError, match=error):
            await worker_request(connection, ref, params, operation='persist')
        assert db.get_session('owned')['title'] is None
        assert db._read_one('SELECT COUNT(*) FROM worker_receipts')[0] == 0
