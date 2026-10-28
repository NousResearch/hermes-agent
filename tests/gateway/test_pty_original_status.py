"""Native Stop preserves a real unit PTY's original payload and termination status."""
import asyncio
import json
import shlex
import signal
import sys

import psutil
import pytest

from hermes_state_runtime import RuntimeStoreError, list_session_admissions
from tests.gateway.test_session_authority_cancel_settlement import ACTOR, REF, _authority, _submit
from tests.waiting import async_wait_until


async def _spawn(registry, root, command, *, parent='s', pty=False):
    from gateway.session_context import clear_session_vars, set_session_vars
    tokens = set_session_vars(session_id=parent, session_key='route')
    try:
        return await asyncio.to_thread(
            registry.spawn_local, command, cwd=str(root), task_id='shared-worker',
            owner_task_id='shared-worker', session_key='route', use_pty=pty)
    finally:
        clear_session_vars(tokens)


def _alive(process):
    try:
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


@pytest.mark.platforms('linux')
@pytest.mark.asyncio
@pytest.mark.parametrize('exit_code', [0, 7, -signal.SIGKILL, -signal.SIGTERM],
                         ids=['exit-zero', 'exit-seven', 'sigkill', 'sigterm'])
async def test_unit_pty_status_survives_refused_stop_and_native_fifo(tmp_path, monkeypatch, exit_code):
    from gateway.session_ancillary import process_stop_pending
    from tools import process_registry as module
    from tools.process_registry_job import job_quiescent, request_job_stop

    db, authority = _authority(tmp_path, monkeypatch)
    monkeypatch.setattr(module, 'CHECKPOINT_PATH', tmp_path / 'processes.json')
    monkeypatch.setattr(module, 'worker_output_log', lambda sid: None)
    registry = module.ProcessRegistry()
    monkeypatch.setattr(module, 'process_registry', registry)

    neighbor_pid = tmp_path / 'foreign-payload.pid'
    neighbor_pid_temp = neighbor_pid.with_suffix('.tmp')
    sleeper = ('import os,threading; from pathlib import Path; '
               f'Path({str(neighbor_pid_temp)!r}).write_text(str(os.getpid())); '
               f'os.replace({str(neighbor_pid_temp)!r}, {str(neighbor_pid)!r}); '
               'threading.Event().wait(30)')
    neighbor = await _spawn(registry, tmp_path, 'exec ' + shlex.join([sys.executable, '-c', sleeper]),
                            parent='foreign-parent')
    own, foreign = None, None
    allowed, attempts, executed = False, [], []

    def native_unit(session, command, suffix, label):
        session.systemd_unit = 'hermes-worker-' + suffix + '.scope'
        return ['/bin/bash', '-c', command]

    def stop_empty(unit):
        assert unit == unit_name
        attempts.append(unit)
        if allowed:
            assert not own._pty.isalive(), 'affirmative unit verdict requires actual PTY death'
            return True
        return False

    expected = 'ORIGINAL PTY payload: λ\r\nEND OF ORIGINAL\r\n'
    path = None

    async def execute(owner, ref, row):
        retained = json.loads(path.read_text())
        assert job_quiescent(own) and not own._pty.isalive()
        assert own._result_complete and retained['result_complete'] is True
        assert own.exit_code == retained['exit_code'] == exit_code
        assert own.output_buffer == retained['output'] == expected
        assert _alive(foreign) and neighbor.process.poll() is None
        executed.append(row['request_id'])
        return 'one human continuation'

    monkeypatch.setattr('gateway.session_finite.execute_finite_admission', execute)
    try:
        await async_wait_until(neighbor_pid.exists, timeout=10)
        foreign = psutil.Process(int(neighbor_pid.read_text()))
        assert neighbor.native_job_scope and not neighbor.systemd_unit and _alive(foreign)
        monkeypatch.setattr(registry, '_scope_argv', native_unit)
        payload = ('import os,signal,sys; '
                   'sys.stdout.write("ORIGINAL PTY payload: λ\\nEND OF ORIGINAL\\n"); sys.stdout.flush(); '
                   + (f'raise SystemExit({exit_code})' if exit_code >= 0
                      else f'os.kill(os.getpid(), {-exit_code})'))
        own = await _spawn(registry, tmp_path, 'exec ' + shlex.join([sys.executable, '-c', payload]), pty=True)
        unit_name = own.systemd_unit
        monkeypatch.setattr(module, '_stop_systemd_unit', stop_empty)
        assert own.parent_session_id == 's' and neighbor.parent_session_id == 'foreign-parent'
        assert own.owner_task_id == neighbor.owner_task_id and own.session_key == neighbor.session_key
        assert own._pty is not None and unit_name.startswith('hermes-worker-')
        assert await asyncio.to_thread(own._completion_event.wait, 10)
        assert not own._pty.isalive()
        if exit_code < 0:
            assert own._pty.exitstatus is None and own._pty.signalstatus == -exit_code
        else:
            assert own._pty.exitstatus == exit_code and own._pty.signalstatus is None
        path = tmp_path / 'logs' / 'process-results' / (own.id + '.json')
        original = json.loads(path.read_text())
        assert own.exit_code == original['exit_code'] == exit_code
        assert own.output_buffer == original['output'] == expected
        assert original['result_complete'] is True

        stopped_generation = authority._handle(REF).execution_generation
        with pytest.raises(RuntimeStoreError, match='process_stop_unknown'):
            await authority.interrupt(ACTOR, REF, stopped_generation)
        marker = process_stop_pending(db, 's')
        assert marker and _alive(foreign)
        await _submit(authority, 'only-human')
        await asyncio.wait_for(authority.sessions['s'].task, 10)
        assert executed == [] and process_stop_pending(db, 's') == marker
        refused = json.loads(path.read_text())
        assert refused['exit_code'] == exit_code and refused['output'] == expected
        assert refused['result_complete'] is True and attempts

        allowed = True
        await authority.interrupt(ACTOR, REF, authority._handle(REF).execution_generation)
        await async_wait_until(lambda: executed == ['only-human'] and authority.sessions['s'].task.done(), timeout=10)
        assert process_stop_pending(db, 's') is None and _alive(foreign)
        final = json.loads(path.read_text())
        assert final['exit_code'] == exit_code and final['output'] == expected
        assert final['result_complete'] is True and final['completion_consumed'] is True
        assert own.id not in {row['session_id'] for row in json.loads(module._checkpoint_path().read_text())}
        rows = [row for row in list_session_admissions(db, session_id='s', pending_only=False)
                if row['principal_id'] != 'native-process-stop']
        assert [(row['request_id'], row['status'], row['outcome']) for row in rows] == [
            ('only-human', 'terminal', 'completed')]

        previous_attempts = list(attempts)
        with pytest.raises(RuntimeStoreError, match='stale_generation'):
            await authority.interrupt(ACTOR, REF, stopped_generation)
        assert attempts == previous_attempts and executed == ['only-human']
        assert process_stop_pending(db, 's') is None and _alive(foreign)
        assert neighbor.process.poll() is None
    finally:
        await asyncio.to_thread(request_job_stop, registry, neighbor)
        assert await asyncio.to_thread(neighbor._completion_event.wait, 10)
        if foreign is not None:
            assert not _alive(foreign)
        if own is not None:
            allowed = True
            await asyncio.to_thread(registry.kill_process, own.id)
            assert await asyncio.to_thread(own._completion_event.wait, 10)
        db.close()
