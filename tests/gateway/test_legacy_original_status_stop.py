"""A legacy complete flag cannot invent original status after process-cache loss."""
import asyncio
import json
import shlex
import signal
import subprocess
import sys

import pytest

from tests.gateway.test_session_authority_cancel_settlement import ACTOR, REF, _authority, _submit
from tests.waiting import async_wait_until

pytestmark = pytest.mark.platforms('linux')


@pytest.mark.parametrize(('captured_status', 'expected'), [('', None), ('unreadable-status', None), ('7', 7)],
                         ids=['missing-backend-status', 'malformed-backend-status', 'actual-backend-status'])
def test_backend_status_source_cannot_fabricate_a_numeric_exit(tmp_path, monkeypatch, captured_status, expected):
    from tests.gateway.test_unknown_backend_stop_checkpoint import ShellBackend
    from tools import process_registry as module
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setenv('HOME', str(tmp_path))
    backend = ShellBackend(tmp_path)
    registry = module.ProcessRegistry()
    log, pid, exited = (tmp_path / name for name in ('original.log', 'original.pid', 'original.exit'))
    with log.open('wb') as output:
        actual = subprocess.Popen([sys.executable, '-c',
            'print("BACKEND ORIGINAL PAYLOAD",flush=True); raise SystemExit(7)'], stdout=output)
        assert actual.wait(5) == 7
    pid.write_text(str(actual.pid))
    exited.write_text(captured_status)
    own = registry._new_session('actual finite backend', 'worker', 'worker', 'route', str(tmp_path),
                                env_ref=backend, pid_scope='sandbox')
    own.pid = actual.pid
    registry._running[own.id] = own
    registry._env_poller_loop(own, backend, str(log), str(pid), str(exited))
    assert own._completion_event.is_set() and own.exit_code == expected
    assert own.output_buffer == 'BACKEND ORIGINAL PAYLOAD\n'
    saved = json.loads((tmp_path / 'logs' / 'process-results' / (own.id + '.json')).read_text())
    assert saved['exit_code'] == expected and saved['result_complete'] is (expected is not None)
    if expected is None:
        assert not own._result_complete and own._result_persisted_generation == -1
    # Original status/output do not add a new opaque-backend cleanup capability.
    verdict = registry.kill_process(own.id)
    assert (verdict['status'], verdict['error']) == ('error', 'process_stop_unknown')
    assert own.id in {row['session_id'] for row in json.loads(module._checkpoint_path().read_text())}


@pytest.mark.asyncio
async def test_live_missing_status_cannot_borrow_unit_quiescence_status(tmp_path, monkeypatch):
    from gateway.session_ancillary import process_stop_pending
    from gateway.session_context import clear_session_vars, set_session_vars
    from hermes_state_runtime import RuntimeStoreError
    from tools import process_registry as module
    from tools.process_registry_job import job_quiescent, read_job_receipt
    from tools.process_registry_results import save_completed_result

    db, authority = _authority(tmp_path, monkeypatch)
    monkeypatch.setenv('HOME', str(tmp_path))
    registry = module.ProcessRegistry()
    monkeypatch.setattr(module, 'process_registry', registry)
    monkeypatch.setattr(module, 'worker_output_log', lambda _sid: None)
    finish = registry._finish_exited
    missing = True
    def unavailable_status(session, code, *args, **kwargs):
        return finish(session, None if missing else code, *args, **kwargs)
    monkeypatch.setattr(registry, '_finish_exited', unavailable_status)
    def native_unit(session, command, suffix, label):
        session.systemd_unit = 'hermes-worker-' + suffix + '.scope'
        return ['/bin/bash', '-c', command]
    monkeypatch.setattr(registry, '_scope_argv', native_unit)
    tokens = set_session_vars(session_id='s', session_key='route')
    try:
        own = registry.spawn_local('exec ' + shlex.join([sys.executable, '-c',
            'import os,signal; print("ACTUAL ORIGINAL PAYLOAD",flush=True); os.kill(os.getpid(),signal.SIGKILL)']),
            cwd=str(tmp_path), use_pty=True, task_id='worker', session_key='route')
    finally:
        clear_session_vars(tokens)
    try:
        assert await asyncio.to_thread(own._completion_event.wait, 10)
        # Only collection was faulted; this is a real reaped PTY and actual signal9.
        assert own._pty.exitstatus is None and own._pty.signalstatus == signal.SIGKILL
        assert own.exit_code is None and not own._result_complete and own._result_persisted_generation == -1
        unit = own.systemd_unit
        def actual_empty(name):
            assert name == unit and not own._pty.isalive()
            return True
        monkeypatch.setattr(module, '_stop_systemd_unit', actual_empty)
        result_path = tmp_path / 'logs' / 'process-results' / (own.id + '.json')
        executed = []
        async def execute(owner, ref, row):
            retained = json.loads(result_path.read_text())
            assert own._result_complete and own.exit_code == retained['exit_code'] == -signal.SIGKILL
            assert retained['result_complete'] and retained['output'] == own.output_buffer
            assert own.output_buffer == 'ACTUAL ORIGINAL PAYLOAD\r\n'
            executed.append(row['request_id'])
            return 'continued once'
        monkeypatch.setattr('gateway.session_finite.execute_finite_admission', execute)
        with pytest.raises(RuntimeStoreError, match='process_stop_unknown'):
            await authority.interrupt(ACTOR, REF, 0)
        assert job_quiescent(own) and read_job_receipt(own)['exit_code'] is None
        assert own.systemd_unit == unit and own._scope_stop_pending and process_stop_pending(db, 's')
        await _submit(authority, 'after-actual-status-recovery')
        await asyncio.wait_for(authority.sessions['s'].task, 10)
        assert executed == [] and process_stop_pending(db, 's')
        saved = json.loads(result_path.read_text())
        assert saved['exit_code'] is None and not saved['result_complete']
        assert own.id in {row['session_id'] for row in json.loads(module._checkpoint_path().read_text())}
        # Restore the genuine collector source, then publish its actual code and
        # already-collected original body through existing native result machinery.
        missing = False
        code = -own._pty.signalstatus  # Actual signal status from this same original PTY.
        assert code == -signal.SIGKILL
        registry._finish_exited(own, code)
        assert save_completed_result(own)
        await async_wait_until(lambda: executed == ['after-actual-status-recovery']
                               and authority.sessions['s'].task.done(), timeout=10)
        assert process_stop_pending(db, 's') is None
        assert json.loads(result_path.read_text())['exit_code'] == -signal.SIGKILL
    finally:
        assert not own._pty.isalive()
        db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('checkpoint_exists', [False, True], ids=['no-checkpoint-file', 'empty-checkpoint'])
@pytest.mark.parametrize(('invalid_code', 'reason'), [(None, 'exited'), (False, 'exited'),
                                                    ('7', 'exited'), (-1, 'lost')],
                         ids=['missing', 'bool', 'string', 'lost'])
async def test_cold_legacy_status_stays_fenced_until_original_result_restored(
        tmp_path, monkeypatch, checkpoint_exists, invalid_code, reason):
    from gateway.session_ancillary import process_stop_pending
    from gateway.session_context import clear_session_vars, set_session_vars
    from hermes_state_runtime import RuntimeStoreError, list_session_admissions
    from tools import process_registry as module
    from tools.process_registry_job import job_quiescent, read_job_receipt
    from tools.process_registry_results import load_completed_results, save_completed_result
    from utils import atomic_json_write

    db, authority = _authority(tmp_path, monkeypatch)
    monkeypatch.setenv('HOME', str(tmp_path))
    original = module.ProcessRegistry()
    monkeypatch.setattr(module, 'process_registry', original)
    monkeypatch.setattr(module, 'worker_output_log', lambda _sid: None)

    def unit_launch(session, command, suffix, label):
        session.systemd_unit = 'hermes-worker-' + suffix + '.scope'
        return ['/bin/bash', '-c', command]
    monkeypatch.setattr(original, '_scope_argv', unit_launch)
    tokens = set_session_vars(session_id='s', session_key='route')
    try:
        own = original.spawn_local('exec ' + shlex.join([sys.executable, '-c',
            'print("ORIGINAL COMPLETE PAYLOAD",flush=True); raise SystemExit(7)']),
            cwd=str(tmp_path), use_pty=True, task_id='worker', owner_task_id='worker', session_key='route')
    finally:
        clear_session_vars(tokens)
    try:
        assert await asyncio.to_thread(own._completion_event.wait, 10)
        assert own._pty.exitstatus == 7 and own.exit_code == 7
        unit = own.systemd_unit
        def actual_empty(name):
            assert name == unit and not own._pty.isalive()
            return True  # The actual finite original process has exited and has no children.
        monkeypatch.setattr(module, '_stop_systemd_unit', actual_empty)
        assert original.kill_process(own.id)['status'] == 'already_exited'
        assert save_completed_result(own)  # Publish the actual discharged in-memory unit binding too.
        result_path = tmp_path / 'logs' / 'process-results' / (own.id + '.json')
        genuine_bytes = result_path.read_bytes()
        genuine = json.loads(genuine_bytes)
        assert genuine['result_complete'] and genuine['exit_code'] == 7 and not genuine['systemd_unit']
        checkpoint = module._checkpoint_path()
        assert json.loads(checkpoint.read_text()) == []
        invalid = {**genuine, 'exit_code': invalid_code, 'completion_reason': reason, 'result_complete': True}
        atomic_json_write(result_path, invalid)
        # Preserve this old invalid record as evidence; it is not a complete-status proof.
        (tmp_path / 'legacy-invalid-result.json').write_text(json.dumps(invalid))
        if not checkpoint_exists:
            checkpoint.unlink()
        current = module.ProcessRegistry()
        restored = load_completed_results(owns_entry=lambda row: row['id'] == own.id, strict=True)[own.id]
        assert restored.exit_code == invalid_code and restored.completion_reason == reason
        assert not restored._result_complete and restored._result_persisted_generation == -1
        assert restored._scope_stop_pending and restored._completion_event.is_set()
        assert job_quiescent(restored) and read_job_receipt(restored)['unit'] == unit
        current._finished[restored.id] = restored
        monkeypatch.setattr(module, 'process_registry', current)
        stale = load_completed_results(owns_entry=lambda row: row['id'] == own.id, strict=True)[own.id]
        executed = []
        async def execute(owner, ref, row):
            actual = current.get(own.id)
            retained = json.loads(result_path.read_text())
            assert actual._completion_event.is_set() and actual._result_complete and job_quiescent(actual)
            assert actual.exit_code == retained['exit_code'] == 7 and retained['result_complete']
            assert actual.output_buffer == retained['output'] == genuine['output']
            executed.append(row['request_id'])
            return 'continued once'
        monkeypatch.setattr('gateway.session_finite.execute_finite_admission', execute)
        with pytest.raises(RuntimeStoreError, match='process_stop_unknown'):
            await authority.interrupt(ACTOR, REF, authority._handle(REF).execution_generation)
        marker = process_stop_pending(db, 's')
        assert marker is not None
        await _submit(authority, 'after-genuine-original-status')
        await asyncio.wait_for(authority.sessions['s'].task, 10)
        assert executed == [] and process_stop_pending(db, 's') == marker
        pending = current.get(own.id)
        assert pending._scope_stop_pending and pending._result_persisted_generation == -1
        assert own.id in {row['session_id'] for row in json.loads(checkpoint.read_text())}
        assert not json.loads(result_path.read_text())['result_complete']
        # These bytes came from the actual original collector and original successful
        # unit cleanup above. Unit emptiness alone never supplied their primary status.
        result_path.write_bytes(genuine_bytes)
        await authority.interrupt(ACTOR, REF, authority._handle(REF).execution_generation)
        await async_wait_until(lambda: executed == ['after-genuine-original-status']
                               and authority.sessions['s'].task.done(), timeout=10)
        assert process_stop_pending(db, 's') is None
        # A cached invalid writer must inherit the genuine same-job complete result,
        # including its current cancellation receipt, rather than downgrade it again.
        assert save_completed_result(stale)
        saved = json.loads(result_path.read_text())
        assert saved['result_complete'] and saved['exit_code'] == 7
        assert saved['output'] == genuine['output'] and saved['completion_consumed']
        assert stale._result_complete and stale.exit_code == 7
        assert json.loads(checkpoint.read_text()) == []
        with pytest.raises(RuntimeStoreError, match='stale_generation'):
            await authority.interrupt(ACTOR, REF, 0)
        assert process_stop_pending(db, 's') is None and executed == ['after-genuine-original-status']
        work = [row for row in list_session_admissions(db, session_id='s', pending_only=False)
                if row['principal_id'] != 'native-process-stop']
        assert [(row['request_id'], row['status'], row['outcome']) for row in work] == [
            ('after-genuine-original-status', 'terminal', 'completed')]
    finally:
        assert not own._pty.isalive()
        db.close()
