"""Physical quiescence cannot substitute for the original complete output payload."""
import asyncio
import json
import os
from pathlib import Path
import shlex
import sys

import pytest

from tests.gateway.test_session_authority_cancel_settlement import ACTOR, REF, _authority, _submit
from tests.waiting import async_wait_until

pytestmark = pytest.mark.platforms('linux')


@pytest.mark.asyncio
@pytest.mark.parametrize('retained', [False, True], ids=['unretained', 'complete-retained'])
@pytest.mark.parametrize('fault', ['permission', 'missing', 'truncated', 'replaced', 'missing-extent'])
async def test_cold_output_failure_stays_fenced_until_real_payload_is_available(tmp_path, monkeypatch, retained, fault):
    if fault == 'permission' and getattr(os, 'geteuid', lambda: 0)() == 0:
        pytest.skip('Actual permission denial requires a non-root test user')
    from gateway.session_ancillary import process_stop_pending
    from gateway.session_context import clear_session_vars, set_session_vars
    from hermes_state_runtime import RuntimeStoreError
    from tools import process_registry as module, process_registry_results as results
    from tools.process_registry_job import job_receipt_path, job_quiescent, read_job_receipt
    from utils import atomic_json_write
    db, authority = _authority(tmp_path, monkeypatch)
    original = module.ProcessRegistry()
    monkeypatch.setattr(module, 'process_registry', original)
    write = results.atomic_json_write
    def refused(path, *args, **kwargs):
        if Path(path).parent.name == 'process-results':
            raise PermissionError('finite original-result write refusal')
        return write(path, *args, **kwargs)
    if not retained:
        monkeypatch.setattr(results, 'atomic_json_write', refused)
    tokens = set_session_vars(session_id='foreign-parent', session_key='route')
    try:
        neighbor = original.spawn_local(shlex.join([sys.executable, '-c',
            'import threading; threading.Event().wait(30)']), cwd=str(tmp_path), task_id='worker', session_key='route')
    finally:
        clear_session_vars(tokens)
    effect = tmp_path / 'command-runs'
    payload = f'from pathlib import Path; Path({str(effect)!r}).open("a").write("once\\n"); print("known original output",flush=True); raise SystemExit(7)'
    tokens = set_session_vars(session_id='s', session_key='route')
    try:
        own = original.spawn_local('exec ' + shlex.join([sys.executable, '-c', payload]),
            cwd=str(tmp_path), task_id='worker', session_key='route')
    finally:
        clear_session_vars(tokens)
    executed = []
    async def execute(owner, ref, row):
        current = module.process_registry.get(own.id)
        assert job_quiescent(current) and current._result_complete
        assert current._result_persisted_generation == current.ownership_generation
        assert current.output_buffer.count('known original output') == 1
        executed.append(row['request_id'])
        return 'continued'
    monkeypatch.setattr('gateway.session_finite.execute_finite_admission', execute)
    path, backup = Path(own.output_log), tmp_path / 'original-log'
    receipt_path = job_receipt_path(own.id)
    row = None
    try:
        assert await asyncio.to_thread(own._completion_event.wait, 10)
        assert own.exit_code == 7 and job_quiescent(own)
        row = read_job_receipt(own)
        source = path.read_bytes()
        assert row['stdout']['bytes'] == len(source) > 0
        assert row['stdout']['inode'] == path.stat().st_ino
        monkeypatch.setattr(results, 'atomic_json_write', refused)
        if retained:
            result = original.kill_process(own.id)
            assert (result['status'], result['error']) == ('error', 'process_stop_unknown')
        with pytest.raises(RuntimeStoreError, match='process_stop_unknown'):
            await authority.interrupt(ACTOR, REF, 0)
        await _submit(authority, 'queued-human')
        await asyncio.wait_for(authority.sessions['s'].task, 10)
        monkeypatch.setattr(results, 'atomic_json_write', write)
        if fault == 'permission':
            path.chmod(0)
        elif fault in {'missing', 'replaced'}:
            os.replace(path, backup)
            if fault == 'replaced':
                path.write_bytes(source)  # Same text/length, a foreign inode.
        elif fault == 'truncated':
            path.write_bytes(source[:3])
        else:
            atomic_json_write(receipt_path, {key: value for key, value in row.items() if key != 'stdout'}, mode=0o600)
        cold = module.ProcessRegistry()
        monkeypatch.setattr(module, 'process_registry', cold)
        assert cold.recover_from_checkpoint() >= 1
        current = cold.get(own.id)
        assert await asyncio.to_thread(current._completion_event.wait, 10)
        assert current.exit_code == 7 and job_quiescent(current)
        if not retained:
            assert not current._result_complete and current._result_persisted_generation == -1
            assert current._scope_stop_pending and process_stop_pending(db, 's')
            assert not executed
            with pytest.raises(RuntimeStoreError, match='process_stop_unknown'):
                await authority.interrupt(ACTOR, REF, 0)
            restored = results.load_completed_results(owns_entry=lambda saved: saved['id'] == own.id)[own.id]
            assert not restored._result_complete and restored._result_persisted_generation == -1
            assert restored._scope_stop_pending
            assert any(saved['session_id'] == own.id for saved in json.loads(module._checkpoint_path().read_text()))
            if fault == 'permission':
                path.chmod(0o600)
            if backup.exists():
                path.unlink(missing_ok=True)
                os.replace(backup, path)
            elif fault == 'truncated':
                path.write_bytes(source)
            if fault == 'missing-extent':
                atomic_json_write(receipt_path, row, mode=0o600)
        else:
            # A retained complete payload is real evidence even when its source is now unreadable.
            assert current._result_complete and current.output_buffer.count('known original output') == 1
        authority._schedule(REF)  # Existing native recovery pump, no new admission or command.
        await async_wait_until(lambda: executed == ['queued-human'] and authority.sessions['s'].task.done(), timeout=10)
        assert process_stop_pending(db, 's') is None
        assert effect.read_text() == 'once\n'
        assert not neighbor.exited
    finally:
        if path.exists():
            path.chmod(0o600)
        monkeypatch.setattr(results, 'atomic_json_write', write)
        if row is not None:
            atomic_json_write(receipt_path, row, mode=0o600)
        await asyncio.to_thread(original.kill_process, neighbor.id)
        db.close()
