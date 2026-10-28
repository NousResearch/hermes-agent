"""Ordinary startup cannot turn an unproved PID into native Stop permission."""
import asyncio
import json
import subprocess
import sys

import pytest

from hermes_state_runtime import list_session_admissions
from tests.gateway.test_session_authority_cancel_settlement import REF, _authority, _submit


@pytest.mark.asyncio
@pytest.mark.parametrize('hint', ['plain', 'recorded-log-exit', 'unit'])
async def test_missing_birth_survives_ordinary_recovery_and_keeps_fifo_fenced(tmp_path, monkeypatch, hint):
    from gateway.session_ancillary import process_snapshot, process_stop_pending, record_process_stop, retry_process_stop
    from tools import process_registry as module
    from tools.process_registry_output_log import output_dir
    db, authority = _authority(tmp_path, monkeypatch)
    neighbor = await asyncio.to_thread(subprocess.Popen,
        [sys.executable, '-c', 'import threading; threading.Event().wait(120)'])
    registry = module.ProcessRegistry()
    monkeypatch.setattr(module, 'process_registry', registry)
    entry = dict(session_id='proc_unproved', command='stale unproved binding', pid=neighbor.pid,
                 pid_scope='host', host_start_time=None, task_id='s', owner_task_id='s',
                 parent_session_id='s', session_key='route')
    if hint == 'recorded-log-exit':
        logs = output_dir()
        logs.mkdir(parents=True, exist_ok=True)
        log = logs / 'proc_unproved.log'
        log.write_text('old wrapper output\n')
        log.with_suffix('.exit').write_text('0\n')
        entry['output_log'] = str(log)
    elif hint == 'unit':
        entry['systemd_unit'] = 'hermes-worker-unproved.scope'
    checkpoint = module._checkpoint_path()
    checkpoint.write_text(json.dumps([entry]))
    record_process_stop(authority, REF, 0)
    token = process_stop_pending(db, 's')
    executed = []

    async def execute(owner, ref, row):
        executed.append(row['request_id'])
        return 'finite fixture answer'

    monkeypatch.setattr('gateway.session_finite.execute_finite_admission', execute)
    monkeypatch.setattr(registry, '_safe_host_start_time',
                        lambda *a: pytest.fail('unproved metadata was rebound to the live PID'))
    monkeypatch.setattr(registry, '_terminate_host_pid',
                        lambda *a, **kw: pytest.fail('unproved PID was signaled'))
    monkeypatch.setattr(module, '_stop_systemd_unit',
                        lambda *a: pytest.fail('a unit hint bypassed unknown PID identity'))
    try:
        assert not await retry_process_stop(authority, REF)
        # This is the actual ordinary startup call, without the native Stop
        # recovery predicate. It must not manufacture authority before retry.
        assert registry.recover_from_checkpoint() == 0
        assert registry.get(entry['session_id']) is None
        assert json.loads(checkpoint.read_text()) == [entry]
        assert process_snapshot(authority, REF, None, ()) == []
        assert module.ProcessRegistry()._write_checkpoint()
        assert json.loads(checkpoint.read_text()) == [entry]
        assert not await retry_process_stop(authority, REF)
        assert process_stop_pending(db, 's') == token and neighbor.poll() is None
        await _submit(authority, 'held human continuation')
        await asyncio.wait_for(authority.sessions['s'].task, 10)
        assert executed == [] and neighbor.poll() is None
        assert list_session_admissions(db, session_id='s')[0]['status'] == 'queued'
        assert json.loads(checkpoint.read_text()) == [entry]
        assert process_stop_pending(db, 's') == token
    finally:
        if neighbor.poll() is None:
            neighbor.kill()
        neighbor.wait(timeout=10)
        db.close()
