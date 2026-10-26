"""Unrelated checkpoint writes cannot release Stop before backend work finishes."""
import asyncio
import json
import shlex
import subprocess
import sys
import tempfile

import psutil
import pytest

from hermes_state_runtime import RuntimeStoreError, list_session_admissions
from tests.gateway.test_session_authority_cancel_settlement import ACTOR, REF, _authority, _submit
from tests.waiting import wait_until


class ShellBackend:
    """Real shell transport implementing the native environment backend contract."""
    def __init__(self, root):
        self.root = root

    def get_temp_dir(self):
        return str(self.root)

    def execute(self, command, cwd='', timeout=10, **kwargs):
        with tempfile.TemporaryFile(mode='w+') as output:
            result = subprocess.run(['bash', '-c', command], cwd=cwd or str(self.root),
                                    stdout=output, stderr=subprocess.STDOUT, timeout=timeout, check=False)
            output.seek(0)
            return {'output': output.read(), 'returncode': result.returncode}


@pytest.mark.asyncio
@pytest.mark.live_system_guard_bypass  # Backend wrapper detaches only these captured finite fixture children.
async def test_lost_backend_handle_keeps_fifo_fenced_through_foreign_writes_after_wrapper_exit(tmp_path, monkeypatch):
    from gateway.session_ancillary import process_stop_pending
    from tools import process_registry as module
    db, authority = _authority(tmp_path, monkeypatch)
    backend = ShellBackend(tmp_path)
    previous = module.ProcessRegistry()
    ready, release = tmp_path / 'ready', tmp_path / 'release'
    code = (f'from pathlib import Path; import time; p=Path({str(release)!r}); '
            f'Path({str(ready)!r}).touch(); deadline=time.monotonic()+120\n'
            'while not p.exists() and time.monotonic()<deadline: time.sleep(.01)\n'
            'print("original backend output", flush=True)')
    process = previous.spawn_via_env(backend, 'exec ' + shlex.join([sys.executable, '-c', code]),
        cwd=str(tmp_path), owner_task_id='s', task_id='s', session_key='route', persist_on_release=True)
    wait_until(ready.exists, timeout=15)
    tree = [psutil.Process(process.pid), *psutil.Process(process.pid).children(recursive=True)]
    current = module.ProcessRegistry()
    monkeypatch.setattr(module, 'process_registry', current)
    executed = []

    def alive():
        for child in tree:
            try:
                if child.is_running() and child.status() != psutil.STATUS_ZOMBIE:
                    return True
            except psutil.NoSuchProcess:
                pass
        return False

    async def execute(owner, ref, row):
        assert not alive(), 'successor executed while the prior backend job was alive'
        executed.append(row['request_id'])
        return 'finite fixture answer'

    monkeypatch.setattr('gateway.session_finite.execute_finite_admission', execute)
    try:
        with pytest.raises(RuntimeStoreError, match='process_stop_unknown'):
            await authority.interrupt(ACTOR, REF, authority._handle(REF).execution_generation)
        token = process_stop_pending(db, 's')
        assert token is not None and alive()
        before = json.loads(module._checkpoint_path().read_text())
        assert any(row['session_id'] == process.id for row in before)
        # Exercise both an empty native table rewrite and its ordinary recovery.
        for writer in (module.ProcessRegistry(), current):
            assert writer._write_checkpoint()
            assert writer.recover_from_checkpoint() == 0
            assert json.loads(module._checkpoint_path().read_text()) == before
        await _submit(authority, 'first human continuation')
        await asyncio.wait_for(authority.sessions['s'].task, 10)
        assert not executed and alive() and process_stop_pending(db, 's') == token
        assert list_session_admissions(db, session_id='s')[0]['status'] == 'queued'
        # The original backend still has the real resource/exit producer.
        # Its wrapper result is history; it cannot prove descendant cleanup.
        release.touch()
        assert await asyncio.to_thread(process._completion_event.wait, 15)
        await asyncio.to_thread(wait_until, lambda: not alive(), timeout=10)
        assert process.exit_code == 0 and 'original backend output' in process.output_buffer
        assert process.id in {row['session_id'] for row in json.loads(module._checkpoint_path().read_text())}
        await _submit(authority, 'second human continuation')
        await asyncio.wait_for(authority.sessions['s'].task, 15)
        assert executed == []
        assert process_stop_pending(db, 's') == token
        from tools.process_registry_results import load_completed_results
        retained = load_completed_results(owns_entry=lambda row: row['id'] == process.id, strict=True)[process.id]
        assert retained.exit_code == 0 and 'original backend output' in retained.output_buffer
    finally:
        release.touch()
        for child in reversed(tree):
            try:
                if child.is_running() and child.status() != psutil.STATUS_ZOMBIE:
                    child.kill()
            except psutil.NoSuchProcess:
                pass
        for child in tree:
            try:
                child.wait(timeout=5)
            except (psutil.NoSuchProcess, psutil.TimeoutExpired):
                pass
        await asyncio.to_thread(process._completion_event.wait, 10)
        db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('binding', ['sandbox', 'opaque-other', 'wsl-chain', 'missing-birth', 'missing-pid'])
async def test_unproved_resource_survives_rewrite_and_recovery_without_signaling_host(tmp_path, monkeypatch, binding):
    from gateway.session_ancillary import process_stop_pending, record_process_stop, retry_process_stop
    from tools import process_registry as module
    db, authority = _authority(tmp_path, monkeypatch)
    registry = module.ProcessRegistry()
    monkeypatch.setattr(module, 'process_registry', registry)
    entry = dict(session_id='proc_unknown', command='opaque native resource', task_id='s',
                 owner_task_id='s', parent_session_id='s', session_key='route',
                 pid=1, pid_scope='host', host_start_time=None)
    if binding in ('sandbox', 'opaque-other'):
        entry.update(pid_scope=binding, host_start_time=1)
    elif binding == 'wsl-chain':
        entry.update(wsl_chain=True, host_start_time=1)
    elif binding == 'missing-pid':
        entry.update(pid=None, host_start_time=1)
    path = module._checkpoint_path()
    path.write_text(json.dumps([entry]))
    record_process_stop(authority, REF, 0)
    token = process_stop_pending(db, 's')
    try:
        monkeypatch.setattr(registry, '_detached_host_fate',
                            lambda *a: pytest.fail('opaque resource was interpreted as a host PID'))
        monkeypatch.setattr(registry, '_terminate_host_pid',
                            lambda *a, **kw: pytest.fail('unproved resource was signaled'))
        assert registry._write_checkpoint()
        assert json.loads(path.read_text()) == [entry]
        assert registry.recover_from_checkpoint() == 0
        assert json.loads(path.read_text()) == [entry]
        assert not await retry_process_stop(authority, REF)
        assert json.loads(path.read_text()) == [entry]
        assert process_stop_pending(db, 's') == token
    finally:
        db.close()
