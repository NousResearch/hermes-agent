"""An exited wrapper cannot erase an unresolved native scope or signal a recycled PID."""
import asyncio
import json
import shlex
import sys

import pytest

from gateway.session_authority import LiveSession, SessionAuthority
from hermes_state_runtime import begin_runtime_epoch
from hermes_state_runtime import RuntimeStoreError
from tests.gateway.test_session_authority_cancel_settlement import ACTOR, REF, _authority


def _spawn(registry, root, owner='s'):
    command = shlex.join([sys.executable, '-c', 'import threading; threading.Event().wait(60)'])
    return registry.spawn_local(command, cwd=str(root), task_id=owner, owner_task_id=owner,
                                session_key='route', persist_on_release=True)


@pytest.mark.asyncio
@pytest.mark.parametrize('reused_pid', [False, True], ids=['dead-wrapper', 'reused-wrapper-pid'])
async def test_refused_owned_scope_survives_reload_without_signaling_foreign_pid(tmp_path, monkeypatch, reused_pid):
    from tools import process_registry as module
    from gateway.session_ancillary import process_stop_pending, retry_process_stop
    db, authority = _authority(tmp_path, monkeypatch)
    original = module.ProcessRegistry()
    monkeypatch.setattr(module, 'process_registry', original)
    own, neighbor = _spawn(original, tmp_path), _spawn(original, tmp_path, 'neighbor')
    unit = own.systemd_unit = 'hermes-worker-owned-refused.scope'
    attempts = []
    monkeypatch.setattr(module, '_stop_systemd_unit', lambda value: attempts.append(value) or False)
    try:
        with pytest.raises(RuntimeStoreError, match='process_stop_unknown'):
            await authority.interrupt(ACTOR, REF, authority._handle(REF).execution_generation)
        assert await asyncio.to_thread(own._completion_event.wait, 10)
        assert own._scope_stop_pending and process_stop_pending(db, 's') is not None
        if reused_pid:
            # The kernel process here really is foreign: only the recorded birth is stale.
            # Serialize through the native writer, without editing the checkpoint file.
            own.pid, own.host_start_time = neighbor.pid, neighbor.host_start_time - 1
            original._write_checkpoint()
        saved = json.loads(module._checkpoint_path().read_text(encoding='utf-8'))
        entry = next(item for item in saved if item['session_id'] == own.id)
        assert entry['_scope_stop_pending'] is True and entry['systemd_unit'] == unit
        epoch = begin_runtime_epoch(db, instance_id='reloaded')
        reloaded = SessionAuthority(authority.runner, profile_id='owned', instance_id='reloaded', db=db, epoch=epoch)
        reloaded.sessions['s'] = LiveSession(authority.sessions['s'].source, 'route')
        registry = module.ProcessRegistry()
        monkeypatch.setattr(module, 'process_registry', registry)
        monkeypatch.setattr(registry, '_terminate_host_pid', lambda *a, **kw: pytest.fail('a dead/reused wrapper PID was signaled'))
        attempts.clear()
        assert not await retry_process_stop(reloaded, REF)
        assert attempts == [unit] and process_stop_pending(db, 's') is not None
        assert not neighbor.exited
        assert own.id in {item['session_id'] for item in json.loads(module._checkpoint_path().read_text(encoding='utf-8'))}
        monkeypatch.setattr(module, '_stop_systemd_unit', lambda value: attempts.append(value) or True)
        assert await retry_process_stop(reloaded, REF)
        assert process_stop_pending(db, 's') is None and not neighbor.exited
        remaining = json.loads(module._checkpoint_path().read_text(encoding='utf-8'))
        assert own.id not in {item['session_id'] for item in remaining}
        assert neighbor.id in {item['session_id'] for item in remaining}
    finally:
        monkeypatch.setattr(module, '_stop_systemd_unit', lambda value: True)
        original.kill_process(own.id)
        original.kill_process(neighbor.id)
        db.close()


@pytest.mark.asyncio
async def test_finished_scope_success_removes_pending_checkpoint_immediately(tmp_path, monkeypatch):
    from tools import process_registry as module
    registry = module.ProcessRegistry()
    monkeypatch.setattr(module, 'process_registry', registry)
    own, neighbor = _spawn(registry, tmp_path), _spawn(registry, tmp_path, 'neighbor')
    own.systemd_unit = 'hermes-worker-owned-refused.scope'
    monkeypatch.setattr(module, '_stop_systemd_unit', lambda value: False)
    try:
        assert registry.kill_process(own.id).get('scope_stop_failed') == own.systemd_unit
        assert await asyncio.to_thread(own._completion_event.wait, 10)
        assert own._scope_stop_pending
        assert own.id in {item['session_id'] for item in json.loads(module._checkpoint_path().read_text(encoding='utf-8'))}
        history = own.exit_code, own.completion_reason, own.output_buffer
        monkeypatch.setattr(module, '_stop_systemd_unit', lambda value: True)
        assert registry.kill_process(own.id)['status'] == 'already_exited'
        assert not own._scope_stop_pending and history == (own.exit_code, own.completion_reason, own.output_buffer)
        remaining = json.loads(module._checkpoint_path().read_text(encoding='utf-8'))
        assert own.id not in {item['session_id'] for item in remaining}
        assert neighbor.id in {item['session_id'] for item in remaining} and not neighbor.exited
    finally:
        monkeypatch.setattr(module, '_stop_systemd_unit', lambda value: True)
        registry.kill_process(own.id)
        registry.kill_process(neighbor.id)


@pytest.mark.asyncio
async def test_completed_log_reader_cannot_hide_unattempted_owned_scope(tmp_path, monkeypatch):
    from gateway.session_ancillary import process_stop_pending, record_process_stop, retry_process_stop
    from tools import process_registry as module
    from tools.process_registry_output_log import output_dir
    db, authority = _authority(tmp_path, monkeypatch)
    original = module.ProcessRegistry()
    monkeypatch.setattr(module, 'process_registry', original)
    neighbor = _spawn(original, tmp_path, 'neighbor')
    output_dir().mkdir(mode=0o700, parents=True, exist_ok=True)
    monkeypatch.setattr(module, 'worker_output_log', lambda sid: output_dir() / (sid + '.log'))
    own = original.spawn_local(shlex.join([sys.executable, '-c', 'print("scope output")']),
                               cwd=str(tmp_path), task_id='s', owner_task_id='s', session_key='route')
    log = output_dir() / (own.id + '.log')
    try:
        assert await asyncio.to_thread(own._completion_event.wait, 10)
        assert log.with_suffix('.exit').read_text().strip() == '0'
        entry = {'session_id': own.id, **{field: getattr(own, field) for field in module._CHECKPOINT_FIELDS}}
        unit = 'hermes-worker-unattempted.scope'
        entry.update(systemd_unit=unit, _scope_stop_pending=False,
                     pid=neighbor.pid, host_start_time=neighbor.host_start_time - 1)
        original._write_checkpoint(extra_entries=[entry])
        record_process_stop(authority, REF, authority._handle(REF).execution_generation)
        epoch = begin_runtime_epoch(db, instance_id='reloaded')
        reloaded = SessionAuthority(authority.runner, profile_id='owned', instance_id='reloaded', db=db, epoch=epoch)
        reloaded.sessions['s'] = LiveSession(authority.sessions['s'].source, 'route')
        registry = module.ProcessRegistry()
        monkeypatch.setattr(module, 'process_registry', registry)
        track = registry._track_started

        def reader_finishes_first(session, *args, **kwargs):
            track(session, *args, **kwargs)
            assert session._completion_event.wait(10)

        monkeypatch.setattr(registry, '_track_started', reader_finishes_first)
        monkeypatch.setattr(registry, '_terminate_host_pid', lambda *a, **kw: pytest.fail('foreign PID signaled'))
        attempts = []
        monkeypatch.setattr(module, '_stop_systemd_unit', lambda value: attempts.append(value) or False)
        assert not await retry_process_stop(reloaded, REF)
        assert attempts == [unit] and process_stop_pending(db, 's') is not None
        assert own.id in {item['session_id'] for item in json.loads(module._checkpoint_path().read_text())}
        restored = registry._finished[own.id]
        history = restored.exit_code, restored.completion_reason, restored.output_buffer
        monkeypatch.setattr(module, '_stop_systemd_unit', lambda value: True)
        assert await retry_process_stop(reloaded, REF)
        assert process_stop_pending(db, 's') is None and not neighbor.exited
        assert history == (restored.exit_code, restored.completion_reason, restored.output_buffer)
        remaining = json.loads(module._checkpoint_path().read_text())
        assert own.id not in {item['session_id'] for item in remaining}
        assert neighbor.id in {item['session_id'] for item in remaining}
    finally:
        original.kill_process(neighbor.id)
        db.close()
