"""Real readers exiting at handoff retain committed ownership before any success ACK."""
import asyncio
import json
import threading

import pytest

from tests.gateway.test_process_handoff_stop import Parent, _child, _spawn
from tests.waiting import wait_until


@pytest.mark.asyncio
@pytest.mark.parametrize("write_refused", [False, True])
async def test_handoff_ack_requires_durable_parent_if_reader_exits_during_commit(tmp_path, monkeypatch, write_refused):
    from tools import process_registry as module
    from tools.delegate_tool import _unregister_subagent
    from tools.process_registry_handoff import handoff_process
    registry = module.ProcessRegistry()
    monkeypatch.setattr(module, 'process_registry', registry)
    parent = Parent()
    sid, _child_agent = _child(parent)
    gate = tmp_path / 'finish-job'
    process = _spawn(registry, tmp_path, sid, 'native-child-session', 'route', logged=True, release=gate)
    process.notify_on_complete = True
    registry._write_checkpoint()
    stale_registry = module.ProcessRegistry()
    stale_registry.recover_from_checkpoint()
    stale = stale_registry._running[process.id]
    assert stale.ownership_generation == 0 and stale.parent_session_id == 'native-child-session'
    stale_save_ready, stale_save_allowed = threading.Event(), threading.Event()
    save = module.save_completed_result
    def held_stale_save(session):
        if session is stale:
            stale_save_ready.set()
            assert stale_save_allowed.wait(20)
        return save(session)
    monkeypatch.setattr(module, 'save_completed_result', held_stale_save)
    import utils
    atomic_write = utils.atomic_json_write
    def refuse_checkpoint(path, *args, **kwargs):
        if write_refused and path == module._checkpoint_path() and process.ownership_generation == 1:
            raise PermissionError("fixture checkpoint transfer refusal after real exit")
        return atomic_write(path, *args, **kwargs)
    monkeypatch.setattr(utils, "atomic_json_write", refuse_checkpoint)
    write = registry._write_checkpoint
    evidence = {}
    # Yield at the boundary between the transfer's running check and the
    # checkpoint writer's running check. Both real log readers observe exit.
    def exit_before_checkpoint(*args, **kwargs):
        if kwargs.get('_lock_held'):
            gate.touch()
            wait_until(lambda: process.exited, timeout=20, what='real reader observes exit')
            assert stale_save_ready.wait(20)
            assert not process._completion_event.is_set(), 'reader settlement waits for transfer lock'
        result = write(*args, **kwargs)
        if kwargs.get('_lock_held'):
            entries = json.loads(module._checkpoint_path().read_text())
            evidence['checkpoint_owns_parent_at_ack'] = any(
                entry['session_id'] == process.id and entry['parent_session_id'] == 's' for entry in entries)
            stale_save_allowed.set()
            assert stale._completion_event.wait(20)
            from hermes_constants import get_hermes_home
            receipt = json.loads((get_hermes_home() / 'logs' / 'process-results' / (process.id + '.json')).read_text())
            evidence['retained_parent_at_ack'] = receipt['parent_session_id']
            event, = list(stale_registry.completion_queue.queue)
            evidence['queued_completion_parent'] = event['parent_session_id']
        return result
    monkeypatch.setattr(registry, '_write_checkpoint', exit_before_checkpoint)
    try:
        result = handoff_process(process.id, {'data': 'parent durable watcher'}, sid)
        evidence['handoff_status'] = result.get('status', result)
        print(json.dumps(evidence))
        if write_refused:
            assert 'error' in result, evidence
            assert not evidence['checkpoint_owns_parent_at_ack'], evidence
            assert evidence['retained_parent_at_ack'] == 'native-child-session', evidence
            assert evidence['queued_completion_parent'] == 'native-child-session', evidence
        else:
            assert result['status'] == 'handed_off', evidence
            assert evidence['checkpoint_owns_parent_at_ack'], evidence
            assert evidence['retained_parent_at_ack'] == 's', evidence
            assert evidence['queued_completion_parent'] == 's', evidence
        assert await asyncio.to_thread(process._completion_event.wait, 20)
        event, = list(registry.completion_queue.queue)
        assert event['parent_session_id'] == ('native-child-session' if write_refused else 's')
        from tools.process_registry_results import load_completed_results
        retained = load_completed_results(owns_entry=lambda row: row['id'] == process.id, strict=True)[process.id]
        assert retained.parent_session_id == event['parent_session_id']
        assert retained.exit_code == 0 and 'retained handoff output' in retained.output_buffer
        assert all(row['session_id'] != process.id for row in json.loads(module._checkpoint_path().read_text()))
    finally:
        stale_save_allowed.set()
        monkeypatch.setattr(registry, '_write_checkpoint', write)
        _unregister_subagent(sid)
        gate.touch()
        await asyncio.to_thread(process._completion_event.wait, 20)
        await asyncio.to_thread(stale._completion_event.wait, 20)
        registry.kill_process(process.id)
