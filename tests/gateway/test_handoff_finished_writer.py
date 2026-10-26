"""A stale finished native reader preserves committed handoff until the result inherits it."""
import asyncio
import json
import threading
import pytest
from tests.gateway.test_process_handoff_stop import Parent, _child, _spawn
from tests.waiting import wait_until

@pytest.mark.asyncio
async def test_finished_writer_after_parent_commit(tmp_path, monkeypatch):
    from tools import process_registry as module
    from tools.delegate_tool import _unregister_subagent
    from tools.process_registry_handoff import handoff_process
    from hermes_constants import get_hermes_home
    registry = module.ProcessRegistry()
    monkeypatch.setattr(module, 'process_registry', registry)
    parent = Parent()
    sid, _child_agent = _child(parent)
    gate = tmp_path / 'release'
    process = _spawn(registry, tmp_path, sid, 'native-child-session', 'route', logged=True, release=gate)
    process.notify_on_complete = True
    registry._write_checkpoint()
    stale_registry = module.ProcessRegistry()
    stale_registry.recover_from_checkpoint()
    stale = stale_registry._running[process.id]
    write = registry._write_checkpoint
    proof = {}
    def at_commit(*args, **kwargs):
        if not kwargs.get('_lock_held'):
            return write(*args, **kwargs)
        gate.touch()
        wait_until(lambda: process.exited, timeout=20, what='real reader exit before transfer commit')
        assert stale._completion_event.wait(20)
        result = write(*args, **kwargs)
        assert result
        rows = json.loads(module._checkpoint_path().read_text())
        proof['parent_committed'] = any(r['session_id']==process.id and r['parent_session_id']=='s' for r in rows)
        # Another interpreter settled this real process before the parent commit.
        # Its ordinary checkpoint writer can run afterward from its finished table.
        assert stale_registry._write_checkpoint()
        rows = json.loads(module._checkpoint_path().read_text())
        receipt = json.loads((get_hermes_home()/'logs'/'process-results'/(process.id+'.json')).read_text())
        proof['checkpoint_remaining_at_ack'] = any(r['session_id']==process.id for r in rows)
        proof['retained_parent_at_ack'] = receipt['parent_session_id']
        return result
    monkeypatch.setattr(registry,'_write_checkpoint',at_commit)
    try:
        result = handoff_process(process.id, {'data':'watch'}, sid)
        print({'result':result, **proof})
        assert result['status']=='handed_off'
        assert proof['parent_committed']
        assert proof['checkpoint_remaining_at_ack'], proof
        assert stale.parent_session_id == 's' and stale.ownership_generation == 1
        assert await asyncio.to_thread(process._completion_event.wait, 20)
        receipt = json.loads((get_hermes_home()/'logs'/'process-results'/(process.id+'.json')).read_text())
        assert receipt['parent_session_id'] == 's' and receipt['ownership_generation'] == 1
        assert receipt['exit_code'] == 0 and 'retained handoff output' in receipt['output']
        event, = list(registry.completion_queue.queue)
        assert event['parent_session_id'] == 's' and event['owner_task_id'] == 'parent-turn'
        assert stale_registry._write_checkpoint()
        assert all(row['session_id'] != process.id for row in json.loads(module._checkpoint_path().read_text()))
    finally:
        monkeypatch.setattr(registry,'_write_checkpoint',write)
        gate.touch()
        _unregister_subagent(sid)
        await asyncio.to_thread(process._completion_event.wait,20)
        await asyncio.to_thread(stale._completion_event.wait,20)
        registry.kill_process(process.id)
