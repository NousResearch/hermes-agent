"""A later Kanban attempt waits for actual native execution and scoped cleanup."""
import asyncio
from dataclasses import replace
import json
import shlex
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.session_kanban_stop import retire_previous_attempts
from hermes_state_runtime import RuntimeStoreError
from tests.gateway.test_session_authority_cancel_settlement import ACTOR, REF, _authority, _submit, _wire_turn_agent
from tests.gateway.test_session_authority_preadopt_stop import Agent


@pytest.fixture
def native_attempt(tmp_path, monkeypatch):
    from gateway import session_kanban_stop, session_policy
    from hermes_state_local import local_receipt
    from tools import process_registry as module
    db, authority = _authority(tmp_path, monkeypatch)
    context = {'db': str(tmp_path / 'board.db'), 'board': 'work', 'task_id': 'task', 'run_id': 2}
    prior = {**context, 'run_id': 1}
    monkeypatch.setattr(session_kanban_stop, 'previous_attempts', lambda _: [
        {'db': str(db.db_path), 'session_id': 's', 'run_id': 1}])
    monkeypatch.setattr(session_kanban_stop, 'previous_workers_gone', lambda *a: True)
    monkeypatch.setattr(sys.modules[local_receipt.__module__], 'local_receipt', lambda *a: {'policy': {}})
    monkeypatch.setattr(session_policy, 'restore_policy', lambda _: SimpleNamespace(source='kanban', kanban_json=json.dumps(prior)))
    registry = module.ProcessRegistry()
    monkeypatch.setattr(module, 'process_registry', registry)
    actor = replace(ACTOR, capabilities=ACTOR.capabilities | {'session:read'})
    connection = SimpleNamespace(authority=authority, actor=actor, resume=AsyncMock())
    async def interrupt(ref, params):
        return await authority.interrupt(actor, ref, params['execution_generation'])
    connection.interrupt = interrupt
    yield db, authority, context, connection, registry
    for process in list(registry._running.values()):
        registry.kill_process(process.id)
    db.close()


def spawn(registry, root, owner='s'):
    command = shlex.join([sys.executable, '-c', 'import threading; threading.Event().wait(60)'])
    return registry.spawn_local(command, cwd=str(root), owner_task_id=owner, task_id=owner,
                                session_key='route', persist_on_release=True)


@pytest.mark.asyncio
async def test_predecessor_cleanup_refusal_blocks_replacement_and_keeps_neighbor(native_attempt, tmp_path, monkeypatch):
    from gateway.session_ancillary import process_stop_pending
    db, _authority, context, connection, registry = native_attempt
    own, neighbor = spawn(registry, tmp_path), spawn(registry, tmp_path, 'neighbor')
    kill = registry.kill_process
    monkeypatch.setattr(registry, 'kill_process', lambda *a, **kw: {'status': 'error', 'error': 'owned fixture refusal'})
    with pytest.raises(RuntimeStoreError, match='kanban_previous_execution_unknown'):
        await retire_previous_attempts(connection, context)
    assert process_stop_pending(db, 's') is not None
    assert not own.exited and not neighbor.exited
    monkeypatch.setattr(registry, 'kill_process', kill)
    await retire_previous_attempts(connection, context)
    assert await asyncio.to_thread(own._completion_event.wait, 10)
    assert process_stop_pending(db, 's') is None and not neighbor.exited


@pytest.mark.asyncio
async def test_predecessor_finalization_precedes_cleanup_of_inflight_spawn(native_attempt, tmp_path, monkeypatch):
    from gateway import session_finite
    _db, authority, context, connection, registry = native_attempt
    started, stopped, finish = asyncio.Event(), asyncio.Event(), asyncio.Event()
    agent = Agent()
    agent.session_id = 's'
    original = agent.interrupt
    def interrupt(*args, **kwargs):
        original(*args, **kwargs)
        stopped.set()
    agent.interrupt = interrupt
    spawned = []
    async def execute(owner, ref, row):
        _wire_turn_agent(owner, row['generation'], agent)
        started.set()
        await finish.wait()
        spawned.append(spawn(registry, tmp_path))
        return 'done'
    monkeypatch.setattr(session_finite, 'execute_finite_admission', execute)
    await _submit(authority, 'old-attempt')
    assert await asyncio.wait_for(started.wait(), 5)
    retiring = asyncio.create_task(retire_previous_attempts(connection, context))
    assert await asyncio.wait_for(stopped.wait(), 5)
    assert not retiring.done(), 'a momentarily empty process table released a running predecessor'
    finish.set()
    await asyncio.wait_for(retiring, 15)
    assert len(spawned) == 1 and spawned[0].exited
    assert 's' not in authority.executing


@pytest.mark.asyncio
async def test_predecessor_unproved_worker_death_keeps_durable_fence(native_attempt, monkeypatch):
    from gateway import session_kanban_stop
    from gateway.session_ancillary import process_stop_pending
    db, _authority, context, connection, _registry = native_attempt
    monkeypatch.setattr(session_kanban_stop, 'previous_workers_gone', lambda *a: False)
    with pytest.raises(RuntimeStoreError, match='kanban_previous_execution_unknown'):
        await retire_previous_attempts(connection, context)
    assert process_stop_pending(db, 's') is not None
