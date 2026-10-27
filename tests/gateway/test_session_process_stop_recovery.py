"""Failed Stop cleanup fences later work across automation, idle state and owner reload."""
import asyncio
import json
import shlex
import sys
from types import SimpleNamespace
import uuid

import pytest

from gateway.session_authority import LiveSession, SessionAuthority
from hermes_state_runtime import RuntimeStoreError, admit_session_input, begin_runtime_epoch, list_session_admissions, recover_session_inputs
from tests.gateway.test_session_authority_cancel_settlement import ACTOR, REF, _authority, _submit, _wire_turn_agent
from tests.gateway.test_session_authority_preadopt_stop import Agent
from tests.waiting import async_wait_until, wait_until


@pytest.fixture
def registry(monkeypatch):
    from tools import process_registry as module
    registry = module.ProcessRegistry()
    monkeypatch.setattr(module, 'process_registry', registry)
    yield registry
    for process in list(registry._running.values()):
        registry.kill_process(process.id)


def _spawn(registry, root, *, owner='s', route='route', command=None):
    ready = None
    if command is None:
        ready = root / ('ready-' + uuid.uuid4().hex)
        code = f'import pathlib,threading; pathlib.Path({str(ready)!r}).touch(); threading.Event().wait(120)'
        # Native recovery/fencing is independent of interactive login-shell
        # teardown. The real owned Python job is ready before the Stop begins.
        command = 'exec ' + shlex.join([sys.executable, '-c', code])
    process = registry.spawn_local(command, cwd=str(root), task_id=owner, owner_task_id=owner,
                                   session_key=route, persist_on_release=True)
    if ready is not None:
        wait_until(ready.exists, timeout=20, what='owned recovery fixture ready')
    return process


@pytest.mark.asyncio
@pytest.mark.parametrize('automation', [False, True], ids=['idle', 'automation'])
@pytest.mark.parametrize('reload_owner', [False, True], ids=['same-owner', 'reloaded-owner'])
async def test_failed_stop_fences_later_inputs_until_actual_cleanup(tmp_path, monkeypatch, registry,
                                                                 automation, reload_owner):
    from gateway import session_finite
    from gateway.session_ancillary import process_stop_pending
    from gateway.session_unattended_unknown import settle_recovered_unattended
    from tools import process_registry as module
    db, authority = _authority(tmp_path, monkeypatch)
    db.append_message('s', 'user', 'retained completed history')
    process = _spawn(registry, tmp_path)
    neighbor = _spawn(registry, tmp_path, owner='neighbor', route='other-route')
    started, finish = asyncio.Event(), asyncio.Event()
    executed = []
    agent = Agent()
    agent.session_id = 's'
    authority.runner._cached_agent_for = lambda route: agent

    async def execute(owner, ref, row):
        executed.append(row['request_id'])
        _wire_turn_agent(owner, row['generation'], agent)
        if row['request_id'] == 'automation-head':
            started.set()
            await asyncio.wait_for(finish.wait(), 10)
        else:
            assert process.exited, 'later work started with prior effects still alive'
        return 'done'

    monkeypatch.setattr(session_finite, 'execute_finite_admission', execute)
    kill = registry.kill_process
    initial_registry, initial_kill = registry, kill
    refusal = lambda *a, **kw: {'status': 'error', 'error': 'fixture termination refusal'}
    monkeypatch.setattr(registry, 'kill_process', refusal)
    try:
        if automation:
            admit_session_input(db, epoch=authority.epoch, principal_id='automation:route', session_id='s',
                                request_id='automation-head', payload={'text': 'work'})
            authority._schedule(REF)
            await asyncio.wait_for(started.wait(), 10)
        with pytest.raises(RuntimeStoreError, match='process_stop_unknown'):
            await authority.interrupt(ACTOR, REF, authority._handle(REF).execution_generation)
        await _submit(authority, 'follower')
        finish.set()
        await asyncio.wait_for(authority.sessions['s'].task, 15)
        assert process_stop_pending(db, 's') is not None and not process.exited
        if reload_owner:
            epoch = begin_runtime_epoch(db, instance_id='reloaded')
            recover_session_inputs(db, epoch=epoch)
            assert settle_recovered_unattended(db, epoch) == 0
            old = authority
            authority = SessionAuthority(old.runner, profile_id='owned', instance_id='reloaded', db=db, epoch=epoch)
            authority.sessions['s'] = LiveSession(old.sessions['s'].source, 'route')
            registry = module.ProcessRegistry()  # simulate a fresh interpreter's process table
            monkeypatch.setattr(module, 'process_registry', registry)
            kill = registry.kill_process
            monkeypatch.setattr(registry, 'kill_process', refusal)
        await _submit(authority, 'later-input')
        await asyncio.wait_for(authority.sessions['s'].task, 15)
        assert executed == (['automation-head'] if automation else [])
        statuses = {r['request_id']: r['status'] for r in list_session_admissions(db, session_id='s')}
        assert statuses['follower'] == statuses['later-input'] == 'queued'
        if automation:
            assert statuses['automation-head'] == 'unknown'
        assert not process.exited and not neighbor.exited
        monkeypatch.setattr(registry, 'kill_process', kill)
        await _submit(authority, 'verified-successor')
        expected = (['automation-head'] if automation else []) + ['follower', 'later-input', 'verified-successor']
        await async_wait_until(lambda: executed == expected and authority.sessions['s'].task.done(), timeout=20)
        assert process_stop_pending(db, 's') is None
        assert await asyncio.to_thread(process._completion_event.wait, 10)
        assert not neighbor.exited
        if reload_owner:
            from tools.process_registry import _checkpoint_path
            retained = json.loads(_checkpoint_path().read_text(encoding='utf-8'))
            assert neighbor.id in {entry['session_id'] for entry in retained}, 'cleanup dropped the neighbor recovery record'
            registry.recover_from_checkpoint()
            assert registry.get(neighbor.id) is not None, 'neighbor stopped being manageable after cleanup'
        assert executed == (['automation-head'] if automation else []) + ['follower', 'later-input', 'verified-successor']
        assert db.get_messages('s')[0]['content'] == 'retained completed history'
    finally:
        monkeypatch.setattr(registry, 'kill_process', kill)
        monkeypatch.setattr(initial_registry, 'kill_process', initial_kill)
        db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('managed', [False, True], ids=['native', 'managed'])
async def test_stop_consumes_ready_own_completion_but_delivers_neighbor(tmp_path, monkeypatch, registry, managed):
    from agent.managed_worker import WorkerControls
    from gateway.session_ancillary import process_stop_pending
    from tools.process_registry_job import read_job_receipt
    import threading
    if not managed:
        db, authority = _authority(tmp_path, monkeypatch)
    gate = tmp_path / 'release'
    code = ('import pathlib,threading; p=pathlib.Path(' + repr(str(gate)) + '); e=threading.Event()\n'
            'while not p.exists(): e.wait(.01)')
    command = shlex.join([sys.executable, '-c', code])
    own = _spawn(registry, tmp_path, command=command)
    neighbor = _spawn(registry, tmp_path, owner='neighbor', command=command)
    for process in (own, neighbor):
        process.notify_on_complete = True
    gate.touch()
    for process in (own, neighbor):
        assert await asyncio.to_thread(process._completion_event.wait, 10)
    if managed:
        controls = WorkerControls.__new__(WorkerControls)
        controls.route, controls.stopped = 'route', threading.Event()
        controls.stop_processes_requested = False
        controls.lock, controls.task_checks, controls.clarifications = threading.Lock(), {}, {}
        controls.agent = SimpleNamespace(session_id='s', hard_interrupt=lambda *a, **kw: None)
        controls.stop(processes=True)
    else:
        try:
            await authority.interrupt(ACTOR, REF, authority._handle(REF).execution_generation)
        except RuntimeStoreError as error:
            state = {p.id: {"receipt": read_job_receipt(p), "exited": p.exited,
                           "pending": p._scope_stop_pending, "completion": p._completion_event.is_set(),
                           "complete_result": p._result_complete, "persisted_generation": p._result_persisted_generation,
                           "ownership_generation": p.ownership_generation}
                     for p in (own, neighbor)}
            pytest.fail(f"Ready native Stop failed: {error.reason}; {state}")
        assert process_stop_pending(db, 's') is None
        db.close()
    assert registry.is_completion_consumed(own.id)
    assert not registry.is_completion_consumed(neighbor.id)
    registry._completions_restored = True
    events = registry.drain_notifications(session_key='route', skip_poll_observed=False)
    assert [event['session_id'] for event, _text in events] == [neighbor.id]
    assert own.completion_reason == neighbor.completion_reason == 'exited'
    assert own.exit_code == neighbor.exit_code == 0


@pytest.mark.asyncio
@pytest.mark.parametrize('refuse', [False, True], ids=['verified-stop', 'refused-stop'])
async def test_idle_stop_recovers_previous_worker_before_clearing_status(tmp_path, monkeypatch, registry, refuse):
    from tools import process_registry as module
    from gateway.session_ancillary import process_stop_pending
    from gateway import session_finite
    db, authority = _authority(tmp_path, monkeypatch)
    previous = module.ProcessRegistry()
    own = _spawn(previous, tmp_path)
    neighbors = [_spawn(previous, tmp_path, owner='neighbor'),
                 _spawn(previous, tmp_path, route='other-route')]
    kill = registry.kill_process
    if refuse:
        monkeypatch.setattr(registry, 'kill_process', lambda *a, **kw: {'status': 'error', 'error': 'fixture refusal'})
    executed = []

    async def execute(owner, ref, row):
        executed.append(row['request_id'])
        assert own.exited, 'idle Stop handed off while a previous worker process still lived'
        return 'done'

    monkeypatch.setattr(session_finite, 'execute_finite_admission', execute)
    try:
        assert not registry._running, 'the gateway starts without the previous worker process table'
        if refuse:
            with pytest.raises(RuntimeStoreError, match='process_stop_unknown'):
                await authority.interrupt(ACTOR, REF, authority._handle(REF).execution_generation)
        else:
            await authority.interrupt(ACTOR, REF, authority._handle(REF).execution_generation)
        if refuse:
            assert process_stop_pending(db, 's') is not None and not own.exited
            await _submit(authority, 'held-follower')
            await asyncio.wait_for(authority.sessions['s'].task, 15)
            assert executed == [] and not own.exited
            monkeypatch.setattr(registry, 'kill_process', kill)
            await authority.interrupt(ACTOR, REF, authority._handle(REF).execution_generation)
            await asyncio.wait_for(authority.sessions['s'].task, 15)
            assert executed == ['held-follower']
        assert await asyncio.to_thread(own._completion_event.wait, 10)
        assert process_stop_pending(db, 's') is None
        assert all(not process.exited for process in neighbors)
        registry.recover_from_checkpoint()
        assert all(registry.get(process.id) is not None for process in neighbors)
    finally:
        monkeypatch.setattr(registry, 'kill_process', kill)
        for process in [own, *neighbors]:
            previous.kill_process(process.id)
        db.close()
