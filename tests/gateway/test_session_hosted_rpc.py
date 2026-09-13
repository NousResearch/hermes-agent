"""Room worker threads use the real authority ledger, never TUI dispatch."""
import asyncio
import threading
from types import SimpleNamespace

import pytest


@pytest.fixture
def owner(tmp_path, monkeypatch):
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_contract import Principal
    from gateway import run, session_policy
    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'model': {'default': 'fixture'}, 'platform_toolsets': {'cli': []}})
    monkeypatch.setattr(run, '_resolve_gateway_model', lambda cfg: 'fixture')
    # Parent-owned private restore hook, explicitly not an ordinary-daemon proof.
    original = session_policy.restore_policy
    def restore(data):
        from dataclasses import replace
        if data['source'] == 'bot_room':
            return replace(original({**data, 'source': 'gui', 'platform': 'desktop'}), source='bot_room', platform='bot_room')
        return original(data)
    monkeypatch.setattr(session_policy, 'restore_policy', restore)
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    agent = SimpleNamespace(interrupted=False)
    agent.interrupt = lambda: setattr(agent, 'interrupted', True)
    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False,
                             _cached_agent_for=lambda route: agent)
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    authority = asyncio.run_coroutine_threadsafe(initialize_session_authority(runner, profile_id='owned', instance_id='first'), loop).result()
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)
    principal = Principal('durable-room-owner', 'owned', frozenset({'session:create', 'session:read', 'session:submit', 'session:control', 'session:approve'}), 'room-worker')
    yield authority, loop, principal, agent
    loop.call_soon_threadsafe(loop.stop)
    thread.join()
    loop.close()
    store._db.close()


def test_room_binding_exact_retry_terminal_history_and_unknown(owner):
    from gateway.session_hosted_rpc import HostedRoomAuthorityRPC
    from gateway.hosted_room_driver import TaskIdentity
    from hermes_state_runtime import RuntimeStoreError, list_session_admissions, claim_session_input, settle_session_input, begin_runtime_epoch, recover_session_inputs
    authority, loop, principal, _ = owner
    allowed = [True]
    seen = []
    def authorize(operation, task, generation):
        seen.append(threading.get_ident())
        return allowed[0]
    def adapter(room='room'):
        return HostedRoomAuthorityRPC(authority, loop, room_id=room, member_id='member', profile='default', principal=principal, authorize=authorize)
    rpc = adapter()
    coords = dict(profile='default', source='bot_room')
    assert rpc.resolve_exact(**coords, title='Group: room') is None
    sid = rpc.create(**coords, title='Group: room')['session_id']
    stored = authority.db.get_session(sid)
    assert stored['source'] == 'bot_room'
    assert stored['title'] == 'Group: room'
    assert stored['hidden']
    assert adapter().resolve_exact(**coords, title='Group: room')['session_id'] == sid
    with pytest.raises(RuntimeStoreError):
        adapter('other').resume(**coords, session_id=sid)
    args = dict(**coords, session_id=sid, prompt='input', task=TaskIdentity('room','task','thread','turn'), execution_generation=1)
    callbacks = []
    first = rpc.submit(**args, on_terminal=callbacks.append)
    assert rpc.submit(**args, on_terminal=callbacks.append)['admission_id'] == first['admission_id']
    with pytest.raises(RuntimeStoreError, match='admission_conflict'):
        rpc.submit(**{**args, 'prompt': 'changed'}, on_terminal=callbacks.append)
    row = claim_session_input(authority.db, epoch=authority.epoch, session_id=sid)
    settle_session_input(authority.db, epoch=authority.epoch, admission_id=row['admission_id'], generation=row['generation'], outcome='completed', result={'result': {'final_response': 'reply'}, 'usage': {}})
    history = rpc.history(**coords, session_id=sid)
    assert history[-1]['content'] == 'reply'
    assert history[-1]['task_id'] == 'task'
    assert history[-1]['execution_generation'] == 1
    assert callbacks and callbacks[0]['status'] == 'settled'
    args['task'] = TaskIdentity('room','unknown','thread','turn2')
    rpc.submit(**args, on_terminal=callbacks.append)
    claim_session_input(authority.db, epoch=authority.epoch, session_id=sid)
    authority.epoch = begin_runtime_epoch(authority.db, instance_id='cold')
    recover_session_inputs(authority.db, epoch=authority.epoch)
    with pytest.raises(RuntimeStoreError, match='unknown_execution') as exc:
        adapter().submit(**args, on_terminal=callbacks.append)
    assert not getattr(exc.value, 'not_admitted', False)
    assert len(list_session_admissions(authority.db, session_id=sid, pending_only=False)) == 2
    allowed[0] = False
    for operation in (lambda: rpc.history(**coords, session_id=sid), lambda: rpc.info(**coords, session_id=sid), lambda: rpc.create(**coords, title='Group: room')):
        with pytest.raises(RuntimeStoreError, match='permission_denied'):
            operation()
    assert seen and all(t != threading.get_ident() for t in seen)


def test_controls_are_exact_current_admission_and_loop_safe(owner):
    from gateway.session_hosted_rpc import HostedRoomAuthorityRPC
    from gateway.hosted_room_driver import TaskIdentity
    from hermes_state_runtime import RuntimeStoreError, claim_session_input
    authority, loop, principal, agent = owner
    rpc = HostedRoomAuthorityRPC(authority, loop, room_id='room', member_id='member', profile='default', principal=principal, authorize=lambda *args: True)
    coords = dict(profile='default', source='bot_room')
    sid = rpc.create(**coords, title='Group: room')['session_id']
    rpc.submit(**coords, session_id=sid, prompt='input', task=TaskIdentity('room','task','thread','turn'), execution_generation=1, on_terminal=lambda receipt: None)
    row = claim_session_input(authority.db, epoch=authority.epoch, session_id=sid)
    assert rpc.info(**coords, session_id=sid)['task_id'] == 'task'
    live = authority.sessions[sid]
    answers = []
    authority.register_approval(sid, row['generation'], live.route, {'request_id': 'approve-me', 'command': 'owned fixture'})
    live.controls.remote_responders['approve-me'] = lambda *answer: answers.append(answer)
    pending = rpc.info(**coords, session_id=sid)['pending_approval']
    assert pending['request_id'] == 'approve-me'
    assert pending['choices'] == ['once', 'deny']
    with pytest.raises(RuntimeStoreError, match='invalid_params'):
        rpc.approve(session_id=sid, request_id='approve-me', choice='always')
    assert not answers
    assert rpc.approve(session_id=sid, request_id='approve-me', choice='once')['status'] == 'resolved'
    assert answers == [('approval', 'approve-me', 'once')]
    with pytest.raises(RuntimeStoreError, match='stale_generation'):
        rpc.interrupt(**coords, session_id=sid, expected_task_id='other')
    assert not agent.interrupted
    assert rpc.interrupt(**coords, session_id=sid, expected_task_id='task')['interrupted']
    assert agent.interrupted
    with pytest.raises(RuntimeStoreError):
        rpc.approve(session_id=sid, request_id='missing', choice='once')
    async def same_loop():
        with pytest.raises(RuntimeStoreError, match='invalid_params'):
            rpc.info(**coords, session_id=sid)
    asyncio.run_coroutine_threadsafe(same_loop(), loop).result()


def test_terminal_callback_follows_canonical_drain_without_polling(owner, monkeypatch):
    from gateway.session_hosted_rpc import HostedRoomAuthorityRPC
    from gateway.session_authority import SessionAuthority
    from gateway.hosted_room_driver import TaskIdentity
    from gateway import session_finite
    from hermes_state_runtime import list_session_admissions
    authority, loop, principal, _ = owner
    rpc = HostedRoomAuthorityRPC(authority, loop, room_id='room', member_id='member', profile='default', principal=principal, authorize=lambda *args: True)
    coords = dict(profile='default', source='bot_room')
    sid = rpc.create(**coords, title='Group: room')['session_id']
    async def execute(authority, ref, row):
        return 'canonical reply'
    # This RPC component fixture supplies the service gate explicitly; daemon
    # coverage exercises the real durable member/task authorizer.
    authority.runner._adapter_for_source = lambda source: authority.runner.adapters[source.platform]
    authority.hosted_room_service = SimpleNamespace(check_admission=lambda ref, row: True)
    monkeypatch.setattr(session_finite, 'execute_finite_admission', execute)
    monkeypatch.setattr(authority, '_schedule', SessionAuthority._schedule.__get__(authority))
    done, receipts = threading.Event(), []
    def terminal(receipt):
        receipts.append(receipt)
        done.set()
    rpc.submit(**coords, session_id=sid, prompt='input', task=TaskIdentity('room','task','thread','turn'), execution_generation=1, on_terminal=terminal)
    assert done.wait(3), 'terminal callback requires no driver polling'
    assert receipts[0]['text'] == 'canonical reply'
    rows = list_session_admissions(authority.db, session_id=sid, pending_only=False)
    assert rows[0]['status'] == 'terminal'


def test_unknown_info_is_inactive_with_hosted_not_canonical_generation(owner):
    from gateway.session_hosted_rpc import HostedRoomAuthorityRPC
    from gateway.hosted_room_driver import TaskIdentity
    from hermes_state_runtime import claim_session_input, begin_runtime_epoch, recover_session_inputs
    authority, loop, principal, _ = owner
    rpc = HostedRoomAuthorityRPC(authority, loop, room_id='room', member_id='member', profile='default', principal=principal, authorize=lambda *args: True)
    coords = dict(profile='default', source='bot_room')
    coords['session_id'] = rpc.create(**coords, title='Group: room')['session_id']
    rpc.submit(**coords, prompt='lost', task=TaskIdentity('room', 'lost', 'thread', 'turn'), execution_generation=17, on_terminal=lambda receipt: None)
    row = claim_session_input(authority.db, epoch=authority.epoch, session_id=coords['session_id'])
    assert row['generation'] != 17
    authority.epoch = begin_runtime_epoch(authority.db, instance_id='cold-info')
    recover_session_inputs(authority.db, epoch=authority.epoch)
    info = rpc.info(**coords)
    assert info['active'] is False
    assert info['status'] == 'unknown'
    assert info['task_id'] == 'lost'
    assert info['execution_generation'] == 17


def test_discard_requires_exact_owned_unknown_tuple_without_replay(owner, monkeypatch):
    from gateway.session_hosted_rpc import HostedRoomAuthorityRPC
    from gateway.hosted_room_driver import TaskIdentity
    from hermes_state_runtime import RuntimeStoreError, list_session_admissions, claim_session_input, begin_runtime_epoch, recover_session_inputs
    authority, loop, principal, _ = owner
    allowed = [True]
    rpc = HostedRoomAuthorityRPC(authority, loop, room_id='room', member_id='member', profile='default', principal=principal, authorize=lambda *args: allowed[0])
    coords = dict(profile='default', source='bot_room')
    sid = rpc.create(**coords, title='Group: room')['session_id']
    coords['session_id'] = sid
    callbacks = []
    rpc.submit(**coords, prompt='lost', task=TaskIdentity('room', 'lost', 'thread', 'turn'), execution_generation=17, on_terminal=callbacks.append)
    exact = dict(expected_task_id='lost', execution_generation=17)
    with pytest.raises(RuntimeStoreError, match='stale_generation'):
        rpc.discard(**coords, **exact)  # queued is not unknown
    row = claim_session_input(authority.db, epoch=authority.epoch, session_id=sid)
    with pytest.raises(RuntimeStoreError, match='stale_generation'):
        rpc.discard(**coords, **exact)  # started is not unknown
    follower = rpc.submit(**coords, prompt='next', task=TaskIdentity('room', 'next', 'thread', 'next'), execution_generation=18, on_terminal=callbacks.append)
    authority.epoch = begin_runtime_epoch(authority.db, instance_id='cold-discard')
    recover_session_inputs(authority.db, epoch=authority.epoch)
    before = list_session_admissions(authority.db, session_id=sid, pending_only=False)
    for changed in ({'expected_task_id': 'other'}, {'execution_generation': row['generation']}, {'execution_generation': True}):
        with pytest.raises(RuntimeStoreError, match='stale_generation|invalid_params'):
            rpc.discard(**coords, **{**exact, **changed})
    allowed[0] = False
    with pytest.raises(RuntimeStoreError, match='permission_denied'):
        rpc.discard(**coords, **exact)
    allowed[0] = True
    with pytest.raises(RuntimeStoreError, match='permission_denied'):
        rpc.discard(**{**coords, 'session_id': 'foreign'}, **exact)
    authority.db._execute_write(lambda conn: conn.execute('UPDATE session_admissions SET principal_id=? WHERE admission_id=?', ('foreign', row['admission_id'])))
    with pytest.raises(RuntimeStoreError, match='stale_generation'):
        rpc.discard(**coords, **exact)
    authority.db._execute_write(lambda conn: conn.execute('UPDATE session_admissions SET principal_id=? WHERE admission_id=?', (principal.subject, row['admission_id'])))
    assert list_session_admissions(authority.db, session_id=sid, pending_only=False) == before
    assert not callbacks
    scheduled = []
    monkeypatch.setattr(authority, '_schedule', scheduled.append)
    receipt = rpc.discard(**coords, **exact)
    assert receipt['discarded'] is True
    assert receipt['task_id'] == 'lost'
    assert receipt['execution_generation'] == 17
    after = list_session_admissions(authority.db, session_id=sid, pending_only=False)
    assert [(r['admission_id'], r['status']) for r in after] == [(row['admission_id'], 'terminal'), (follower['admission_id'], 'queued')]
    assert after[0]['outcome'] == 'interrupted'
    assert scheduled == [rpc.ref]
    assert rpc.discard(**coords, **exact) == receipt
    assert scheduled == [rpc.ref]
    assert rpc.info(**coords)['task_id'] == 'next'
    assert rpc.history(**coords)[-1]['status'] == 'cancelled'


def test_queued_cancellation_is_a_cancelled_receipt_not_storage_unavailable(owner):
    """A queued task that is stopped never ran, so it has no execution result; history/info
    must project it as cancelled instead of failing the whole room read. A completed row with
    a missing result is still storage corruption."""
    from gateway.session_hosted_rpc import HostedRoomAuthorityRPC
    from gateway.hosted_room_driver import TaskIdentity
    from hermes_state_runtime import RuntimeStoreError, claim_session_input, settle_session_input
    authority, loop, principal, _ = owner
    rpc = HostedRoomAuthorityRPC(authority, loop, room_id='room', member_id='member', profile='default', principal=principal, authorize=lambda *args: True)
    coords = dict(profile='default', source='bot_room')
    sid = rpc.create(**coords, title='Group: room')['session_id']
    receipts = []
    rpc.submit(**coords, session_id=sid, prompt='input', task=TaskIdentity('room', 'task', 'thread', 'turn'), execution_generation=1, on_terminal=receipts.append)
    assert rpc.interrupt(**coords, session_id=sid, expected_task_id='task')['interrupted']
    history = rpc.history(**coords, session_id=sid)
    assert history[-1]['status'] == 'cancelled'
    assert history[-1]['task_id'] == 'task'
    assert rpc.info(**coords, session_id=sid)['status'] == 'idle'
    assert receipts and receipts[0]['status'] == 'cancelled'
    # Negative control: a completed admission whose result row vanished is still unreadable.
    rpc.submit(**coords, session_id=sid, prompt='input', task=TaskIdentity('room', 'task2', 'thread', 'turn2'), execution_generation=2, on_terminal=receipts.append)
    row = claim_session_input(authority.db, epoch=authority.epoch, session_id=sid)
    settle_session_input(authority.db, epoch=authority.epoch, admission_id=row['admission_id'], generation=row['generation'], outcome='completed', result=None)
    with pytest.raises(RuntimeStoreError, match='storage_unavailable'):
        rpc.history(**coords, session_id=sid)
