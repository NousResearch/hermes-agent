"""Canonical room controls retain the original attempt fence."""
import time
from types import SimpleNamespace

import pytest

from hermes_state import SessionDB
from hermes_state_runtime import RuntimeStoreError, begin_runtime_epoch


def test_unknown_discard_is_exact_and_never_requeues(tmp_path, monkeypatch):
    from gateway.session_hosted_service import CanonicalHostedRoomService
    from gateway import hosted_room_driver as tasks
    from gateway.hosted_rooms import create_room, local_authority_gateway_id
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    class Service(CanonicalHostedRoomService):
        pass
    with SessionDB(tmp_path / 'state.db') as db:
        authority = SimpleNamespace(db=db, profile_id=str(tmp_path), epoch=begin_runtime_epoch(db, instance_id='test'))
        service = Service(authority, None)
        service.authorize_room('alice', 'room', create=True)
        gateway = local_authority_gateway_id()
        create_room(db.db_path, room_id='room', name='Room', authority_gateway_id=gateway, members=[
            {'member_id': 'one', 'profile': 'default', 'handle': 'one'},
            {'member_id': 'two', 'profile': 'other', 'handle': 'two'}])
        identity = tasks.TaskIdentity('room', 'task', 'thread', 'turn')
        tasks.admit_task(db.db_path, identity, payload={'target_profile': 'default', 'target_member_id': 'one', 'source_event_seq': 1, 'prompt': 'frozen'}, clock=time.time)
        lease = tasks.acquire_lease(db.db_path, room_id='room', gateway_id=gateway, authority_epoch=1, process_generation='test', ttl_seconds=30, clock=time.time)
        tasks.start_task(db.db_path, identity, lease, expected_cancel_generation=0, clock=time.time)
        later = time.time() + 31
        monkeypatch.setattr(service.runtime, 'clock', lambda: later)
        recovered = service.runtime._ensure_lease(service.bindings()[0])
        tasks.recover_room(db.db_path, recovered, clock=lambda: later)
        # Only the remote canonical boundary is substituted; the room task/lease is real.
        calls = []
        rpc = SimpleNamespace(ref=SimpleNamespace(session_id='session'),
            info=lambda **kw: {'status': 'unknown'}, results=lambda **kw: [],
            discard=lambda **kw: calls.append(kw) or {'discarded': True})
        monkeypatch.setattr(service, '_resolve_member_transport', lambda *args: rpc)
        monkeypatch.setattr(service.runtime, 'publish_terminal', None)
        args = dict(room_id='room', task_id='task', member_id='one', execution_generation=1)
        action, = service.status('room')['pending_actions']
        assert action == {'kind': 'discard', 'member_id': 'one', 'task_id': 'task', 'execution_generation': 1}
        for change in ({'member_id': 'two'}, {'execution_generation': 2}, {'execution_generation': True}):
            with pytest.raises(RuntimeStoreError):
                service.discard_room_task(**{**args, **change})
        with pytest.raises(RuntimeStoreError, match='unknown_execution'):
            service.retry_room_task(**args)
        assert not calls
        # A deferred attempt is explicit retryable work, but a canonical unknown
        # still blocks it; no generation advances on either refusal.
        tasks.defer_indeterminate_task(db.db_path, identity, recovered,
            expected_execution_generation=1, expected_cancel_generation=0,
            reason='test-deferred', clock=lambda: later)
        with pytest.raises(RuntimeStoreError, match='unknown_execution'):
            service.retry_room_task(**args)
        rpc.info = lambda **kw: {'status': 'idle', 'active': False}
        retried = service.retry_room_task(**args)
        assert retried['status'] == 'queued'
        assert retried['execution_generation'] == 1
        tasks.start_task(db.db_path, identity, recovered, expected_cancel_generation=0, clock=lambda: later)
        later += 31
        recovered = service.runtime._ensure_lease(service.bindings()[0])
        tasks.recover_room(db.db_path, recovered, clock=lambda: later)
        args['execution_generation'] = 2
        result = service.discard_room_task(**args)
        assert result['status'] == 'cancelled'
        assert calls[0]['execution_generation'] == 2
        assert calls[0]['expected_task_id'] == 'task'
        assert tasks.get_task(db.db_path, identity)['execution_generation'] == 2
        assert service.discard_room_task(**args)['status'] == 'cancelled'
        assert len(calls) == 1
        with pytest.raises(RuntimeStoreError):
            service.discard_room_task(**{**args, 'execution_generation': 1})


def _room_with_attempt(tmp_path, monkeypatch, db):
    """A real room with one started-then-recovered (indeterminate) attempt for member 'one'."""
    from gateway.session_hosted_service import CanonicalHostedRoomService
    from gateway import hosted_room_driver as tasks
    from gateway.hosted_rooms import create_room, local_authority_gateway_id
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    authority = SimpleNamespace(db=db, profile_id=str(tmp_path), epoch=begin_runtime_epoch(db, instance_id='test'))
    service = CanonicalHostedRoomService(authority, None)
    service.authorize_room('alice', 'room', create=True)
    gateway = local_authority_gateway_id()
    create_room(db.db_path, room_id='room', name='Room', authority_gateway_id=gateway, members=[
        {'member_id': 'one', 'profile': 'default', 'handle': 'one'}])
    identity = tasks.TaskIdentity('room', 'task', 'thread', 'turn')
    tasks.admit_task(db.db_path, identity, payload={'target_profile': 'default', 'target_member_id': 'one', 'source_event_seq': 1, 'prompt': 'frozen'}, clock=time.time)
    lease = tasks.acquire_lease(db.db_path, room_id='room', gateway_id=gateway, authority_epoch=1, process_generation='test', ttl_seconds=30, clock=time.time)
    tasks.start_task(db.db_path, identity, lease, expected_cancel_generation=0, clock=time.time)
    later = time.time() + 31
    monkeypatch.setattr(service.runtime, 'clock', lambda: later)
    recovered = service.runtime._ensure_lease(service.bindings()[0])
    tasks.recover_room(db.db_path, recovered, clock=lambda: later)
    monkeypatch.setattr(service.runtime, 'publish_terminal', None)
    return service, tasks, identity


def _fresh_owner_rpc(tmp_path, monkeypatch, replies):
    """The real cross-profile transport; only the owner socket is replaced. A fresh one has no `ref`."""
    from gateway import session_hosted_transport as transport
    sent = []
    def owner_request(home, kind, request):
        sent.append((request['operation'], request['params']))
        return replies[request['operation']]
    monkeypatch.setattr(transport, 'owner_request', owner_request)
    rpc = transport.HostedRoomOwnerRPC(home=tmp_path / 'other', source_home=tmp_path,
                                       room_id='room', member_id='one', profile='default')
    assert not hasattr(rpc, 'ref')
    return rpc, sent


@pytest.mark.parametrize('outcome', ['cancelled', 'settled', 'missing_session'])
def test_discard_on_fresh_member_transport_resolves_the_session_first(tmp_path, monkeypatch, outcome):
    # A fresh cross-profile transport must identify the session before resolving this attempt.
    with SessionDB(tmp_path / 'state.db') as db:
        service, tasks, identity = _room_with_attempt(tmp_path, monkeypatch, db)
        session = None if outcome == 'missing_session' else {'session_id': 'member-session'}
        rpc, sent = _fresh_owner_rpc(tmp_path, monkeypatch, {
            'resolve_exact': session,
            'discard': ({'status': 'settled', 'text': 'Retained answer.'}
                        if outcome == 'settled' else {'discarded': True})})
        monkeypatch.setattr(service, '_resolve_member_transport', lambda *args: rpc)
        args = dict(room_id='room', task_id='task', member_id='one', execution_generation=1)
        if session is None:
            with pytest.raises(RuntimeStoreError, match='stale_generation'):
                service.discard_room_task(**args)
            assert tasks.get_task(db.db_path, identity)['status'] == 'indeterminate'
            assert [op for op, _ in sent] == ['resolve_exact']
        else:
            result = service.discard_room_task(**args)
            assert result['status'] == outcome
            assert result['execution_generation'] == 1
            if outcome == 'settled':
                assert result['result']['text'] == 'Retained answer.'
            assert [op for op, _ in sent] == ['resolve_exact', 'discard']
            assert sent[1][1] == {
                'profile': 'default', 'source': 'bot_room', 'session_id': 'member-session',
                'expected_task_id': 'task', 'execution_generation': 1}
        assert sent[0][1] == {'profile': 'default', 'title': 'Group: room', 'source': 'bot_room'}


def test_approve_skips_member_transports_that_have_no_session_ref_yet(tmp_path, monkeypatch):
    # A fresh transport must not break approval lookup for another member's session.
    from gateway.session_hosted_service import CanonicalHostedRoomService
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    with SessionDB(tmp_path / 'state.db') as db:
        authority = SimpleNamespace(db=db, profile_id=str(tmp_path), epoch=begin_runtime_epoch(db, instance_id='test'))
        service = CanonicalHostedRoomService(authority, None)
        fresh, sent = _fresh_owner_rpc(tmp_path, monkeypatch, {})
        approved = []
        owner = SimpleNamespace(ref=SimpleNamespace(session_id='member-session'),
                                approve=lambda **kw: approved.append(kw) or {'ok': True})
        service.member_rpcs = {'fresh': fresh, 'owner': owner}
        assert service.approve(session_id='member-session', request_id='r1', choice='once') == {'ok': True}
        assert approved == [{'session_id': 'member-session', 'request_id': 'r1', 'choice': 'once'}]
        with pytest.raises(RuntimeStoreError, match='permission_denied'):
            service.approve(session_id='unknown-session', request_id='r1', choice='once')
        assert not sent
