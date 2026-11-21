"""Canonical room controls retain the original attempt fence."""
import time
from types import SimpleNamespace

import pytest

from tests.gateway.fixtures.storage_handles import close_storage_handles, storage_handle
from persistence.core.errors import AuthorityRefusal
from persistence_assembly import open_persistence, profile_scope
from persistence.gateway.hosted_rooms import HostedTaskIdentity


def test_unknown_discard_is_exact_and_never_requeues(tmp_path, monkeypatch):
    from gateway.session_hosted_service import CanonicalHostedRoomService
    from gateway import hosted_room_driver as tasks
    from gateway.hosted_rooms import local_authority_gateway_id
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    class Service(CanonicalHostedRoomService):
        pass
    with open_persistence(profile_scope(home=tmp_path)) as db:
        authority = SimpleNamespace(db=db, persistence=storage_handle(tmp_path), profile_id=str(tmp_path), epoch=db.runtime.begin_epoch(instance_id='test'))
        service = Service(authority, None)
        gateway = local_authority_gateway_id()
        service.insert_room(room_id='room', name='Room', owner='alice', members=[
            {'member_id': 'one', 'profile': 'default', 'handle': 'one'},
            {'member_id': 'two', 'profile': 'other', 'handle': 'two'}])
        identity = HostedTaskIdentity('room', 'task', 'thread', 'turn')
        tasks.admit_task(db, identity, placement='profile', payload={'target_profile': 'default', 'target_member_id': 'one', 'source_event_seq': 1, 'prompt': 'frozen'}, clock=time.time)
        lease = tasks.acquire_lease(db, placement='profile', room_id='room', gateway_id=gateway, authority_epoch=1, process_generation='test', ttl_seconds=30, clock=time.time)
        tasks.start_task(db, identity, lease, placement='profile', expected_cancel_generation=0, clock=time.time)
        later = time.time() + 31
        monkeypatch.setattr(service.runtime, 'clock', lambda: later)
        recovered = service.runtime._ensure_lease(service.bindings()[0])
        tasks.recover_room(db, recovered, placement='profile', clock=lambda: later)
        # Only the remote canonical boundary is substituted; the room task/lease is real.
        calls = []
        rpc = SimpleNamespace(ref=SimpleNamespace(session_id='session'),
            info=lambda **kw: {'status': 'unknown'}, results=lambda **kw: [],
            discard=lambda **kw: calls.append(kw) or {'discarded': True})
        monkeypatch.setattr(service, '_resolve_member_transport', lambda *args: rpc)
        monkeypatch.setattr(service.runtime, 'publish_terminal', None)
        args = dict(room_id='room', task_id='task', member_id='one', execution_generation=1)
        with monkeypatch.context() as scoped:
            # groups.state reads only the shown (live/retryable) rows, never every retained task.
            scoped.setattr(tasks, 'list_tasks', lambda *a, **k: pytest.fail('status() decoded every task'))
            action, = service.status('room')['pending_actions']
            # A native room lists its exact attempt controls after the approval cards.
            service._set_pending_action(
                'room', 'one', {'kind': 'approval', 'task_id': 'task', 'execution_generation': 1, 'request_id': 'r2'})
            assert [a['kind'] for a in service.status('room')['pending_actions']] == ['approval', 'discard']
            service._set_pending_action('room', 'one', None)
        assert action == {'kind': 'discard', 'member_id': 'one', 'task_id': 'task', 'execution_generation': 1}
        for change in ({'member_id': 'two'}, {'execution_generation': 2}, {'execution_generation': True}):
            with pytest.raises(AuthorityRefusal):
                service.discard_room_task(**{**args, **change})
        with pytest.raises(AuthorityRefusal, match='unknown_execution'):
            service.retry_room_task(**args)
        assert not calls
        # A deferred attempt is explicit retryable work, but a canonical unknown
        # still blocks it; no generation advances on either refusal.
        tasks.defer_indeterminate_task(db, identity, recovered, placement='profile',
            expected_execution_generation=1, expected_cancel_generation=0,
            reason='test-deferred', clock=lambda: later)
        with pytest.raises(AuthorityRefusal, match='unknown_execution'):
            service.retry_room_task(**args)
        rpc.info = lambda **kw: {'status': 'idle', 'active': False}
        retried = service.retry_room_task(**args)
        assert retried['status'] == 'queued'
        assert retried['execution_generation'] == 1
        tasks.start_task(db, identity, recovered, placement='profile', expected_cancel_generation=0, clock=lambda: later)
        later += 31
        recovered = service.runtime._ensure_lease(service.bindings()[0])
        tasks.recover_room(db, recovered, placement='profile', clock=lambda: later)
        args['execution_generation'] = 2
        result = service.discard_room_task(**args)
        assert result['status'] == 'cancelled'
        assert calls[0]['execution_generation'] == 2
        assert calls[0]['expected_task_id'] == 'task'
        assert tasks.get_task(db, identity, placement='profile')['execution_generation'] == 2
        assert service.discard_room_task(**args)['status'] == 'cancelled'
        assert len(calls) == 1
        with pytest.raises(AuthorityRefusal):
            service.discard_room_task(**{**args, 'execution_generation': 1})


def _room_with_attempt(tmp_path, monkeypatch, db):
    """A real room with one started-then-recovered (indeterminate) attempt for member 'one'."""
    from gateway.session_hosted_service import CanonicalHostedRoomService
    from gateway import hosted_room_driver as tasks
    from gateway.hosted_rooms import local_authority_gateway_id
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    authority = SimpleNamespace(db=db, persistence=db, profile_id=str(tmp_path), epoch=db.runtime.begin_epoch(instance_id='test'))
    service = CanonicalHostedRoomService(authority, None)
    gateway = local_authority_gateway_id()
    service.insert_room(room_id='room', name='Room', owner='alice', members=[
        {'member_id': 'one', 'profile': 'default', 'handle': 'one'}])
    identity = HostedTaskIdentity('room', 'task', 'thread', 'turn')
    tasks.admit_task(db, identity, placement='profile', payload={'target_profile': 'default', 'target_member_id': 'one', 'source_event_seq': 1, 'prompt': 'frozen'}, clock=time.time)
    lease = tasks.acquire_lease(db, placement='profile', room_id='room', gateway_id=gateway, authority_epoch=1, process_generation='test', ttl_seconds=30, clock=time.time)
    tasks.start_task(db, identity, lease, placement='profile', expected_cancel_generation=0, clock=time.time)
    later = time.time() + 31
    monkeypatch.setattr(service.runtime, 'clock', lambda: later)
    recovered = service.runtime._ensure_lease(service.bindings()[0])
    tasks.recover_room(db, recovered, placement='profile', clock=lambda: later)
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
    with open_persistence(profile_scope(home=tmp_path)) as db:
        service, tasks, identity = _room_with_attempt(tmp_path, monkeypatch, db)
        session = None if outcome == 'missing_session' else {'session_id': 'member-session'}
        rpc, sent = _fresh_owner_rpc(tmp_path, monkeypatch, {
            'resolve_exact': session,
            'discard': ({'status': 'settled', 'text': 'Retained answer.'}
                        if outcome == 'settled' else {'discarded': True})})
        monkeypatch.setattr(service, '_resolve_member_transport', lambda *args: rpc)
        args = dict(room_id='room', task_id='task', member_id='one', execution_generation=1)
        if session is None:
            with pytest.raises(AuthorityRefusal, match='stale_generation'):
                service.discard_room_task(**args)
            assert tasks.get_task(db, identity, placement='profile')['status'] == 'indeterminate'
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
    with open_persistence(profile_scope(home=tmp_path)) as db:
        authority = SimpleNamespace(db=db, persistence=db, profile_id=str(tmp_path), epoch=db.runtime.begin_epoch(instance_id='test'))
        service = CanonicalHostedRoomService(authority, None)
        fresh, sent = _fresh_owner_rpc(tmp_path, monkeypatch, {})
        approved = []
        owner = SimpleNamespace(ref=SimpleNamespace(session_id='member-session'),
                                approve=lambda **kw: approved.append(kw) or {'ok': True})
        service.member_rpcs = {'fresh': fresh, 'owner': owner}
        assert service.approve(session_id='member-session', request_id='r1', choice='once') == {'ok': True}
        assert approved == [{'session_id': 'member-session', 'request_id': 'r1', 'choice': 'once'}]
        with pytest.raises(AuthorityRefusal, match='permission_denied'):
            service.approve(session_id='unknown-session', request_id='r1', choice='once')
        assert not sent
