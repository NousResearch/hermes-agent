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
            info=lambda **kw: {'status': 'unknown'},
            discard=lambda **kw: calls.append(kw) or {'discarded': True})
        monkeypatch.setattr(service, '_resolve_member_transport', lambda *args: rpc)
        monkeypatch.setattr(service.runtime, 'publish_terminal', None)
        args = dict(room_id='room', task_id='task', member_id='one', execution_generation=1)
        action, = service.status('room')['pending_actions']
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
