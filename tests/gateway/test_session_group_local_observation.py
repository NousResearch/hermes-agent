"""The same recovery contract applies to a local member owned by another profile daemon."""

import asyncio
import time
from types import SimpleNamespace

import pytest

from gateway import hosted_room_driver as driver, run
from gateway.session_authority import admission_row
from gateway.session_hosted_service import CanonicalHostedRoomService
from gateway.session_hosted_transport import HostedRoomOwnerRPC, install_hosted_transport
from gateway.session_persistence import authority_persistence
from persistence.gateway.hosted_rooms import HostedTaskIdentity
from persistence_assembly import open_persistence, profile_scope
from tests.gateway.test_session_hosted_transport import _server
from tests.tui_gateway.test_hosted_room_driver_runtime import FakeSessionRPC


def list_session_admissions(authority, *, session_id, pending_only=True):
    """The session's admission rows, as the owner's store lists them."""
    return [admission_row(row) for row in authority_persistence(authority).admissions.list(
        session_id=session_id, pending_only=pending_only)]


@pytest.mark.parametrize('retry_during_wait', [False, True])
def test_renewed_local_member_turn_defers_when_its_owner_socket_disappears(
    hosted_owner, tmp_path, monkeypatch, retry_during_wait
):
    target_authority, loop, _, _ = hosted_owner
    source, target = tmp_path / 'source', tmp_path / 'profiles' / 'other'
    source.mkdir(mode=0o700)
    target.mkdir(mode=0o700, parents=True)
    target_authority.profile_id = str(target)
    monkeypatch.setenv('HERMES_HOME', str(source))
    config = run._load_gateway_config()
    monkeypatch.setattr(run, '_load_gateway_config', lambda: {
        **config, 'hosted_rooms': {'profiles': {'other': str(target)}}})
    with open_persistence(profile_scope(home=source)) as db:
        authority = SimpleNamespace(profile_id=str(source), db=db, persistence=db,
                                    epoch=db.runtime.begin_epoch(instance_id='source'))
        service = CanonicalHostedRoomService(authority, loop)
        service.insert_room(room_id='local', name='Local group chat', owner='alice', members=[
            {'member_id': 'host', 'profile': 'default', 'handle': 'host'},
            {'member_id': 'helper', 'profile': 'other', 'handle': 'helper'}])
        runtime, binding = service.runtime, service.bindings()[0]
        runtime.prepare_room = runtime.publish_terminal = None
        now = [time.time()]
        start = now[0]
        runtime.clock = lambda: now[0]
        tasks = dict(placement='profile')
        older = HostedTaskIdentity('local', 'older', 'thread', 'older-turn')
        driver.admit_task(db, older, **tasks, payload={'target_profile': 'default',
            'target_member_id': 'host', 'source_event_seq': 1, 'prompt': 'Earlier input'}, clock=runtime.clock)
        old_lease = driver.acquire_lease(db, **tasks, room_id='local', gateway_id=binding.gateway_id,
            authority_epoch=1, process_generation='previous', ttl_seconds=1, clock=lambda: start - 2)
        driver.start_task(db, older, old_lease, **tasks, expected_cancel_generation=0, clock=lambda: start - 2)
        lease = runtime._ensure_lease(binding)
        driver.recover_room(db, lease, **tasks, clock=runtime.clock)
        driver.defer_indeterminate_task(db, older, lease, **tasks, expected_execution_generation=1,
            expected_cancel_generation=0, reason='member_unavailable', clock=runtime.clock)
        identity = HostedTaskIdentity('local', 'task', 'thread', 'turn')
        driver.admit_task(db, identity, **tasks, payload={'target_profile': 'other',
            'target_member_id': 'helper', 'source_event_seq': 2, 'prompt': 'input'}, clock=runtime.clock)
        rpc = service._resolve_member_transport(binding, driver.get_task(db, identity, **tasks))
        assert isinstance(rpc, HostedRoomOwnerRPC)
        healthy = FakeSessionRPC()
        healthy.ref = SimpleNamespace(session_id=healthy.add_session(profile='default', title='Group: local'))
        resolve = service._resolve_member_transport
        def member_transport(binding, task):
            assert not runtime._status_lock.locked()
            return healthy if task['payload']['target_member_id'] == 'host' else resolve(binding, task)
        monkeypatch.setattr(service, '_resolve_member_transport', member_transport)
        runtime.transport_resolver = member_transport
        servers = [_server(source), _server(target)]
        install_hosted_transport(servers[0], authority, loop, attest=service.attest)
        install_hosted_transport(servers[1], target_authority, loop, attest=lambda *a: None)
        for server in servers:
            assert asyncio.run_coroutine_threadsafe(server.start(), loop).result(10)
        history, reads, retries = rpc.history, [], []

        def tick(_timeout):
            assert now[0] < start + 100, 'observation did not retire'
            now[0] += 16 if now[0] < start + 32 else 1
            if retry_during_wait and now[0] == start + 40:
                retried = service.retry_room_task('local', member_id='host', task_id='older', execution_generation=1)
                assert retried['status'] == 'queued'
                assert runtime._leases['local'].lease_generation == lease.lease_generation
                retries.append(retried)

        monkeypatch.setattr(runtime._wake, 'wait', tick)

        def lose_owner_after_renewal(**kwargs):
            reads.append(now[0])
            if len(reads) == 2:
                now[0] += 16
                asyncio.run_coroutine_threadsafe(servers[1].stop(), loop).result(10)
            return history(**kwargs)

        monkeypatch.setattr(rpc, 'history', lose_owner_after_renewal)
        try:
            runtime._process_room(binding)
            assert reads[:2] == [start, start + 16], runtime._last_error
            assert 'observation failed after submit' in runtime._last_error
            assert len(list_session_admissions(target_authority,
                       session_id=rpc.ref.session_id, pending_only=False)) == 1
            assert bool(retries) == retry_during_wait
            assert now[0] == start + 62
            assert driver.get_task(db, identity, **tasks)['status'] == 'indeterminate'
            assert driver.get_task(db, identity, **tasks)['indeterminate_at'] == now[0]
            later = HostedTaskIdentity('local', 'later', 'thread', 'later-turn')
            driver.admit_task(db, later, **tasks, payload={'target_profile': 'default',
                'target_member_id': 'host', 'source_event_seq': 3, 'prompt': 'Continue'}, clock=runtime.clock)
            now[0] += runtime.indeterminate_defer_seconds
            runtime._process_room(binding)
            assert driver.get_task(db, identity, **tasks)['status'] == 'deferred'
            assert driver.get_task(db, later, **tasks)['status'] == 'settled'
            if retry_during_wait:
                assert driver.get_task(db, older, **tasks)['execution_generation'] == 2
                assert driver.get_task(db, older, **tasks)['status'] == 'settled'
        finally:
            for server in servers:
                asyncio.run_coroutine_threadsafe(server.stop(), loop).result(10)
            if rpc._monitor is not None:
                rpc._monitor.join(10)
                assert not rpc._monitor.is_alive()
