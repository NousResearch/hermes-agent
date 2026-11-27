"""Real driver transitions retain passive evidence without becoming replay authority."""
import json

import pytest

from gateway import hosted_room_driver as driver
from gateway import hosted_room_work_records as work
from persistence.gateway.hosted_rooms import HostedTaskIdentity, InvalidTaskTransitionError
from persistence_assembly import open_persistence, profile_scope
from tests.fixtures.storage_holder import raw_holder


@pytest.fixture
def store(tmp_path):
    store = open_persistence(profile_scope(home=tmp_path))
    yield store
    store.close()


def snapshot(store):
    with raw_holder(store)._read_ctx() as conn:
        row = conn.execute(f"SELECT record_json FROM {work.SOURCE_TABLE} WHERE disposition='current'").fetchone()
        assert row is not None
        return work.validate(json.loads(row[0]))


def test_driver_transitions_capture_without_manual_publication(store):
    store.hosted_rooms.create_room(placement='profile', room_id='room', name='Room',
                                   members=[{'profile': 'ops', 'handle': 'ops'}], authority_gateway_id='owner', now=10)
    store.hosted_rooms.append_event(placement='profile', room_id='room', event_id='input', kind='message.user',
        actor={'kind': 'user', 'id': 'alice'}, payload={'text': 'hello'},
        authority_gateway_id='owner', authority_epoch=1)
    identity = HostedTaskIdentity('room', 'task', 'thread', 'turn')
    payload = {'target_profile': 'ops', 'prompt': 'private input', 'source_event_seq': 1}
    driver.admit_task(store, identity, placement='profile', payload=payload, clock=lambda: 10)
    queued = snapshot(store)
    assert queued['tasks'][0]['phase'] == 'queued'
    assert 'private input' not in work.encode(queued)
    driver.admit_task(store, identity, placement='profile', payload=payload, clock=lambda: 10)
    assert snapshot(store) == queued
    lease = driver.acquire_lease(store, placement='profile', room_id='room', gateway_id='owner', authority_epoch=1,
        process_generation='process-a', ttl_seconds=10, clock=lambda: 10)
    driver.start_task(store, identity, lease, placement='profile', expected_cancel_generation=0, clock=lambda: 10)
    running = snapshot(store)
    assert running['revision'] > queued['revision']
    assert running['tasks'][0]['phase'] == 'running'
    successor = driver.acquire_lease(store, placement='profile', room_id='room', gateway_id='owner', authority_epoch=1,
        process_generation='process-b', ttl_seconds=10, clock=lambda: 21)
    driver.recover_room(store, successor, placement='profile', clock=lambda: 21)
    unknown = snapshot(store)
    assert unknown['tasks'][0]['phase'] == 'indeterminate'
    with pytest.raises(InvalidTaskTransitionError):
        driver.start_task(store, identity, successor, placement='profile', expected_cancel_generation=0,
                          clock=lambda: 21)
    assert snapshot(store) == unknown
    driver.begin_task_cancel(store, identity, placement='profile', cancel_id='stop', expected_cancel_generation=0,
                             clock=lambda: 21)
    assert snapshot(store)['tasks'][0]['phase'] == 'stopping'
    driver.complete_task_cancel(store, identity, placement='profile', cancel_id='stop', expected_cancel_generation=1,
                                clock=lambda: 21)
    assert snapshot(store)['tasks'][0]['phase'] == 'cancelled'


def test_broken_passive_evidence_does_not_authorize_or_block_stop(store):
    store.hosted_rooms.create_room(placement='profile', room_id='room', name='Room',
                                   members=[{'profile': 'ops', 'handle': 'ops'}], authority_gateway_id='owner', now=10)
    store.hosted_rooms.append_event(placement='profile', room_id='room', event_id='input', kind='message.user',
        actor={'kind': 'user', 'id': 'alice'}, payload={'text': 'hello'},
        authority_gateway_id='owner', authority_epoch=1)
    identity = HostedTaskIdentity('room', 'task', 'thread', 'turn')
    driver.admit_task(store, identity, placement='profile',
                      payload={'target_profile': 'ops', 'prompt': 'hello', 'source_event_seq': 1}, clock=lambda: 10)
    snapshot(store)
    raw_holder(store)._execute_write(lambda conn: conn.execute(f"UPDATE {work.SOURCE_TABLE} SET record_json='broken'"))
    driver.cancel_task(store, identity, placement='profile', cancel_id='stop', expected_cancel_generation=0,
                       clock=lambda: 10)
    assert driver.get_task(store, identity, placement='profile')['status'] == 'cancelled'
    with raw_holder(store)._read_ctx() as conn:
        assert tuple(conn.execute(f"SELECT disposition,record_json FROM {work.SOURCE_TABLE}").fetchone()) == (
            'invalid', 'broken')
