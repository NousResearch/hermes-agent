"""Real driver transitions retain passive evidence without becoming replay authority."""
import json
from contextlib import closing
import sqlite3

import pytest

from gateway import hosted_room_driver as driver
from gateway import hosted_room_work_records as work
from gateway import hosted_rooms as rooms


def snapshot(path):
    with closing(sqlite3.connect(path)) as conn:
        row = conn.execute(f"SELECT record_json FROM {work.SOURCE_TABLE} WHERE disposition='current'").fetchone()
        assert row is not None
        return work.validate(json.loads(row[0]))


def test_driver_transitions_capture_without_manual_publication(tmp_path):
    path = tmp_path / 'state.db'
    rooms.create_room(path, room_id='room', name='Room', members=[{'profile': 'ops', 'handle': 'ops'}],
                      authority_gateway_id='owner', now=10)
    rooms.append_event(path, room_id='room', event_id='input', kind='message.user',
        actor={'kind': 'user', 'id': 'alice'}, payload={'text': 'hello'},
        authority_gateway_id='owner', authority_epoch=1)
    identity = driver.TaskIdentity('room', 'task', 'thread', 'turn')
    payload = {'target_profile': 'ops', 'prompt': 'private input', 'source_event_seq': 1}
    driver.admit_task(path, identity, payload=payload, clock=lambda: 10)
    queued = snapshot(path)
    assert queued['tasks'][0]['phase'] == 'queued'
    assert 'private input' not in work.encode(queued)
    driver.admit_task(path, identity, payload=payload, clock=lambda: 10)
    assert snapshot(path) == queued
    lease = driver.acquire_lease(path, room_id='room', gateway_id='owner', authority_epoch=1,
        process_generation='process-a', ttl_seconds=10, clock=lambda: 10)
    driver.start_task(path, identity, lease, expected_cancel_generation=0, clock=lambda: 10)
    running = snapshot(path)
    assert running['revision'] > queued['revision']
    assert running['tasks'][0]['phase'] == 'running'
    successor = driver.acquire_lease(path, room_id='room', gateway_id='owner', authority_epoch=1,
        process_generation='process-b', ttl_seconds=10, clock=lambda: 21)
    driver.recover_room(path, successor, clock=lambda: 21)
    unknown = snapshot(path)
    assert unknown['tasks'][0]['phase'] == 'indeterminate'
    with pytest.raises(driver.InvalidTaskTransitionError):
        driver.start_task(path, identity, successor, expected_cancel_generation=0, clock=lambda: 21)
    assert snapshot(path) == unknown
    driver.begin_task_cancel(path, identity, cancel_id='stop', expected_cancel_generation=0, clock=lambda: 21)
    assert snapshot(path)['tasks'][0]['phase'] == 'stopping'
    driver.complete_task_cancel(path, identity, cancel_id='stop', expected_cancel_generation=1, clock=lambda: 21)
    assert snapshot(path)['tasks'][0]['phase'] == 'cancelled'


def test_broken_passive_evidence_does_not_authorize_or_block_stop(tmp_path):
    path = tmp_path / 'state.db'
    rooms.create_room(path, room_id='room', name='Room', members=[{'profile': 'ops', 'handle': 'ops'}],
                      authority_gateway_id='owner', now=10)
    rooms.append_event(path, room_id='room', event_id='input', kind='message.user',
        actor={'kind': 'user', 'id': 'alice'}, payload={'text': 'hello'},
        authority_gateway_id='owner', authority_epoch=1)
    identity = driver.TaskIdentity('room', 'task', 'thread', 'turn')
    driver.admit_task(path, identity, payload={'target_profile': 'ops', 'prompt': 'hello', 'source_event_seq': 1}, clock=lambda: 10)
    snapshot(path)
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute(f"UPDATE {work.SOURCE_TABLE} SET record_json='broken'")
    driver.cancel_task(path, identity, cancel_id='stop', expected_cancel_generation=0, clock=lambda: 10)
    assert driver.get_task(path, identity)['status'] == 'cancelled'
    with closing(sqlite3.connect(path)) as conn:
        assert conn.execute(f"SELECT disposition,record_json FROM {work.SOURCE_TABLE}").fetchone() == ('invalid', 'broken')
