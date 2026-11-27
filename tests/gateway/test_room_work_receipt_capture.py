"""A real peer receipt insertion immediately updates passive evidence."""
import json

from gateway import hosted_room_driver as driver
from gateway import hosted_room_work_records as work
from persistence.gateway.hosted_rooms import HostedTaskIdentity, RemoteRunReceipt
from persistence_assembly import open_persistence, profile_scope
from tests.fixtures.storage_holder import raw_holder


def test_remote_receipt_is_captured_without_another_driver_transition(tmp_path):
    with open_persistence(profile_scope(home=tmp_path)) as store:
        store.hosted_rooms.create_room(placement='profile', room_id='room', name='Room', members=[{'profile':'ops','handle':'ops', 'target': {'kind':'peer','installation_id':'peer','profile':'ops'}}], authority_gateway_id='owner')
        store.hosted_rooms.append_event(placement='profile', room_id='room', event_id='input', kind='message.user',
            actor={'kind':'user','id':'alice'}, payload={'text':'hello'}, authority_gateway_id='owner', authority_epoch=1)
        driver.admit_task(store, HostedTaskIdentity('room','task','thread','turn'), placement='profile',
            payload={'target_profile':'ops','prompt':'hello','source_event_seq':1}, clock=lambda: 10)
        record = dict(room_id='room', home_install_id='owner', authority_gateway_id='owner',
            authority_epoch=1, member_id='ops', target_install_id='peer', target_profile='ops',
            task_id='task', execution_generation=1, run_id='run', session_id='session')
        store.hosted_room_peers.upsert_remote_run_receipt(placement='profile', receipt=RemoteRunReceipt(**record))
        with raw_holder(store)._read_ctx() as conn:
            evidence = json.loads(conn.execute(f"SELECT record_json FROM {work.SOURCE_TABLE} WHERE disposition='current' ORDER BY revision DESC LIMIT 1").fetchone()[0])
    assert evidence['receipts'] == [record], json.dumps(evidence)
