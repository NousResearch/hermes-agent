"""A real peer receipt insertion immediately updates passive evidence."""
import json
import sqlite3
from contextlib import closing

from gateway import hosted_room_driver as driver, hosted_rooms as rooms
from gateway import hosted_room_work_records as work


def test_remote_receipt_is_captured_without_another_driver_transition(tmp_path):
    db = tmp_path / 'state.db'
    rooms.create_room(db, room_id='room', name='Room', members=[{'profile':'ops','handle':'ops', 'target': {'kind':'peer','installation_id':'peer','profile':'ops'}}], authority_gateway_id='owner')
    rooms.append_event(db, room_id='room', event_id='input', kind='message.user',
        actor={'kind':'user','id':'alice'}, payload={'text':'hello'}, authority_gateway_id='owner', authority_epoch=1)
    driver.admit_task(db, driver.TaskIdentity('room','task','thread','turn'),
        payload={'target_profile':'ops','prompt':'hello','source_event_seq':1}, clock=lambda: 10)
    record = dict(room_id='room', home_install_id='owner', authority_gateway_id='owner',
        authority_epoch=1, member_id='ops', target_install_id='peer', target_profile='ops',
        task_id='task', execution_generation=1, run_id='run', session_id='session')
    rooms.upsert_remote_run_receipt(db, record=record)
    with closing(sqlite3.connect(db)) as conn:
        evidence = json.loads(conn.execute(f"SELECT record_json FROM {work.SOURCE_TABLE} WHERE disposition='current' ORDER BY revision DESC LIMIT 1").fetchone()[0])
    assert evidence['receipts'] == [record], json.dumps(evidence)
