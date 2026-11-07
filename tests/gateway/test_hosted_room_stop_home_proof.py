"""A replacement home's absence cannot finish or replace a legacy live attempt."""
import json

from gateway import hosted_room_driver as tasks
from gateway.session_hosted_member_homes import _HOMES
from tests.gateway.test_hosted_room_attempt_home_moves import _gone, homes as homes_fixture

homes = homes_fixture


def test_stopping_scan_cannot_take_a_replacement_homes_empty_session_as_proof(homes):
    homes.source.homes['other'] = str(homes.b)
    rpc = homes.service._resolve_member_transport(homes.binding, homes.task)
    rpc.create(profile='other', source='bot_room', title='Group: room')
    homes.submit(homes.a)
    homes.claim(homes.a)
    key = _HOMES + json.dumps(['room', 'two', homes.source.identity.task_id, 1], separators=(',', ':'))
    homes.source.db._execute_write(lambda conn: conn.execute('DELETE FROM state_meta WHERE key=?', (key,)))
    homes.source.homes['other'] = str(homes.b)
    runtime = homes.service.runtime
    tasks.begin_task_cancel(homes.service.db_path, homes.source.identity, cancel_id='stop',
        expected_cancel_generation=0, clock=runtime.clock)
    assert runtime._retry_stopping_tasks(homes.binding, runtime._ensure_lease(homes.binding))
    assert homes.task['status'] == 'stopping'
    homes.now += runtime.indeterminate_defer_seconds + 1
    assert not runtime._retry_stopping_tasks(homes.binding, runtime._ensure_lease(homes.binding))
    homes.pending()
    assert homes.rows(homes.a)[0]['status'] == 'started'
    _gone(homes, homes.a)
    assert 'tombstone' in homes.disband()
