"""A turn deferred before attempt homes were recorded, after its profile moved from home A to B
while A's gateway stays up: A is asked for that exact turn's Stop. Its "holds nothing" settles the
turn, a live hold keeps it (and gets the Stop), and Retry never starts another generation while A
could still run the old one."""
import json
import sqlite3

import pytest

from hermes_agent.hermes_state_runtime import RuntimeStoreError
from tests.gateway import test_hosted_room_attempt_home_moves as _moves
from tests.gateway.test_hosted_room_attempt_home_moves import _HOMES, _gone

homes = _moves.homes  # the two-home fixture, shared with the moved-profile tests


def _legacy_move(homes):
    """Defer the turn at A, forget its attempt home (as a turn from before they were recorded),
    then point the profile at B."""
    homes.submit(homes.a)
    homes.defer(homes.a)
    with sqlite3.connect(homes.source.db.db_path) as conn:
        for (key,) in conn.execute('SELECT key FROM state_meta WHERE substr(key, 1, ?)=?',
                                   (len(_HOMES), _HOMES)).fetchall():
            if len(json.loads(key[len(_HOMES):])) == 4:
                conn.execute('DELETE FROM state_meta WHERE key=?', (key,))
    homes.source.homes['other'] = str(homes.b)
    homes.calls.clear()


def _retry(homes):
    return homes.service.retry_room_task('room', member_id='two', task_id=homes.source.identity.task_id,
                                         execution_generation=homes.task['execution_generation'])


def test_disband_settles_once_the_live_older_home_answers_it_holds_nothing(homes):
    _legacy_move(homes)
    homes.finish(homes.a, homes.claim(homes.a))  # A's run of it ended; A stays up
    homes.calls.clear()
    assert 'tombstone' in homes.disband()
    assert homes.a in {call[0] for call in homes.calls}  # A answered for this turn, not only B


def test_a_live_older_home_holding_the_turn_is_stopped_and_holds_disband(homes):
    _legacy_move(homes)
    row = homes.claim(homes.a)  # A is running it
    with pytest.raises(Exception) as refused:
        homes.disband()
    assert 'still stopping' in str(refused.value) or 'room_stop_pending' in str(refused.value)
    assert (homes.a, 'interrupt') in {call[:2] for call in homes.calls}  # its Stop went to A
    assert homes.task['status'] == 'deferred'
    homes.finish(homes.a, row)  # A's run ends
    assert 'tombstone' in homes.disband()


def test_retry_waits_for_a_live_older_home_that_may_run_the_turn(homes):
    _legacy_move(homes)
    homes.claim(homes.a)
    generation = homes.task['execution_generation']
    with pytest.raises(RuntimeStoreError) as refused:
        _retry(homes)
    assert refused.value.reason == 'session_busy'
    assert refused.value.field == 'home'  # the detail names the older home
    assert homes.task['status'] == 'deferred' and homes.task['execution_generation'] == generation
    _gone(homes, homes.a)
    _retry(homes)
    assert homes.task['status'] == 'queued'


def test_an_older_home_restarted_mid_turn_settles_like_a_live_stop(homes):
    # A restart leaves A's started input 'unknown': its agent died with the old process, so it
    # reads inactive, as a live Stop's acknowledgement does, and Disband no longer waits on A.
    _legacy_move(homes)
    row = homes.claim(homes.a)
    db = homes.targets[homes.a].db
    db._execute_write(lambda conn: conn.execute(
        "UPDATE session_admissions SET status='unknown' WHERE admission_id=?", (row['admission_id'],)))
    assert 'tombstone' in homes.disband()
