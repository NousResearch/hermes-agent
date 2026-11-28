"""Public Retry and retirement behavior for an exact deferred attempt."""
import pytest

from gateway import hosted_room_driver as tasks
from gateway.session_hosted_service import CanonicalHostedRoomService
from tests.fixtures.storage_holder import raw_holder
from tests.gateway.test_hosted_room_stop_terminal import _turn, _finish
from tests.gateway.test_hosted_room_stop_attempt import _defer


@pytest.mark.parametrize('state', ['completed', 'interrupted'])
def test_retry_recovers_the_selected_attempt_after_restart(hosted_owner, monkeypatch, state):
    turn = _turn(hosted_owner, monkeypatch)
    _defer(turn)
    _finish(turn, state, 'retained answer')
    homes = turn.service.profile_homes()
    tasks.release_lease(turn.service.store, turn.lease, placement='profile', clock=turn.service.runtime.clock)
    service = CanonicalHostedRoomService(turn.authority, turn.member.loop)
    monkeypatch.setattr(service, 'profile_homes', lambda: homes)
    args = dict(room_id='room', member_id='one', task_id=turn.identity.task_id, execution_generation=1)
    result = service.retry_room_task(**args)
    assert result['status'] == ('settled' if state == 'completed' else 'cancelled')
    assert result['execution_generation'] == 1
    if state == 'completed':
        assert result['result']['text'] == 'retained answer'


@pytest.mark.parametrize('retirement', ['delete', 'prune', 'empty_sweep'])
def test_session_retirement_reclaims_closed_keys(hosted_owner, monkeypatch, tmp_path, retirement):
    from tests.fixtures.storage_holder import reclaim_tombstones
    turn = _turn(hosted_owner, monkeypatch, admitted=False)
    turn.service.stop_room('room', cancel_id='unused')
    db, sid = turn.authority.persistence, turn.sid
    with raw_holder(db)._read_ctx() as conn:
        assert conn.execute("SELECT 1 FROM state_meta WHERE key GLOB 'gateway.hosted.closed_attempt.v1:*'").fetchone()
    if retirement == 'delete':
        assert db.sessions.delete(sid, media_root=str(tmp_path / 'media'))
    elif retirement == 'empty_sweep':
        db.sessions.set_title(sid, None)
        assert db.sessions.delete_if_empty(sid, media_root=str(tmp_path / 'media'))
    else:
        db.sessions.end(sid, 'done')
        assert db.maintenance.prune_sessions(older_than_days=None, media_root=str(tmp_path / 'media')).count == 1
    reclaim_tombstones(db, older_than=-1)
    assert db.runtime.retired_session(sid)
    with raw_holder(db)._read_ctx() as conn:
        assert conn.execute("SELECT key FROM state_meta WHERE key GLOB 'gateway.hosted.closed_attempt.v1:*'").fetchall() == []


def test_stop_racing_session_retirement_cannot_recreate_a_closed_key(hosted_owner, monkeypatch, tmp_path):
    from gateway import session_runtime_workers
    turn = _turn(hosted_owner, monkeypatch, admitted=False)
    retired = []
    original = session_runtime_workers.tracked_write

    async def retire_before_write(authority, work):
        assert authority.persistence.sessions.delete(turn.sid, media_root=str(tmp_path / 'media'))
        retired.append(turn.sid)
        return await original(authority, work)

    monkeypatch.setattr(session_runtime_workers, 'tracked_write', retire_before_write)
    turn.service.stop_room('room', cancel_id='deleted')
    assert retired == [turn.sid]
    with raw_holder(turn.authority.persistence)._read_ctx() as conn:
        assert conn.execute("SELECT key FROM state_meta WHERE key GLOB 'gateway.hosted.closed_attempt.v1:*'").fetchall() == []


def test_exact_lookup_decodes_only_the_selected_task_in_long_history(hosted_owner, monkeypatch):
    import json
    from gateway import owner_identity
    from persistence.gateway.hosted_rooms import HostedTaskIdentity
    turn = _turn(hosted_owner, monkeypatch)
    db, rpc = turn.authority.persistence, turn.member
    row = turn.admission()
    parsed = []
    validate = HostedTaskIdentity.__post_init__

    def count_identity(identity):
        parsed.append(identity.task_id)
        validate(identity)

    def seed(conn):
        conn.executemany('''INSERT INTO session_admissions
            (admission_id, request_id, principal_id, target_session_id, lineage_json,
             payload_json, payload_digest, intent, status, outcome)
            VALUES (?, ?, ?, ?, '[]', ?, 'digest', 'queue', 'terminal', 'interrupted')''',
            [(f'old-{n}', 'hosted:' + json.dumps([{'room_id': 'room', 'task_id': f'old-{n}',
              'thread_id': 'thread', 'turn_id': str(n)}, 1]), rpc.principal.subject, turn.sid,
              json.dumps({'text': 'x' * 1024})) for n in range(10000)])
        # Older JSON spacing and key order still describe the same attempt.
        request = json.loads(row['request_id'][7:])
        request[0] = dict(reversed(list(request[0].items())))
        conn.execute('UPDATE session_admissions SET request_id=?, principal_id=? WHERE admission_id=?',
            ('hosted:' + json.dumps(request, indent=2), 'basic:old-login',
             row['admission_id']))

    raw_holder(db)._execute_write(seed)
    monkeypatch.setattr(owner_identity, 'host_owner_subject', lambda: rpc.principal.subject)
    monkeypatch.setattr(HostedTaskIdentity, '__post_init__', count_identity)
    assert rpc.lookup_attempt(profile='default', source='bot_room', session_id=turn.sid,
        expected_task_id=turn.identity.task_id, execution_generation=1) == {'status': 'running'}
    assert len(parsed) == 1, 'exact Stop lookup decoded unrelated admissions under the writer lock'
