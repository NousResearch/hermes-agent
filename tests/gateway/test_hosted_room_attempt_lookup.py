"""Public Retry and retirement behavior for an exact deferred attempt."""
import pytest

from gateway import hosted_room_driver as tasks
from gateway.session_hosted_service import CanonicalHostedRoomService
from tests.gateway.test_hosted_room_stop_terminal import _turn, _finish
from tests.gateway.test_hosted_room_stop_attempt import _defer


@pytest.mark.parametrize('state', ['completed', 'interrupted'])
def test_retry_recovers_the_selected_attempt_after_restart(hosted_owner, monkeypatch, state):
    turn = _turn(hosted_owner, monkeypatch)
    _defer(turn)
    _finish(turn, state, 'retained answer')
    homes = turn.service.profile_homes()
    tasks.release_lease(turn.service.db_path, turn.lease, clock=turn.service.runtime.clock)
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
    from hermes_state_mutation_retirement import retired_session, prune_retired_tombstones
    turn = _turn(hosted_owner, monkeypatch, admitted=False)
    turn.service.stop_room('room', cancel_id='unused')
    db, sid = turn.authority.db, turn.sid
    if retirement == 'delete':
        assert db.delete_session(sid, sessions_dir=tmp_path / 'sessions')
    elif retirement == 'empty_sweep':
        db._execute_write(lambda conn: conn.execute('UPDATE sessions SET title=NULL WHERE id=?', (sid,)))
        assert db.delete_session_if_empty(sid, sessions_dir=tmp_path / 'sessions')
    else:
        db.end_session(sid, 'done')
        assert db.prune_sessions(older_than_days=None, sessions_dir=tmp_path / 'sessions') == 1
    db._execute_write(lambda conn: prune_retired_tombstones(conn, older_than=-1))
    assert retired_session(db, sid)
    with db._read_ctx() as conn:
        assert conn.execute("SELECT key FROM state_meta WHERE key GLOB 'gateway.hosted.closed_attempt.v1:*'").fetchall() == []


def test_stop_racing_session_retirement_cannot_recreate_a_closed_key(hosted_owner, monkeypatch, tmp_path):
    from gateway import session_runtime_workers
    turn = _turn(hosted_owner, monkeypatch, admitted=False)
    original = session_runtime_workers.tracked_write

    async def retire_before_write(authority, work):
        assert authority.db.delete_session(turn.sid, sessions_dir=tmp_path / 'sessions')
        return await original(authority, work)

    monkeypatch.setattr(session_runtime_workers, 'tracked_write', retire_before_write)
    turn.service.stop_room('room', cancel_id='deleted')
    with turn.authority.db._read_ctx() as conn:
        assert conn.execute("SELECT key FROM state_meta WHERE key GLOB 'gateway.hosted.closed_attempt.v1:*'").fetchall() == []


def test_exact_lookup_decodes_only_the_selected_task_in_long_history(hosted_owner, monkeypatch):
    import json
    from gateway import owner_identity, session_hosted_rpc
    turn = _turn(hosted_owner, monkeypatch)
    db, rpc = turn.authority.db, turn.member
    row = turn.admission()
    parsed = []
    identity_type = session_hosted_rpc.TaskIdentity

    def identity(**fields):
        parsed.append(fields)
        return identity_type(**fields)

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

    db._execute_write(seed)
    monkeypatch.setattr(owner_identity, 'host_owner_subject', lambda: rpc.principal.subject)
    monkeypatch.setattr(session_hosted_rpc, 'TaskIdentity', identity)
    assert rpc.lookup_attempt(profile='default', source='bot_room', session_id=turn.sid,
        expected_task_id=turn.identity.task_id, execution_generation=1) == {'status': 'running'}
    assert len(parsed) == 1, 'exact Stop lookup decoded unrelated admissions under the writer lock'
