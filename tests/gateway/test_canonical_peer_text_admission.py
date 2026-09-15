"""Real canonical HTTP-to-admission chain with inert post-admission execution."""
import asyncio
import hashlib
import json
import sqlite3

import pytest

from tests.gateway.test_canonical_peer_target_setup import target, invite, request  # noqa: F401


def dispatch(invitation, task='task-one', prompt='hello'):
    from gateway.hosted_room_peer import PROTOCOL_VERSION, HostedMemberDispatch
    catalog = invitation['catalog']
    return HostedMemberDispatch.from_mapping(dict(protocol_version=PROTOCOL_VERSION,
        room_id='room-one', home_install_id='home-install', authority_gateway_id='home-gateway',
        authority_epoch=1, member_id='member-one', target_install_id=catalog['installation_id'],
        target_profile='default', task_id=task, execution_generation=1, source_event_seq=1,
        cancellation_scope_id='cancel-one', prompt=prompt,
        prompt_digest=hashlib.sha256(prompt.encode()).hexdigest(),
        capability_digest=catalog['catalog_digest'], execution_policy_digest=catalog['execution_policy']['policy_digest'],
        trace_id='trace-one'))


def run_request(grant, value):
    return request({'hosted_room_dispatch': value.as_mapping()}, token=grant,
                   key=f'room:{value.task_id}:{value.execution_generation}')


@pytest.mark.asyncio
async def test_real_http_admission_holds_both_grant_fences_and_preserves_replay(target, monkeypatch):
    from gateway.platforms import api_server_runs
    invitation = await invite(target)
    value = dispatch(invitation)
    executed = []
    async def inert(adapter, launch, **kwargs):
        executed.append(launch.admission)
    monkeypatch.setattr(api_server_runs, '_execute_run', inert)
    checks = []
    def at_insert():
        for path in (target.home / 'shared-state.db', target.home / 'state.db'):
            with sqlite3.connect(path, timeout=0) as other:
                try:
                    other.execute('BEGIN IMMEDIATE')
                except sqlite3.OperationalError:
                    checks.append(True)
                else:
                    checks.append(False)
                    other.rollback()
        return 1
    target.db._conn.create_function('check_grant_fences', 0, at_insert)
    target.db._conn.execute('CREATE TRIGGER check_fences BEFORE INSERT ON session_admissions BEGIN SELECT check_grant_fences(); END')
    response = await target.adapter._handle_runs(run_request(invitation['grant'], value))
    assert response.status == 202, response.text
    await asyncio.sleep(0)
    assert checks == [True, True]
    assert len(executed) == 1 and executed[0] is not None
    authority, ref, row = executed[0]
    assert authority is target.authority
    assert row['payload']['text'] == value.prompt
    settings = row['payload']['api_turn_v1']['settings']
    assert settings['room_dispatch'] == value.as_mapping()
    assert settings['room_execution_policy'] == invitation['catalog']['execution_policy']
    assert len(row['payload']['api_turn_v1']['run_owner_scope']) == 64
    assert invitation['grant'] not in json.dumps(row)
    from gateway.session_api_turn import prepare_api_execution, api_execution, api_policy_scope
    from gateway.hosted_room_execution_policy import current_room_execution_policy
    from tools.approval_context import _get_approval_mode
    prepared = prepare_api_execution(authority, ref, row['payload'])
    context = api_execution.set(prepared)
    try:
        with api_policy_scope():
            assert current_room_execution_policy().as_mapping() == settings['room_execution_policy']
            assert _get_approval_mode() == 'manual'
    finally:
        api_execution.reset(context)
    assert current_room_execution_policy() is None
    session = target.db.get_session(ref.session_id)
    assert session['hidden'] == 1 and session['source'] == 'bot_room'
    replay = await target.adapter._handle_runs(run_request(invitation['grant'], value))
    assert replay.status == 202 and json.loads(replay.text)['run_id'] == json.loads(response.text)['run_id']
    assert json.loads(replay.text)['replayed'] is True
    conflict = await target.adapter._handle_runs(run_request(invitation['grant'], dispatch(invitation, prompt='changed')))
    assert conflict.status == 409
    continued = await target.adapter._handle_runs(run_request(invitation['grant'], dispatch(invitation, task='task-two')))
    assert continued.status == 202, continued.text
    await asyncio.sleep(0)
    assert len(executed) == 2 and executed[1][1] == ref
    assert target.db._conn.execute('SELECT count(*) FROM sessions').fetchone()[0] == 1
    from gateway.hosted_room_peer import decode_room_grant
    from gateway import hosted_rooms
    from hermes_state_runtime import admit_session_input
    claims = decode_room_grant(target.adapter._room_grant_secret(), invitation['grant'], permission='status')
    hosted_rooms.revoke_room_grant_scope(target.home / 'state.db', claims=claims, expires_at=claims['status_expires_at'])
    def denied(conn):
        raise PermissionError('new work must be denied')
    accepted = admit_session_input(target.db, epoch=authority.epoch, principal_id='api',
        session_id=ref.session_id, request_id=row['request_id'], payload=row['payload'], _authorize_write=denied)
    assert accepted['admission_id'] == row['admission_id']
    assert target.db._conn.execute('SELECT count(*) FROM session_admissions').fetchone()[0] == 2


@pytest.mark.asyncio
@pytest.mark.parametrize('store', ['shared-state.db', 'state.db', 'scope', 'supersede', 'epoch', 'policy', 'legacy'])
async def test_revocation_between_authentication_and_insert_denies_and_cleans_own_run(target, monkeypatch, store):
    from gateway import hosted_rooms
    from gateway.hosted_room_peer import decode_room_grant
    from gateway import session_api_turn
    from gateway.platforms import api_server_runs
    invitation = await invite(target)
    claims = decode_room_grant(target.adapter._room_grant_secret(), invitation['grant'], permission='status')
    original = session_api_turn.admit_api_turn
    def interleave(*args, **kwargs):
        if store in {'shared-state.db', 'state.db'}:
            hosted_rooms.revoke_room_grant_id(target.home / store, claims=claims, expires_at=claims['status_expires_at'])
        elif store == 'scope':
            hosted_rooms.revoke_room_grant_scope(target.home / 'state.db', claims=claims, expires_at=claims['status_expires_at'])
        elif store == 'supersede':
            hosted_rooms.reserve_peer_room(target.home / 'state.db', claims=claims | {'authority_epoch': 2}, expires_at=claims['status_expires_at'])
        elif store == 'epoch':
            from hermes_state_runtime import begin_runtime_epoch
            begin_runtime_epoch(target.db, instance_id='new-owner')
        elif store == 'policy':
            (target.home / 'config.yaml').write_text('agent:\n  max_turns: 8\napprovals:\n  mode: manual\n')
        else:
            target.db._execute_write(lambda conn: conn.execute(
                'INSERT INTO hosted_room_revoked_grant_ids(scope_key,grant_id,expires_at) VALUES(?,?,?)',
                (hosted_rooms._room_grant_scope_key(claims), claims['grant_id'], claims['status_expires_at'])))
        return original(*args, **kwargs)
    monkeypatch.setattr(session_api_turn, 'admit_api_turn', interleave)
    async def forbidden(*args, **kwargs):
        pytest.fail('refused work reached execution')
    monkeypatch.setattr(api_server_runs, '_execute_run', forbidden)
    response = await target.adapter._handle_runs(run_request(invitation['grant'], dispatch(invitation)))
    assert response.status == 409, response.text
    assert target.db._conn.execute('SELECT count(*) FROM session_admissions').fetchone()[0] == 0
    assert target.adapter._run_idempotency_store._conn.execute('SELECT count(*) FROM run_idempotency').fetchone()[0] == 0
    assert not target.adapter._active_run_tasks


@pytest.mark.asyncio
@pytest.mark.parametrize('unsupported', ['store', 'owner', 'approval', 'attachments'])
async def test_unsupported_refused_before_hidden_binding(target, unsupported):
    invitation = await invite(target)
    body = {'hosted_room_dispatch': dispatch(invitation).as_mapping()}
    if unsupported == 'store':
        target.adapter._run_idempotency_store._db_path = None
    elif unsupported == 'owner':
        target.runner.session_authorities._by_key.clear()
    elif unsupported == 'approval':
        (target.home / 'config.yaml').write_text('approvals:\n  mode: off\n')
    else:
        body['attachments'] = []
    response = await target.adapter._handle_runs(request(body, token=invitation['grant'], key='room:task-one:1'))
    assert response.status in (403, 409, 400), response.text
    assert target.db._conn.execute('SELECT count(*) FROM sessions').fetchone()[0] == 0
    assert target.db._conn.execute('SELECT count(*) FROM session_admissions').fetchone()[0] == 0


@pytest.mark.asyncio
async def test_false_real_catalog_is_not_an_ingress_authorization(target):
    from gateway.platforms.api_server_room_grants import _local_room_catalog
    from gateway import hosted_rooms
    invitation = await invite(target)
    target.adapter._run_idempotency_store._db_path = None
    _, catalog = _local_room_catalog(target.adapter, 'default', hosted_rooms.local_authority_gateway_id())
    assert catalog['text'] is False
    value = dispatch(invitation | {'catalog': catalog})
    response = await target.adapter._handle_runs(run_request(invitation['grant'], value))
    assert response.status in (403, 409), response.text
    assert target.db._conn.execute('SELECT count(*) FROM sessions').fetchone()[0] == 0


def test_cleanup_cannot_delete_a_replacement_reservation(target):
    store = target.adapter._run_idempotency_store
    store.reserve('scope', 'key', 'fingerprint', 'winner', {'status': 'queued'}, owner_pid=1, owner_started=1)
    store.forget('scope', 'key', run_id='loser')
    assert store.lookup('scope', 'key', 'fingerprint')[1]['run_id'] == 'winner'
    store.forget('scope', 'key', run_id='winner')
    assert store.lookup('scope', 'key', 'fingerprint')[0] == 'missing'


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', ['sqlite', 'owner_removed', 'runner_detached'])
async def test_canonical_admission_failure_never_selects_legacy_executor(target, monkeypatch, failure):
    from gateway.platforms import api_server_runs
    first = await invite(target)
    calls = []
    async def inert(adapter, launch, **kwargs):
        calls.append(launch.admission)
    monkeypatch.setattr(api_server_runs, '_execute_run', inert)
    if failure == 'sqlite':
        target.db._conn.execute("CREATE TRIGGER fail_insert BEFORE INSERT ON session_admissions BEGIN SELECT RAISE(ABORT, 'test full'); END")
    else:
        # Clear the owner after normalization, at a real HTTP pre-admission seam.
        original = target.adapter._conversation_history_for_session
        async def remove_owner(*args):
            result = await original(*args)
            target.runner.session_authority = None
            target.runner.session_authorities = None
            if failure == 'runner_detached':
                target.adapter.gateway_runner = None
            return result
        monkeypatch.setattr(target.adapter, '_conversation_history_for_session', remove_owner)
    response = None
    try:
        response = await target.adapter._handle_runs(run_request(first['grant'], dispatch(first)))
    except sqlite3.Error:
        pass
    await asyncio.sleep(0)
    assert response is not None and response.status in (409, 503)
    assert not calls
    assert not target.db._conn.execute('SELECT 1 FROM session_admissions').fetchone()
    assert not target.adapter._run_idempotency_store._conn.execute('SELECT 1 FROM run_idempotency').fetchone()


@pytest.mark.asyncio
async def test_bearer_rotation_replays_the_original_logical_run(target, monkeypatch):
    from gateway.hosted_room_peer import decode_room_grant, issue_room_grant
    from gateway.platforms import api_server_runs
    first = await invite(target)
    runs = []
    async def inert(adapter, launch, **kwargs):
        runs.append(launch.admission)
    monkeypatch.setattr(api_server_runs, '_execute_run', inert)
    value = dispatch(first)
    initial = await target.adapter._handle_runs(run_request(first['grant'], value))
    assert initial.status == 202, initial.text
    await asyncio.sleep(0)
    claims = decode_room_grant(target.adapter._room_grant_secret(), first['grant'], permission='status')
    fields = {key: claims[key] for key in ('grant_id', 'room_id', 'home_install_id', 'authority_gateway_id',
        'authority_epoch', 'member_id', 'target_install_id', 'target_profile', 'execution_policy_digest', 'permissions')}
    replacement = issue_room_grant(target.adapter._room_grant_secret(), **fields,
        issued_at=claims['issued_at'] + 1, ttl_seconds=599, status_expires_at=claims['status_expires_at'])
    revoked = await target.connection.dispatch(dict(id=1, method='groups.peer.revoke_exact', params={'grant': first['grant']}))
    assert 'result' in revoked
    replay = await target.adapter._handle_runs(run_request(replacement, value))
    assert replay.status == 202, replay.text
    assert json.loads(replay.text)['run_id'] == json.loads(initial.text)['run_id']
    assert json.loads(replay.text)['replayed'] is True
    await asyncio.sleep(0)
    assert len(runs) == 1
    assert target.adapter._run_idempotency_store._conn.execute('SELECT retention_until FROM run_idempotency').fetchone()[0] == claims['status_expires_at']
