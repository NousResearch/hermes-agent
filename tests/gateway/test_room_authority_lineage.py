"""Target-authorized succession links homes without merging unrelated room origins."""
import asyncio
import time
from unittest.mock import MagicMock

import pytest
from aiohttp.test_utils import TestClient, TestServer

from gateway import hosted_rooms
from gateway.hosted_room_peer import decode_room_grant, issue_room_grant
from gateway.platforms.api_server_run_authority import room_authority, room_run_scope
from gateway.platforms.api_server_run_idempotency import RunIdempotencyStore
from tests.gateway.test_api_server_room_cancellation import _adapter, _headers, _invitation
from tests.gateway.test_room_cancellation_retirement_http import app


def successor_body(**changes):
    return {'room_id': 'room-1', 'home_install_id': 'successor', 'authority_gateway_id': 'successor',
            'authority_epoch': 2, 'member_id': 'member-1', 'previous_authority': {
                'home_install_id': 'home', 'authority_gateway_id': 'home', 'authority_epoch': 1}, **changes}


async def invite(cli, body, authorization='Bearer test-room-key'):
    return await cli.post('/v1/room-members/invitations', headers={'Authorization': authorization}, json=body)


@pytest.mark.asyncio
async def test_changed_home_compacts_with_explicit_lineage_and_survives_restart(tmp_path):
    path = tmp_path / 'runs.db'
    adapter = _adapter(path)
    try:
        async with TestClient(TestServer(app(adapter))) as cli:
            grant, body = await _invitation(cli)
            claims = decode_room_grant(adapter._room_grant_secret(), grant, permission='status')
            for index in range(64):
                body['hosted_room_dispatch']['task_id'] = f'task-{index}'
                headers = {**_headers(grant), 'Idempotency-Key': f'room:task-{index}:1'}
                assert (await cli.post('/v1/runs/stop', headers=headers, json=body)).status == 200
            store = adapter._run_idempotency_store
            assert store._conn.execute('SELECT COUNT(*) FROM run_idempotency').fetchone()[0] == 64
            # Signed peer authority cannot authorize a new lineage on the target's behalf.
            assert (await invite(cli, successor_body(), f'HermesRoom {grant}')).status == 401
            assert (await invite(cli, successor_body())).status == 201
            assert (await invite(cli, successor_body())).status == 201
            assert store._conn.execute('SELECT COUNT(*) FROM run_idempotency').fetchone()[0] == 0
            assert store._conn.execute('SELECT COUNT(*) FROM run_room_authorities').fetchone()[0] == 1
            store.close()
            store = adapter._run_idempotency_store = RunIdempotencyStore(str(path))
            for index in range(64):
                assert store.reserve(room_run_scope(claims), f'room:task-{index}:1', 'late', 'never-start',
                    {'status': 'queued'}, room_authority=room_authority(claims)) == ('authority_retired', None)
            # A second actual home move uses the retained successor, never the original host.
            third = successor_body(home_install_id='third', authority_gateway_id='third', authority_epoch=3,
                previous_authority={'home_install_id': 'successor', 'authority_gateway_id': 'successor', 'authority_epoch': 2})
            assert (await invite(cli, third)).status == 201
            assert (await invite(cli, successor_body(authority_epoch=4))).status == 400
            assert store._conn.execute('SELECT COUNT(*) FROM run_room_authorities').fetchone()[0] == 1
    finally:
        adapter._run_idempotency_store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('member_id', ['member-1', 'member-2'])
async def test_unrelated_origin_collision_does_not_compact_or_merge(tmp_path, member_id):
    adapter = _adapter(tmp_path / 'runs.db')
    try:
        async with TestClient(TestServer(app(adapter))) as cli:
            grant, body = await _invitation(cli)
            assert (await cli.post('/v1/runs/stop', headers=_headers(grant), json=body)).status == 200
            unrelated = successor_body(member_id=member_id)
            unrelated.pop('previous_authority')
            assert (await invite(cli, unrelated)).status == 400
            store = adapter._run_idempotency_store
            assert store._conn.execute('SELECT COUNT(*) FROM run_idempotency').fetchone()[0] == 1
            assert store._conn.execute('SELECT COUNT(*) FROM run_room_authorities').fetchone()[0] == 1
            # Refusing the collision also preserves the original grant's live reservation.
            assert (await cli.post('/v1/runs/stop', headers=_headers(grant), json=body)).status == 200
            assert store._conn.execute('SELECT COUNT(*) FROM run_idempotency').fetchone()[0] == 1
            assert store._conn.execute('SELECT COUNT(*) FROM run_room_authority_aliases').fetchone()[0] == 0
    finally:
        adapter._run_idempotency_store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('legacy', [False, True])
async def test_unlinked_collision_refused_after_restart_or_with_legacy_reservation(tmp_path, legacy):
    from gateway import hosted_rooms
    path = tmp_path / 'runs.db'
    adapter = _adapter(path)
    try:
        async with TestClient(TestServer(app(adapter))) as cli:
            await _invitation(cli)
            if legacy:
                adapter._run_idempotency_store._conn.execute('DELETE FROM run_room_namespaces')
                adapter._run_idempotency_store._conn.commit()
            else:
                with hosted_rooms._transaction(hosted_rooms.default_db_path(), immediate=True) as conn:
                    conn.execute('DELETE FROM hosted_room_peer_reservations')
            adapter._run_idempotency_store.close()
            adapter._run_idempotency_store = RunIdempotencyStore(str(path))
            unrelated = successor_body()
            unrelated.pop('previous_authority')
            assert (await invite(cli, unrelated)).status == 400
            assert (await invite(cli, successor_body())).status == 201
    finally:
        adapter._run_idempotency_store.close()


def test_explicit_binding_cannot_absorb_a_preexisting_unrelated_origin(tmp_path):
    store = RunIdempotencyStore(str(tmp_path / 'runs.db'))
    try:
        first, unrelated = ('first', 1, 'home'), ('unrelated', 2, 'other')
        store.observe_room_authority('first-scope', first)
        store.observe_room_authority('other-scope', unrelated)
        with pytest.raises(ValueError, match='another room origin'):
            store.observe_room_authority('new-scope', ('unrelated', 3, 'other'), first, 'home')
        assert store._conn.execute('SELECT COUNT(*) FROM run_room_authority_aliases').fetchone()[0] == 0
    finally:
        store.close()


@pytest.mark.asyncio
async def test_shared_room_epoch_fences_all_members_but_retirement_stays_per_member(tmp_path):
    adapter = _adapter(tmp_path / 'runs.db')
    try:
        async with TestClient(TestServer(app(adapter))) as cli:
            first, first_body = await _invitation(cli)
            second_body = successor_body(home_install_id='home', authority_gateway_id='home', authority_epoch=1,
                                         member_id='member-2')
            second_body.pop('previous_authority')
            response = await invite(cli, second_body)
            assert response.status == 201
            second = (await response.json())['grant']
            claims = decode_room_grant(adapter._room_grant_secret(), second, permission='status')
            body = {**first_body, 'hosted_room_dispatch': {**first_body['hosted_room_dispatch'], 'member_id': 'member-2'}}
            assert (await cli.post('/v1/runs/stop', headers=_headers(second), json=body)).status == 200
            assert (await cli.post('/v1/room-members/grants/revoke', headers=_headers(first),
                                   json={'retire_authority': True})).status == 200
            store = adapter._run_idempotency_store
            assert store.accepts_room_authority(room_authority(claims))
            assert (await cli.post('/v1/runs/stop', headers=_headers(second), json=body)).status == 200
            # A different member advances the room. Member 2 has no new invitation,
            # but its captured writer and cancellation state share that epoch fence.
            assert (await invite(cli, successor_body())).status == 201
            assert store._conn.execute('SELECT COUNT(*) FROM run_idempotency').fetchone()[0] == 0
            assert store.reserve(room_run_scope(claims), 'room:task-1:1', 'late', 'late', {'status': 'queued'},
                                 room_authority=room_authority(claims)) == ('authority_retired', None)
            # A member introduced after takeover starts its own transcript at its
            # first home; future moves can carry that independently of older members.
            newcomer = successor_body(member_id='new-member')
            newcomer.pop('previous_authority')
            response = await invite(cli, newcomer)
            assert response.status == 201
            fresh = decode_room_grant(adapter._room_grant_secret(), (await response.json())['grant'], permission='status')
            assert store.room_origin_home(fresh) == 'successor'
            third = successor_body(member_id='new-member', home_install_id='third', authority_gateway_id='third',
                authority_epoch=3, previous_authority={
                    'home_install_id': 'successor', 'authority_gateway_id': 'successor', 'authority_epoch': 2})
            response = await invite(cli, third)
            assert response.status == 201
            continued = decode_room_grant(adapter._room_grant_secret(), (await response.json())['grant'], permission='status')
            assert store.room_origin_home(continued) == 'successor'
    finally:
        adapter._run_idempotency_store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('previous', [
    {'home_install_id': 'forged', 'authority_gateway_id': 'home', 'authority_epoch': 1},
    {'home_install_id': 'home', 'authority_gateway_id': 'forged', 'authority_epoch': 1},
    {'home_install_id': 'home', 'authority_gateway_id': 'home', 'authority_epoch': 2}])
async def test_invalid_predecessor_does_not_advance_authority(tmp_path, previous):
    adapter = _adapter(tmp_path / 'runs.db')
    try:
        async with TestClient(TestServer(app(adapter))) as cli:
            grant, body = await _invitation(cli)
            assert (await invite(cli, successor_body(previous_authority=previous))).status == 400
            response = await cli.post('/v1/runs/stop', headers=_headers(grant), json=body)
            assert response.status == 200 and (await response.json())['admission_cancelled']
    finally:
        adapter._run_idempotency_store.close()


def test_legacy_watermark_migrates_without_guessing_opaque_receipt_identity(tmp_path):
    import sqlite3
    path = tmp_path / 'legacy.db'
    with sqlite3.connect(path) as conn:
        conn.execute('''CREATE TABLE run_room_authorities (authority_key TEXT PRIMARY KEY,
            authority_epoch INTEGER NOT NULL, gateway_id TEXT NOT NULL, retired_through INTEGER NOT NULL DEFAULT 0)''')
        conn.execute("INSERT INTO run_room_authorities VALUES('old-home-key',1,'home',0)")
    store = RunIdempotencyStore(str(path))
    try:
        old = ('old-home-key', 1, 'home')
        for scope in ('known-old', 'opaque-old'):
            store.reserve(scope, 'cancelled', '', scope, {'status': 'cancelled', 'admission_cancelled': True},
                          cancel_if_missing=True)
        assert store.observe_room_authority('known-old', old)
        assert store.observe_room_authority('new-scope', ('new-home-key', 2, 'successor'), old, 'home')
        assert store._conn.execute('SELECT scope,room_authority_key FROM run_idempotency').fetchall() == [('opaque-old', None)]
        assert store.reserve('known-old', 'cancelled', '', 'late', {'status': 'queued'},
                             room_authority=old) == ('authority_retired', None)
        assert store.lookup('opaque-old', 'cancelled', '')[0] == 'reused'
    finally:
        store.close()


def _member_claims(**changes):
    claims = {
        'room_id': 'room-1', 'home_install_id': 'home', 'authority_gateway_id': 'home',
        'authority_epoch': 1, 'member_id': 'member-1', 'target_install_id': 'install-1',
        'target_profile': 'default'}
    claims.update(changes)
    return claims


def test_successor_scope_replays_and_stops_the_original_generation(tmp_path):
    """A new epoch is not a new Idempotency-Key. The live row stays the only run."""
    store = RunIdempotencyStore(str(tmp_path / 'runs.db'))
    try:
        claims = _member_claims()
        scope, authority = room_run_scope(claims), room_authority(claims)
        assert store.observe_room_authority(scope, authority, claims=claims)
        created, record = store.reserve(
            scope, 'room:task-1:1', 'fp-1', 'run-live', {'status': 'running'}, room_authority=authority)
        assert created == 'created' and record['run_id'] == 'run-live'
        successor = _member_claims(authority_epoch=2)
        assert store.observe_room_authority(
            room_run_scope(successor), room_authority(successor), claims=successor)
        outcome, replayed = store.reserve(
            room_run_scope(successor), 'room:task-1:1', 'fp-2', 'run-other', {'status': 'running'},
            room_authority=room_authority(successor))
        assert outcome == 'reused' and replayed['run_id'] == 'run-live'
        assert store._conn.execute('SELECT COUNT(*) FROM run_idempotency').fetchone()[0] == 1
        stopped, stopped_record = store.reserve(
            room_run_scope(successor), 'room:task-1:1', 'fp-2', 'run-stop',
            {'status': 'cancelled', 'admission_cancelled': True}, cancel_if_missing=True,
            room_authority=room_authority(successor))
        assert stopped == 'reused' and stopped_record['run_id'] == 'run-live'
        assert store._conn.execute(
            'SELECT run_id, stop_requested FROM run_idempotency').fetchone() == ('run-live', 1)
    finally:
        store.close()


def test_moved_home_replays_the_predecessor_generation(tmp_path):
    store = RunIdempotencyStore(str(tmp_path / 'runs.db'))
    try:
        claims = _member_claims()
        scope, authority = room_run_scope(claims), room_authority(claims)
        assert store.observe_room_authority(scope, authority, claims=claims)
        assert store.reserve(
            scope, 'room:task-1:1', 'fp-1', 'run-live', {'status': 'running'},
            room_authority=authority)[0] == 'created'
        moved = _member_claims(home_install_id='successor', authority_gateway_id='successor', authority_epoch=2)
        assert store.observe_room_authority(
            room_run_scope(moved), room_authority(moved), authority, 'home', claims=moved)
        outcome, replayed = store.reserve(
            room_run_scope(moved), 'room:task-1:1', 'fp-moved', 'run-other', {'status': 'running'},
            room_authority=room_authority(moved))
        assert outcome == 'reused' and replayed['run_id'] == 'run-live'
        assert store._conn.execute('SELECT COUNT(*) FROM run_idempotency').fetchone()[0] == 1
        unrelated = _member_claims(home_install_id='other-home', member_id='member-b')
        assert not store.observe_room_authority(
            room_run_scope(unrelated), room_authority(unrelated), claims=unrelated)
    finally:
        store.close()


def test_compacted_absent_barrier_blocks_the_successor_key(tmp_path):
    store = RunIdempotencyStore(str(tmp_path / 'runs.db'))
    try:
        claims = _member_claims()
        scope, authority = room_run_scope(claims), room_authority(claims)
        assert store.observe_room_authority(scope, authority, claims=claims)
        assert store.reserve(
            scope, 'room:task-1:1', '', 'barrier-run',
            {'status': 'cancelled', 'admission_cancelled': True}, cancel_if_missing=True,
            room_authority=authority)[0] == 'created'
        successor = _member_claims(authority_epoch=2)
        assert store.observe_room_authority(
            room_run_scope(successor), room_authority(successor), claims=successor)
        assert store._conn.execute('SELECT COUNT(*) FROM run_idempotency').fetchone()[0] == 0
        assert store._conn.execute('SELECT run_id FROM run_room_fenced_attempts').fetchone()[0] == 'barrier-run'
        outcome, replayed = store.reserve(
            room_run_scope(successor), 'room:task-1:1', 'fp-new', 'run-new', {'status': 'queued'},
            room_authority=room_authority(successor))
        assert outcome == 'reused' and replayed['run_id'] == 'barrier-run'
        assert store.reserve(
            scope, 'room:task-1:1', '', 'late', {'status': 'queued'},
            room_authority=authority) == ('authority_retired', None)
    finally:
        store.close()


def _finishing_agent():
    agent = MagicMock()
    agent.run_conversation.return_value = {'final_response': 'once', 'messages': []}
    agent.session_prompt_tokens = agent.session_completion_tokens = agent.session_total_tokens = 0
    return agent


async def _await_runs(adapter):
    tasks = tuple(adapter._active_run_tasks.values())
    if tasks:
        await asyncio.wait_for(asyncio.gather(*tasks), 10)


@pytest.mark.asyncio
async def test_http_successor_replays_the_same_run(tmp_path, monkeypatch):
    adapter = _adapter(tmp_path / 'runs.db')
    create = MagicMock(return_value=_finishing_agent())
    monkeypatch.setattr(adapter, '_create_agent', create)
    try:
        async with TestClient(TestServer(app(adapter))) as cli:
            grant, body = await _invitation(cli)
            first = await cli.post('/v1/runs', headers=_headers(grant), json=body)
            assert first.status == 202, await first.text()
            run_id = (await first.json())['run_id']
            await _await_runs(adapter)
            invited = await invite(cli, {
                'room_id': 'room-1', 'home_install_id': 'home', 'authority_gateway_id': 'home',
                'authority_epoch': 2, 'member_id': 'member-1'})
            assert invited.status == 201, await invited.text()
            grant2 = (await invited.json())['grant']
            replay_body = {'input': body['input'], 'hosted_room_dispatch': {
                **body['hosted_room_dispatch'], 'authority_epoch': 2}}
            second = await cli.post('/v1/runs', headers={**_headers(grant2), 'Idempotency-Key': 'room:task-1:1'},
                                    json=replay_body)
            assert second.status == 202, await second.text()
            payload = await second.json()
            assert payload['run_id'] == run_id and payload['replayed'] is True
            assert create.call_count == 1
    finally:
        adapter._run_idempotency_store.close()


@pytest.mark.asyncio
async def test_current_reservation_cannot_run_a_rejected_home(tmp_path, monkeypatch):
    adapter = _adapter(tmp_path / 'runs.db')
    create = MagicMock(side_effect=AssertionError('executed'))
    monkeypatch.setattr(adapter, '_create_agent', create)
    try:
        async with TestClient(TestServer(app(adapter))) as cli:
            _grant, body = await _invitation(cli)
            refused = await invite(cli, {
                'room_id': 'room-1', 'home_install_id': 'other-home', 'authority_gateway_id': 'home',
                'authority_epoch': 1, 'member_id': 'member-b'})
            assert refused.status == 400
            dispatch = body['hosted_room_dispatch']
            claims = {
                'room_id': 'room-1', 'home_install_id': 'other-home', 'authority_gateway_id': 'home',
                'authority_epoch': 1, 'member_id': 'member-b', 'target_profile': 'default'}
            hosted_rooms.reserve_peer_room(
                hosted_rooms.default_db_path(), claims=claims, expires_at=time.time() + 3600)
            assert hosted_rooms.peer_room_grant_is_current(hosted_rooms.default_db_path(), claims=claims)
            grant = issue_room_grant(
                adapter._room_grant_secret(), grant_id='other-home', room_id='room-1',
                home_install_id='other-home', authority_gateway_id='home', authority_epoch=1,
                member_id='member-b', target_install_id=dispatch['target_install_id'],
                target_profile='default', execution_policy_digest=dispatch['execution_policy_digest'],
                permissions=('dispatch', 'status', 'stop'))
            other = {**dispatch, 'home_install_id': 'other-home', 'member_id': 'member-b', 'task_id': 'task-b'}
            response = await cli.post(
                '/v1/runs', headers={'Authorization': f'HermesRoom {grant}', 'Idempotency-Key': 'room:task-b:1'},
                json={'input': body['input'], 'hosted_room_dispatch': other})
            assert response.status == 409, await response.text()
            assert (await response.json())['error']['code'] == 'run_history_retired'
            assert not create.called
    finally:
        adapter._run_idempotency_store.close()


def test_unbound_invitation_does_not_reserve_peer_room(monkeypatch):
    import tui_gateway.methods_groups as groups
    monkeypatch.setattr(groups, '_bound_server', None)

    def reserved(*_args, **_kwargs):
        raise AssertionError('reserved')

    monkeypatch.setattr(hosted_rooms, 'reserve_peer_room', reserved)
    with pytest.raises(ValueError, match='gateway run store'):
        groups._record_peer_run_authority({'room_id': 'room-1'})
