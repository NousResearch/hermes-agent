"""Target-authorized succession links homes without merging unrelated room origins."""
import asyncio
import hashlib
import json
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


def test_successor_stamps_and_replays_an_unstamped_predecessor_row(tmp_path):
    """A room row written before room_authority_key existed is the same generation."""
    store = RunIdempotencyStore(str(tmp_path / 'runs.db'))
    try:
        claims = _member_claims()
        scope = room_run_scope(claims)
        assert store.reserve(
            scope, 'room:task-1:1', 'fp', 'run-live', {'status': 'running'})[0] == 'created'
        assert store._conn.execute(
            'SELECT room_authority_key FROM run_idempotency WHERE run_id=?', ('run-live',)).fetchone()[0] is None
        successor = _member_claims(authority_epoch=2)
        successor_scope, successor_authority = room_run_scope(successor), room_authority(successor)
        assert store.observe_room_authority(successor_scope, successor_authority, claims=successor)
        outcome, record = store.reserve(
            successor_scope, 'room:task-1:1', 'fp-2', 'run-again', {'status': 'queued'},
            room_authority=successor_authority)
        assert outcome == 'reused' and record['run_id'] == 'run-live'
        assert store.owns_run(successor_scope, 'run-live', room_authority=successor_authority)
        store.request_stop(successor_scope, 'run-live', room_authority=successor_authority)
        assert store.stop_requested('run-live') is True
        assert store._conn.execute('SELECT COUNT(*) FROM run_idempotency').fetchone()[0] == 1
    finally:
        store.close()


def test_unstamped_room_generation_is_not_pruned(tmp_path):
    store = RunIdempotencyStore(str(tmp_path / 'runs.db'))
    try:
        assert store.reserve(
            'old-scope', 'room:task-1:1', 'fp', 'run-done',
            {'status': 'completed', 'output': 'once'}, retention_until=1)[0] == 'created'
        # reserve() without a room authority is a client key. A row written before the
        # column existed has the migration default and must stay until a successor stamps it.
        store._conn.execute("UPDATE run_idempotency SET client_replay=0 WHERE run_id='run-done'")
        store._conn.commit()
        assert store.reserve(
            'api-key-scope', 'client-key', 'fp', 'other', {'status': 'queued'})[0] == 'created'
        row = store._conn.execute(
            "SELECT run_id, status_json FROM run_idempotency WHERE idempotency_key='room:task-1:1'"
        ).fetchone()
        assert row[0] == 'run-done'
        assert 'once' in row[1]
        assert store._conn.execute('SELECT COUNT(*) FROM run_room_fenced_attempts').fetchone()[0] == 0
    finally:
        store.close()


def test_fresh_room_receipt_survives_the_grant_horizon(tmp_path):
    """A short status TTL must not discard the only copy of a reply that just settled."""
    store = RunIdempotencyStore(str(tmp_path / 'runs.db'))
    try:
        claims = _member_claims()
        scope, authority = room_run_scope(claims), room_authority(claims)
        assert store.observe_room_authority(scope, authority, claims=claims)
        assert store.reserve(
            scope, 'room:task-1:1', 'fp', 'run-done', {'status': 'completed', 'output': 'the real answer'},
            retention_until=1, room_authority=authority)[0] == 'created'
        assert store.reserve(
            scope, 'room:task-2:1', 'fp', 'run-other', {'status': 'queued'},
            room_authority=authority)[0] == 'created'
        row = store._conn.execute(
            "SELECT status_json FROM run_idempotency WHERE run_id='run-done'").fetchone()
        assert row is not None and 'the real answer' in row[0]
        assert store._conn.execute(
            "SELECT COUNT(*) FROM run_room_fenced_attempts WHERE idempotency_key='room:task-1:1'"
        ).fetchone()[0] == 0
    finally:
        store.close()


def test_status_for_run_reads_the_fence_when_the_row_disappears(tmp_path):
    store = RunIdempotencyStore(str(tmp_path / 'runs.db'))
    try:
        claims = _member_claims()
        scope, authority = room_run_scope(claims), room_authority(claims)
        assert store.observe_room_authority(scope, authority, claims=claims)
        assert store.reserve(
            scope, 'room:task-1:1', 'fp', 'run-done', {'status': 'completed', 'output': 'once'},
            room_authority=authority)[0] == 'created'
        from gateway.platforms.api_server_run_authority import remember_fenced_attempt
        remember_fenced_attempt(
            store._conn, authority[0], 'room:task-1:1', 'run-done',
            '{"status":"completed","output":"once"}')
        store._conn.commit()
        raw = store._conn

        class _DeleteAfterScopeLookup:
            def execute(self, sql, params=()):
                result = raw.execute(sql, params)
                if 'room_authority_key=? AND run_id=?' in sql:
                    raw.execute('DELETE FROM run_idempotency WHERE run_id=?', ('run-done',))
                return result

            def __getattr__(self, name):
                return getattr(raw, name)

        store._conn = _DeleteAfterScopeLookup()
        status = store.status_for_run(room_run_scope(_member_claims(authority_epoch=2)), 'run-done',
                                      room_authority=room_authority(_member_claims(authority_epoch=2)))
        assert status is not None
        assert status['status']['status'] == 'completed'
        assert status['status'].get('body_discarded') is True
        assert 'output' not in status['status']
    finally:
        store.close()


def test_existing_fence_drops_reply_fields(tmp_path):
    store = RunIdempotencyStore(str(tmp_path / 'runs.db'))
    try:
        from gateway.platforms.api_server_run_authority import remember_fenced_attempt
        body = '{"error":"boom","output":"secret reply","pending_steer":"keep this private steer","status":"completed","usage":{"total_tokens":9}}'
        store._conn.execute(
            """INSERT INTO run_room_fenced_attempts(authority_key, idempotency_key, run_id, status_json)
               VALUES('auth','room:task-1:1','run-done',?)""", (body,))
        store._conn.commit()
        remember_fenced_attempt(store._conn, 'auth', 'room:task-1:1', 'run-done', body)
        store._conn.commit()
        stored = json.loads(store._conn.execute(
            'SELECT status_json FROM run_room_fenced_attempts').fetchone()[0])
        assert stored['status'] == 'completed'
        assert stored['body_discarded'] is True
        assert 'output' not in stored and 'error' not in stored and 'usage' not in stored
        assert 'pending_steer' not in stored
    finally:
        store.close()


def test_unreadable_owner_start_does_not_count_as_dead(monkeypatch):
    import os
    from gateway.platforms.api_server_runs import _owner_alive
    monkeypatch.setattr('gateway.status._pid_exists', lambda pid: True)
    monkeypatch.setattr('gateway.status.get_process_start_time', lambda pid: None)
    # Outside the 2s start-time tolerance, so a missing fingerprint is not a mismatch.
    assert _owner_alive(os.getpid(), 10_000_000) is True

    def _explode(_pid):
        raise OSError("start time unreadable")

    monkeypatch.setattr('gateway.status.get_process_start_time', _explode)
    assert _owner_alive(os.getpid(), 10_000_000) is True
    monkeypatch.setattr('gateway.status._pid_exists', lambda pid: False)
    assert _owner_alive(os.getpid(), 10_000_000) is False


def test_gateway_change_and_epoch_gap_replay_an_unstamped_row(tmp_path):
    """A pre-column row stays one generation across a takeover and a long epoch gap."""
    store = RunIdempotencyStore(str(tmp_path / 'runs.db'))
    try:
        original = _member_claims(authority_gateway_id='gateway-a')
        scope = room_run_scope(original)
        now = 1_700_000_000.0
        store._conn.execute(
            """INSERT INTO run_idempotency(
                scope, idempotency_key, fingerprint, run_id, status_json,
                owner_pid, owner_started, retention_until, created_at, updated_at)
               VALUES(?,?,?,?,?,?,?,?,?,?)""",
            (scope, 'room:task-1:1', 'fp', 'run-live', '{"status":"running"}', 0, 0, 0, now, now))
        store._conn.commit()
        takeover = _member_claims(authority_gateway_id='gateway-b', authority_epoch=2)
        assert store.observe_room_authority(
            room_run_scope(takeover), room_authority(takeover),
            room_authority(original), claims=takeover)
        outcome, record = store.reserve(
            room_run_scope(takeover), 'room:task-1:1', 'fp-2', 'run-again', {'status': 'queued'},
            room_authority=room_authority(takeover))
        assert outcome == 'reused' and record['run_id'] == 'run-live'
        store.request_stop(room_run_scope(takeover), 'run-live', room_authority=room_authority(takeover))
        assert store.stop_requested('run-live') is True
        assert store._conn.execute('SELECT COUNT(*) FROM run_idempotency').fetchone()[0] == 1
    finally:
        store.close()

    store = RunIdempotencyStore(str(tmp_path / 'gap.db'))
    try:
        original = _member_claims()
        scope = room_run_scope(original)
        now = 1_700_000_000.0
        store._conn.execute(
            """INSERT INTO run_idempotency(
                scope, idempotency_key, fingerprint, run_id, status_json,
                owner_pid, owner_started, retention_until, created_at, updated_at)
               VALUES(?,?,?,?,?,?,?,?,?,?)""",
            (scope, 'room:task-1:1', 'fp', 'run-live', '{"status":"running"}', 0, 0, 0, now, now))
        store._conn.commit()
        jumped = _member_claims(authority_epoch=10)
        assert store.observe_room_authority(
            room_run_scope(jumped), room_authority(jumped), claims=jumped)
        outcome, record = store.reserve(
            room_run_scope(jumped), 'room:task-1:1', 'fp-2', 'run-again', {'status': 'queued'},
            room_authority=room_authority(jumped))
        assert outcome == 'reused' and record['run_id'] == 'run-live'
        assert store._conn.execute('SELECT COUNT(*) FROM run_idempotency').fetchone()[0] == 1
    finally:
        store.close()


def test_public_room_shaped_key_is_pruned(tmp_path):
    store = RunIdempotencyStore(str(tmp_path / 'runs.db'))
    try:
        assert store.reserve(
            'api-scope', 'room:not-a-room:1', 'fp', 'run-public',
            {'status': 'completed', 'output': 'secret'}, retention_until=1)[0] == 'created'
        assert store.reserve(
            'api-scope', 'ordinary-key', 'fp', 'run-other', {'status': 'queued'})[0] == 'created'
        assert store._conn.execute(
            "SELECT run_id FROM run_idempotency WHERE idempotency_key='room:not-a-room:1'"
        ).fetchone() is None
        assert store._conn.execute(
            "SELECT run_id FROM run_idempotency WHERE run_id='run-other'").fetchone()[0] == 'run-other'
    finally:
        store.close()


def _forget_room_stamp(store):
    """Drop the authority stamp the way a pre-column row was stored. Keep the scope."""
    columns = {str(row[1]) for row in store._conn.execute('PRAGMA table_info(run_idempotency)')}
    assignment = 'room_authority_key=NULL, room_authority_epoch=NULL'
    if 'room_lineage_key' in columns:
        assignment += ', room_lineage_key=NULL'
    store._conn.execute(f'UPDATE run_idempotency SET {assignment}')
    store._conn.commit()


def test_predecessor_epochs_do_not_allocate_the_full_span():
    from gateway.platforms.api_server_run_authority import _predecessor_epochs
    started = time.perf_counter()
    epochs = _predecessor_epochs(8_000_000)
    assert time.perf_counter() - started < 0.05
    assert len(epochs) <= 64
    assert 1 in epochs and 7_999_999 in epochs
    assert 4_000_000 not in epochs
    # The middle of a jump past the cap stays outside the hash walk. Epoch 1 does not.
    assert 40 not in _predecessor_epochs(100)
    assert 1 in _predecessor_epochs(100) and 99 in _predecessor_epochs(100)
    assert set(_predecessor_epochs(10)) == set(range(1, 10))


def test_lineage_key_replays_across_gateway_and_epoch_without_previous(tmp_path):
    """The lineage key finds the row when the epoch walk does not, on another gateway."""
    store = RunIdempotencyStore(str(tmp_path / 'runs.db'))
    try:
        original = _member_claims(authority_gateway_id='gateway-a', authority_epoch=40)
        scope, authority = room_run_scope(original), room_authority(original)
        assert store.observe_room_authority(scope, authority, claims=original)
        assert store.reserve(
            scope, 'room:task-1:1', 'fp', 'run-live', {'status': 'running'},
            room_authority=authority)[0] == 'created'
        store._conn.execute(
            "UPDATE run_idempotency SET room_authority_key=NULL, room_authority_epoch=NULL "
            "WHERE run_id='run-live'")
        store._conn.commit()
        jumped = _member_claims(authority_gateway_id='gateway-b', authority_epoch=100)
        started = time.perf_counter()
        assert store.observe_room_authority(
            room_run_scope(jumped), room_authority(jumped), claims=jumped)
        outcome, record = store.reserve(
            room_run_scope(jumped), 'room:task-1:1', 'fp-2', 'run-again', {'status': 'queued'},
            room_authority=room_authority(jumped))
        assert time.perf_counter() - started < 0.5
        assert outcome == 'reused' and record['run_id'] == 'run-live'
        other = _member_claims(
            member_id='member-2', authority_gateway_id='gateway-b', authority_epoch=100)
        assert store.observe_room_authority(
            room_run_scope(other), room_authority(other), claims=other)
        other_outcome, other_record = store.reserve(
            room_run_scope(other), 'room:task-1:1', 'fp', 'run-other', {'status': 'queued'},
            room_authority=room_authority(other))
        assert other_outcome == 'created' and other_record['run_id'] == 'run-other'
        assert store.stop_requested('run-live') is False
        store.request_stop(
            room_run_scope(other), 'run-other', room_authority=room_authority(other))
        assert store.stop_requested('run-live') is False
        assert store._conn.execute('SELECT COUNT(*) FROM run_idempotency').fetchone()[0] == 2
    finally:
        store.close()


def test_remembered_gateway_replays_a_null_row_across_later_hops(tmp_path):
    """Naming only the immediate predecessor still finds the gateway that admitted the row."""
    store = RunIdempotencyStore(str(tmp_path / 'runs.db'))
    try:
        original = _member_claims(authority_gateway_id='gateway-a')
        scope, authority = room_run_scope(original), room_authority(original)
        assert store.observe_room_authority(scope, authority, claims=original)
        assert store.reserve(
            scope, 'room:task-1:1', 'fp', 'run-live', {'status': 'running'},
            room_authority=authority)[0] == 'created'
        _forget_room_stamp(store)
        second = _member_claims(authority_gateway_id='gateway-b', authority_epoch=2)
        assert store.observe_room_authority(
            room_run_scope(second), room_authority(second), claims=second)
        _forget_room_stamp(store)
        third = _member_claims(authority_gateway_id='gateway-c', authority_epoch=3)
        assert store.observe_room_authority(
            room_run_scope(third), room_authority(third), room_authority(second), claims=third)
        outcome, record = store.reserve(
            room_run_scope(third), 'room:task-1:1', 'fp-3', 'run-again', {'status': 'queued'},
            room_authority=room_authority(third))
        assert outcome == 'reused' and record['run_id'] == 'run-live'
        store.request_stop(
            room_run_scope(third), 'run-live', room_authority=room_authority(third))
        assert store.stop_requested('run-live') is True
        assert store._conn.execute('SELECT COUNT(*) FROM run_idempotency').fetchone()[0] == 1
        foreign = _member_claims(home_install_id='other-home', authority_gateway_id='gateway-c',
                                 authority_epoch=3, member_id='member-1')
        with pytest.raises(ValueError, match='previous room authority is not current'):
            store.observe_room_authority(
                room_run_scope(foreign), room_authority(foreign), room_authority(second), claims=foreign)
        assert store.stop_requested('run-live') is True
        assert store._conn.execute(
            "SELECT room_authority_key FROM run_idempotency WHERE run_id='run-live'"
        ).fetchone()[0] == room_authority(original)[0]
    finally:
        store.close()


def _dispatch_for(body, catalog, *, gateway, epoch, member, task_id='task-1'):
    prompt = body['input']
    return {
        'protocol_version': 2, 'room_id': 'room-1', 'home_install_id': 'home',
        'authority_gateway_id': gateway, 'authority_epoch': epoch, 'member_id': member,
        'target_install_id': catalog['installation_id'], 'target_profile': 'default',
        'task_id': task_id, 'execution_generation': 1, 'source_event_seq': 1,
        'cancellation_scope_id': 'cancel-1', 'prompt': prompt,
        'prompt_digest': hashlib.sha256(prompt.encode()).hexdigest(),
        'capability_digest': catalog['catalog_digest'],
        'execution_policy_digest': catalog['execution_policy']['policy_digest'],
        'trace_id': 'trace-1'}


@pytest.mark.asyncio
async def test_http_takeover_without_previous_replays_an_unstamped_row(tmp_path, monkeypatch):
    """POST /v1/runs never sends previous_authority. A later gateway must still replay."""
    adapter = _adapter(tmp_path / 'runs.db')
    create = MagicMock(return_value=_finishing_agent())
    monkeypatch.setattr(adapter, '_create_agent', create)
    try:
        application = app(adapter)
        application.router.add_post('/v1/runs/{run_id}/stop', adapter._handle_stop_run)
        async with TestClient(TestServer(application)) as cli:
            invited = await invite(cli, {
                'room_id': 'room-1', 'home_install_id': 'home', 'authority_gateway_id': 'gateway-a',
                'authority_epoch': 1, 'member_id': 'member-1'})
            assert invited.status == 201, await invited.text()
            invitation = await invited.json()
            grant, catalog = invitation['grant'], invitation['catalog']
            prompt = 'Run this generation once.'
            body = {'input': prompt, 'hosted_room_dispatch': _dispatch_for(
                {'input': prompt}, catalog, gateway='gateway-a', epoch=1, member='member-1')}
            first = await cli.post('/v1/runs', headers=_headers(grant), json=body)
            assert first.status == 202, await first.text()
            run_id = (await first.json())['run_id']
            await _await_runs(adapter)
            assert create.call_count == 1

            async def takeover(gateway, epoch):
                _forget_room_stamp(adapter._run_idempotency_store)
                next_invite = await invite(cli, {
                    'room_id': 'room-1', 'home_install_id': 'home', 'authority_gateway_id': gateway,
                    'authority_epoch': epoch, 'member_id': 'member-1'})
                assert next_invite.status == 201, await next_invite.text()
                next_grant = (await next_invite.json())['grant']
                replay_body = {'input': prompt, 'hosted_room_dispatch': _dispatch_for(
                    {'input': prompt}, catalog, gateway=gateway, epoch=epoch, member='member-1')}
                replay = await cli.post(
                    '/v1/runs', headers={**_headers(next_grant), 'Idempotency-Key': 'room:task-1:1'},
                    json=replay_body)
                payload = await replay.json()
                assert replay.status == 202, payload
                assert payload['run_id'] == run_id and payload['replayed'] is True
                assert create.call_count == 1
                return next_grant

            await takeover('gateway-b', 2)
            grant_c = await takeover('gateway-c', 3)
            status = await cli.get(f'/v1/runs/{run_id}', headers=_headers(grant_c))
            assert status.status == 200, await status.text()
            assert (await status.json())['run_id'] == run_id

            other_claims = {
                'room_id': 'room-1', 'home_install_id': 'home', 'authority_gateway_id': 'gateway-c',
                'authority_epoch': 3, 'member_id': 'member-2', 'target_profile': 'default',
                'target_install_id': catalog['installation_id']}
            hosted_rooms.reserve_peer_room(
                hosted_rooms.default_db_path(), claims=other_claims, expires_at=time.time() + 3600)
            other_grant = issue_room_grant(
                adapter._room_grant_secret(), grant_id='member-2', room_id='room-1',
                home_install_id='home', authority_gateway_id='gateway-c', authority_epoch=3,
                member_id='member-2', target_install_id=catalog['installation_id'],
                target_profile='default',
                execution_policy_digest=catalog['execution_policy']['policy_digest'],
                permissions=('dispatch', 'status', 'stop'))
            other_body = {'input': prompt, 'hosted_room_dispatch': _dispatch_for(
                {'input': prompt}, catalog, gateway='gateway-c', epoch=3, member='member-2')}
            # Admission stop does not open a room session, so it cannot collide with the
            # live generation's title. It must still not adopt that generation.
            other = await cli.post('/v1/runs/stop', headers={
                'Authorization': f'HermesRoom {other_grant}', 'Idempotency-Key': 'room:task-1:1'},
                json=other_body)
            other_payload = await other.json()
            assert other.status == 200, other_payload
            assert other_payload['run_id'] != run_id
            assert create.call_count == 1
            assert adapter._run_idempotency_store.stop_requested(run_id) is False
            denied = await cli.get(f'/v1/runs/{run_id}', headers={
                'Authorization': f'HermesRoom {other_grant}'})
            assert denied.status == 404
            denied_stop = await cli.post(f'/v1/runs/{run_id}/stop', headers={
                'Authorization': f'HermesRoom {other_grant}'}, json={})
            assert denied_stop.status == 404
            assert adapter._run_idempotency_store.stop_requested(run_id) is False

            foreign_home = {
                'room_id': 'room-1', 'home_install_id': 'other-home',
                'authority_gateway_id': 'gateway-c', 'authority_epoch': 3,
                'member_id': 'member-9', 'target_profile': 'default'}
            hosted_rooms.reserve_peer_room(
                hosted_rooms.default_db_path(), claims=foreign_home, expires_at=time.time() + 3600)
            foreign_grant = issue_room_grant(
                adapter._room_grant_secret(), grant_id='other-home', room_id='room-1',
                home_install_id='other-home', authority_gateway_id='gateway-c', authority_epoch=3,
                member_id='member-9', target_install_id=catalog['installation_id'],
                target_profile='default',
                execution_policy_digest=catalog['execution_policy']['policy_digest'],
                permissions=('status', 'stop'))
            foreign_get = await cli.get(f'/v1/runs/{run_id}', headers={
                'Authorization': f'HermesRoom {foreign_grant}'})
            assert foreign_get.status == 404, await foreign_get.text()
            foreign_stop = await cli.post(f'/v1/runs/{run_id}/stop', headers={
                'Authorization': f'HermesRoom {foreign_grant}'}, json={})
            assert foreign_stop.status == 404, await foreign_stop.text()
            assert create.call_count == 1
            assert adapter._run_idempotency_store.stop_requested(run_id) is False

            stopped = await cli.post(
                f'/v1/runs/{run_id}/stop', headers=_headers(grant_c), json={})
            assert stopped.status == 200, await stopped.text()
            assert adapter._run_idempotency_store.stop_requested(run_id) is True
    finally:
        adapter._run_idempotency_store.close()


@pytest.mark.asyncio
async def test_http_legacy_row_replays_after_gateway_was_already_overwritten(tmp_path, monkeypatch):
    """Main already replaced the reservation. POST still finds the pre-column run and stop reaches it."""
    adapter = _adapter(tmp_path / 'runs.db')
    create = MagicMock(return_value=_finishing_agent())
    monkeypatch.setattr(adapter, '_create_agent', create)
    try:
        application = app(adapter)
        application.router.add_post('/v1/runs/{run_id}/stop', adapter._handle_stop_run)
        async with TestClient(TestServer(application)) as cli:
            invited = await invite(cli, {
                'room_id': 'room-1', 'home_install_id': 'home', 'authority_gateway_id': 'gateway-a',
                'authority_epoch': 1, 'member_id': 'member-1'})
            assert invited.status == 201, await invited.text()
            invitation = await invited.json()
            grant, catalog = invitation['grant'], invitation['catalog']
            prompt = 'Run this generation once.'
            body = {'input': prompt, 'hosted_room_dispatch': _dispatch_for(
                {'input': prompt}, catalog, gateway='gateway-a', epoch=1, member='member-1')}
            first = await cli.post(
                '/v1/runs', headers={**_headers(grant), 'Idempotency-Key': 'room:task-1:1'}, json=body)
            assert first.status == 202, await first.text()
            run_id = (await first.json())['run_id']
            await _await_runs(adapter)
            assert create.call_count == 1
            store = adapter._run_idempotency_store
            stored = json.loads(store._conn.execute(
                'SELECT status_json FROM run_idempotency WHERE run_id=?', (run_id,)).fetchone()[0])
            assert str(stored.get('session_id') or '').startswith('room_')
            # The state main leaves: the reservation names only the new gateway, and this
            # build has not yet recorded a watermark or a seen-gateway id.
            _forget_room_stamp(store)
            store._conn.execute('DELETE FROM run_room_authorities')
            store._conn.execute('DELETE FROM run_room_seen_gateways')
            store._conn.commit()
            moved = {
                'room_id': 'room-1', 'home_install_id': 'home', 'authority_gateway_id': 'gateway-b',
                'authority_epoch': 2, 'member_id': 'member-1', 'target_profile': 'default',
                'target_install_id': catalog['installation_id']}
            hosted_rooms.reserve_peer_room(
                hosted_rooms.default_db_path(), claims=moved, expires_at=time.time() + 3600)
            grant_b = issue_room_grant(
                adapter._room_grant_secret(), grant_id='gateway-b', room_id='room-1',
                home_install_id='home', authority_gateway_id='gateway-b', authority_epoch=2,
                member_id='member-1', target_install_id=catalog['installation_id'],
                target_profile='default',
                execution_policy_digest=catalog['execution_policy']['policy_digest'],
                permissions=('dispatch', 'status', 'stop'))
            replay_body = {'input': prompt, 'hosted_room_dispatch': _dispatch_for(
                {'input': prompt}, catalog, gateway='gateway-b', epoch=2, member='member-1')}
            replay = await cli.post(
                '/v1/runs', headers={**_headers(grant_b), 'Idempotency-Key': 'room:task-1:1'},
                json=replay_body)
            payload = await replay.json()
            assert replay.status == 202, payload
            assert payload['run_id'] == run_id and payload['replayed'] is True
            assert create.call_count == 1
            other_claims = {
                'room_id': 'room-1', 'home_install_id': 'home', 'authority_gateway_id': 'gateway-b',
                'authority_epoch': 2, 'member_id': 'member-2', 'target_profile': 'default',
                'target_install_id': catalog['installation_id']}
            hosted_rooms.reserve_peer_room(
                hosted_rooms.default_db_path(), claims=other_claims, expires_at=time.time() + 3600)
            other_grant = issue_room_grant(
                adapter._room_grant_secret(), grant_id='member-2', room_id='room-1',
                home_install_id='home', authority_gateway_id='gateway-b', authority_epoch=2,
                member_id='member-2', target_install_id=catalog['installation_id'],
                target_profile='default',
                execution_policy_digest=catalog['execution_policy']['policy_digest'],
                permissions=('dispatch', 'status', 'stop'))
            other = await cli.post('/v1/runs/stop', headers={
                'Authorization': f'HermesRoom {other_grant}', 'Idempotency-Key': 'room:task-1:1'},
                json=replay_body | {'hosted_room_dispatch': _dispatch_for(
                    {'input': prompt}, catalog, gateway='gateway-b', epoch=2, member='member-2')})
            other_payload = await other.json()
            assert other.status == 200, other_payload
            assert other_payload['run_id'] != run_id
            assert store.stop_requested(run_id) is False
            stopped = await cli.post(f'/v1/runs/{run_id}/stop', headers=_headers(grant_b), json={})
            assert stopped.status == 200, await stopped.text()
            assert store.stop_requested(run_id) is True
    finally:
        adapter._run_idempotency_store.close()


@pytest.mark.asyncio
async def test_failed_observe_keeps_the_admitting_gateway_for_the_retry(tmp_path, monkeypatch):
    """The rooms commit must not be the only copy of the gateway observe still needs."""
    adapter = _adapter(tmp_path / 'runs.db')
    create = MagicMock(return_value=_finishing_agent())
    monkeypatch.setattr(adapter, '_create_agent', create)
    try:
        async with TestClient(TestServer(app(adapter))) as cli:
            other = await invite(cli, {
                'room_id': 'room-1', 'home_install_id': 'home', 'authority_gateway_id': 'gateway-a',
                'authority_epoch': 1, 'member_id': 'member-2'})
            assert other.status == 201, await other.text()
            catalog = (await other.json())['catalog']
            claims = {
                'room_id': 'room-1', 'home_install_id': 'home', 'authority_gateway_id': 'gateway-a',
                'authority_epoch': 1, 'member_id': 'member-1', 'target_profile': 'default',
                'target_install_id': catalog['installation_id']}
            hosted_rooms.reserve_peer_room(
                hosted_rooms.default_db_path(), claims=claims, expires_at=time.time() + 3600)
            scope = room_run_scope(claims)
            store = adapter._run_idempotency_store
            now = time.time()
            store._conn.execute(
                """INSERT INTO run_idempotency(
                       scope, idempotency_key, fingerprint, run_id, status_json,
                       owner_pid, owner_started, retention_until, created_at, updated_at, client_replay)
                   VALUES(?,?,?,?,?,?,?,?,?,?,0)""",
                (scope, 'room:task-p1:1', 'fp', 'run-live', json.dumps({'status': 'running'}),
                 0, 0, now + 3600, now, now))
            store._conn.commit()
            import gateway.platforms.api_server_run_authority as auth
            original_observe = auth.observe

            def killed(*_args, **_kwargs):
                raise RuntimeError('killed after reservation commit')

            monkeypatch.setattr(auth, 'observe', killed)
            crashed = await invite(cli, {
                'room_id': 'room-1', 'home_install_id': 'home', 'authority_gateway_id': 'gateway-b',
                'authority_epoch': 2, 'member_id': 'member-1'})
            assert crashed.status == 400, await crashed.text()
            seen = [row[0] for row in store._conn.execute(
                'SELECT gateway_id FROM run_room_seen_gateways')]
            assert 'gateway-a' in seen
            monkeypatch.setattr(auth, 'observe', original_observe)
            retried = await invite(cli, {
                'room_id': 'room-1', 'home_install_id': 'home', 'authority_gateway_id': 'gateway-b',
                'authority_epoch': 2, 'member_id': 'member-1'})
            assert retried.status == 201, await retried.text()
            grant_b = (await retried.json())['grant']
            prompt = 'Run this generation once.'
            replay_body = {'input': prompt, 'hosted_room_dispatch': _dispatch_for(
                {'input': prompt}, catalog, gateway='gateway-b', epoch=2, member='member-1',
                task_id='task-p1')}
            replay = await cli.post(
                '/v1/runs', headers={**_headers(grant_b), 'Idempotency-Key': 'room:task-p1:1'},
                json=replay_body)
            payload = await replay.json()
            assert replay.status == 202, payload
            assert payload['run_id'] == 'run-live' and payload['replayed'] is True
            assert create.call_count == 0
            stopped = await cli.post('/v1/runs/stop', headers={
                **_headers(grant_b), 'Idempotency-Key': 'room:task-p1:1'}, json=replay_body)
            assert stopped.status == 200, await stopped.text()
            assert store.stop_requested('run-live') is True
    finally:
        adapter._run_idempotency_store.close()


def _grant_for(adapter, catalog, *, gateway, epoch, member):
    return issue_room_grant(
        adapter._room_grant_secret(), grant_id=f'{gateway}-{epoch}-{member}', room_id='room-1',
        home_install_id='home', authority_gateway_id=gateway, authority_epoch=epoch,
        member_id=member, target_install_id=catalog['installation_id'], target_profile='default',
        execution_policy_digest=catalog['execution_policy']['policy_digest'],
        permissions=('dispatch', 'status', 'stop'))


def _generation_body(prompt, catalog, *, gateway, epoch, member, generation, task_id='task-1'):
    dispatch = _dispatch_for(
        {'input': prompt}, catalog, gateway=gateway, epoch=epoch, member=member, task_id=task_id)
    dispatch['execution_generation'] = generation
    return {'input': prompt, 'hosted_room_dispatch': dispatch}


@pytest.mark.asyncio
async def test_http_compression_tip_replays_after_its_gateway_is_gone(tmp_path, monkeypatch):
    """A pre-column row stored as the compression tip is the live generation once main prunes the hash."""
    import threading
    adapter = _adapter(tmp_path / 'runs.db')
    release = threading.Event()
    create = MagicMock()

    def agent(*_args, **_kwargs):
        built = MagicMock()
        built.session_prompt_tokens = built.session_completion_tokens = built.session_total_tokens = 0
        if create.call_count == 1:
            built.run_conversation.return_value = {'final_response': 'once', 'messages': []}
        else:
            def block(**_ignored):
                release.wait(15)
                return {'interrupted': True}
            built.run_conversation.side_effect = block
        return built

    create.side_effect = agent
    monkeypatch.setattr(adapter, '_create_agent', create)
    prompt = 'Run this generation once.'
    try:
        application = app(adapter)
        application.router.add_post('/v1/runs/{run_id}/stop', adapter._handle_stop_run)
        async with TestClient(TestServer(application)) as cli:
            invited = await invite(cli, {
                'room_id': 'room-1', 'home_install_id': 'home', 'authority_gateway_id': 'gateway-a',
                'authority_epoch': 1, 'member_id': 'member-1'})
            assert invited.status == 201, await invited.text()
            catalog = (await invited.json())['catalog']
            grant_a = _grant_for(adapter, catalog, gateway='gateway-a', epoch=1, member='member-1')
            first = await cli.post(
                '/v1/runs', headers={**_headers(grant_a), 'Idempotency-Key': 'room:task-1:1'},
                json=_generation_body(prompt, catalog, gateway='gateway-a', epoch=1, member='member-1', generation=1))
            assert first.status == 202, await first.text()
            run1 = (await first.json())['run_id']
            store = adapter._run_idempotency_store
            deadline = time.time() + 10
            while time.time() < deadline:
                row = store._conn.execute(
                    'SELECT status_json FROM run_idempotency WHERE run_id=?', (run1,)).fetchone()
                if row and json.loads(row[0]).get('status') == 'completed':
                    break
                await asyncio.sleep(0.05)
            else:
                raise AssertionError('generation 1 did not complete')
            canonical = json.loads(row[0])['session_id']
            db = await adapter._ensure_session_db_async()
            db.end_session(canonical, 'compression')
            db.create_session('cont-tip', source='cli', parent_session_id=canonical)
            db.append_message('cont-tip', role='assistant', content='continued')
            assert db.resolve_resume_session_id(canonical) == 'cont-tip'
            # Main overwrites the reservation before this build has a watermark.
            _wipe_upgrade_state(store)
            hosted_rooms.reserve_peer_room(hosted_rooms.default_db_path(), claims={
                'room_id': 'room-1', 'home_install_id': 'home', 'authority_gateway_id': 'gateway-b',
                'authority_epoch': 2, 'member_id': 'member-1', 'target_profile': 'default',
                'target_install_id': catalog['installation_id']}, expires_at=time.time() + 3600)
            grant_b = _grant_for(adapter, catalog, gateway='gateway-b', epoch=2, member='member-1')
            second = await cli.post(
                '/v1/runs', headers={**_headers(grant_b), 'Idempotency-Key': 'room:task-1:2'},
                json=_generation_body(prompt, catalog, gateway='gateway-b', epoch=2, member='member-1', generation=2))
            assert second.status == 202, await second.text()
            run2 = (await second.json())['run_id']
            assert run2 != run1
            deadline = time.time() + 10
            while time.time() < deadline:
                stored = store._conn.execute(
                    'SELECT status_json FROM run_idempotency WHERE run_id=?', (run2,)).fetchone()
                if stored and json.loads(stored[0]).get('status') == 'running':
                    break
                await asyncio.sleep(0.05)
            else:
                raise AssertionError('generation 2 did not start')
            status = json.loads(stored[0])
            assert status['session_id'] == 'cont-tip'
            # The shape main persisted: the tip id only, no canonical field, no authority columns.
            status.pop('hosted_session_id', None)
            store._conn.execute(
                'UPDATE run_idempotency SET status_json=? WHERE run_id=?',
                (json.dumps(status), run2))
            store._conn.execute('DELETE FROM run_idempotency WHERE run_id=?', (run1,))
            store._conn.commit()
            _wipe_upgrade_state(store)
            hosted_rooms.reserve_peer_room(hosted_rooms.default_db_path(), claims={
                'room_id': 'room-1', 'home_install_id': 'home', 'authority_gateway_id': 'gateway-c',
                'authority_epoch': 3, 'member_id': 'member-1', 'target_profile': 'default',
                'target_install_id': catalog['installation_id']}, expires_at=time.time() + 3600)
            grant_c = _grant_for(adapter, catalog, gateway='gateway-c', epoch=3, member='member-1')
            before = create.call_count
            replay = await cli.post(
                '/v1/runs', headers={**_headers(grant_c), 'Idempotency-Key': 'room:task-1:2'},
                json=_generation_body(prompt, catalog, gateway='gateway-c', epoch=3, member='member-1', generation=2))
            payload = await replay.json()
            assert replay.status == 202, payload
            assert payload['run_id'] == run2 and payload['replayed'] is True
            assert create.call_count == before
            stopped = await cli.post(f'/v1/runs/{run2}/stop', headers=_headers(grant_c), json={})
            assert stopped.status == 200, await stopped.text()
            assert store.stop_requested(run2) is True
    finally:
        release.set()
        adapter._run_idempotency_store.close()


def _wipe_upgrade_state(store):
    _forget_room_stamp(store)
    for table in (
            'run_room_authorities', 'run_room_seen_gateways', 'run_room_authority_aliases',
            'run_room_namespaces', 'run_room_origins', 'run_room_origin_homes'):
        store._conn.execute(f'DELETE FROM {table}')
    store._conn.commit()


@pytest.mark.asyncio
async def test_http_public_run_is_not_replayed_as_the_room_generation(tmp_path, monkeypatch):
    """A migrated public row that carries the member session id is not the room replay."""
    adapter = _adapter(tmp_path / 'runs.db')
    create = MagicMock(return_value=_finishing_agent())
    monkeypatch.setattr(adapter, '_create_agent', create)
    prompt = 'Run this generation once.'
    try:
        application = app(adapter)
        async with TestClient(TestServer(application)) as cli:
            invited = await invite(cli, {
                'room_id': 'room-1', 'home_install_id': 'home', 'authority_gateway_id': 'gateway-a',
                'authority_epoch': 1, 'member_id': 'member-1'})
            assert invited.status == 201, await invited.text()
            catalog = (await invited.json())['catalog']
            grant_a = _grant_for(adapter, catalog, gateway='gateway-a', epoch=1, member='member-1')
            first = await cli.post(
                '/v1/runs', headers={**_headers(grant_a), 'Idempotency-Key': 'room:task-1:1'},
                json=_generation_body(prompt, catalog, gateway='gateway-a', epoch=1, member='member-1', generation=1))
            assert first.status == 202, await first.text()
            live = (await first.json())['run_id']
            await _await_runs(adapter)
            store = adapter._run_idempotency_store
            session_id = json.loads(store._conn.execute(
                'SELECT status_json FROM run_idempotency WHERE run_id=?', (live,)).fetchone()[0])['session_id']
            public = await cli.post(
                '/v1/runs', headers={'Authorization': 'Bearer test-room-key', 'Idempotency-Key': 'room:task-1:1'},
                json={'input': prompt, 'session_id': session_id})
            assert public.status == 202, await public.text()
            public_id = (await public.json())['run_id']
            assert public_id != live
            await _await_runs(adapter)
            sibling = await cli.post(
                '/v1/runs', headers={'Authorization': 'Bearer test-room-key', 'Idempotency-Key': 'ordinary-key'},
                json={'input': prompt, 'session_id': session_id})
            assert sibling.status == 202, await sibling.text()
            sibling_id = (await sibling.json())['run_id']
            await _await_runs(adapter)
            # Migration adds client_replay as 0. This branch's own public reserve writes 1.
            store._conn.execute(
                """UPDATE run_idempotency
                   SET client_replay=0, updated_at=updated_at+10,
                       status_json=json_set(status_json, '$.status', 'completed')
                   WHERE run_id IN (?, ?)""", (public_id, sibling_id))
            store._conn.commit()
            _wipe_upgrade_state(store)
            hosted_rooms.reserve_peer_room(hosted_rooms.default_db_path(), claims={
                'room_id': 'room-1', 'home_install_id': 'home', 'authority_gateway_id': 'gateway-b',
                'authority_epoch': 2, 'member_id': 'member-1', 'target_profile': 'default',
                'target_install_id': catalog['installation_id']}, expires_at=time.time() + 3600)
            grant_b = _grant_for(adapter, catalog, gateway='gateway-b', epoch=2, member='member-1')
            replay = await cli.post(
                '/v1/runs', headers={**_headers(grant_b), 'Idempotency-Key': 'room:task-1:1'},
                json=_generation_body(prompt, catalog, gateway='gateway-b', epoch=2, member='member-1', generation=1))
            payload = await replay.json()
            assert replay.status == 202, payload
            assert payload['run_id'] == live and payload['replayed'] is True
            assert store.stop_requested(live) is False
            hidden = await cli.get(f'/v1/runs/{public_id}', headers=_headers(grant_b))
            assert hidden.status == 404
            other = await cli.get(f'/v1/runs/{sibling_id}', headers=_headers(grant_b))
            assert other.status == 404
    finally:
        adapter._run_idempotency_store.close()


@pytest.mark.asyncio
async def test_http_unresolvable_running_legacy_row_is_not_started_again(tmp_path, monkeypatch):
    """A running pre-column row whose session cannot be walked must not admit a second agent."""
    import threading
    adapter = _adapter(tmp_path / 'runs.db')
    release = threading.Event()
    create = MagicMock()

    def agent(*_args, **_kwargs):
        built = MagicMock()
        built.session_prompt_tokens = built.session_completion_tokens = built.session_total_tokens = 0
        def block(**_ignored):
            release.wait(15)
            return {'interrupted': True}
        built.run_conversation.side_effect = block
        return built

    create.side_effect = agent
    monkeypatch.setattr(adapter, '_create_agent', create)
    prompt = 'Run this generation once.'
    try:
        application = app(adapter)
        async with TestClient(TestServer(application)) as cli:
            invited = await invite(cli, {
                'room_id': 'room-1', 'home_install_id': 'home', 'authority_gateway_id': 'gateway-a',
                'authority_epoch': 1, 'member_id': 'member-1'})
            assert invited.status == 201, await invited.text()
            catalog = (await invited.json())['catalog']
            grant_a = _grant_for(adapter, catalog, gateway='gateway-a', epoch=1, member='member-1')
            first = await cli.post(
                '/v1/runs', headers={**_headers(grant_a), 'Idempotency-Key': 'room:task-1:1'},
                json=_generation_body(prompt, catalog, gateway='gateway-a', epoch=1, member='member-1', generation=1))
            assert first.status == 202, await first.text()
            live = (await first.json())['run_id']
            store = adapter._run_idempotency_store
            deadline = time.time() + 10
            while time.time() < deadline:
                row = store._conn.execute(
                    'SELECT status_json FROM run_idempotency WHERE run_id=?', (live,)).fetchone()
                if row and json.loads(row[0]).get('status') == 'running':
                    break
                await asyncio.sleep(0.05)
            else:
                raise AssertionError('run did not start')
            status = json.loads(row[0])
            status['session_id'] = 'missing-tip'
            status.pop('hosted_session_id', None)
            store._conn.execute(
                'UPDATE run_idempotency SET status_json=? WHERE run_id=?', (json.dumps(status), live))
            store._conn.commit()
            _wipe_upgrade_state(store)
            hosted_rooms.reserve_peer_room(hosted_rooms.default_db_path(), claims={
                'room_id': 'room-1', 'home_install_id': 'home', 'authority_gateway_id': 'gateway-b',
                'authority_epoch': 2, 'member_id': 'member-1', 'target_profile': 'default',
                'target_install_id': catalog['installation_id']}, expires_at=time.time() + 3600)
            grant_b = _grant_for(adapter, catalog, gateway='gateway-b', epoch=2, member='member-1')
            before = create.call_count
            replay = await cli.post(
                '/v1/runs', headers={**_headers(grant_b), 'Idempotency-Key': 'room:task-1:1'},
                json=_generation_body(prompt, catalog, gateway='gateway-b', epoch=2, member='member-1', generation=1))
            body = await replay.json()
            assert replay.status == 409, body
            assert body['error']['code'] == 'legacy_run_unresolved'
            assert create.call_count == before
            assert store.stop_requested(live) is False
    finally:
        release.set()
        adapter._run_idempotency_store.close()


def test_unbound_invitation_does_not_reserve_peer_room(monkeypatch):
    import tui_gateway.methods_groups as groups
    monkeypatch.setattr(groups, '_bound_server', None)

    def reserved(*_args, **_kwargs):
        raise AssertionError('reserved')

    monkeypatch.setattr(hosted_rooms, 'reserve_peer_room', reserved)
    with pytest.raises(ValueError, match='gateway run store'):
        groups._record_peer_run_authority({'room_id': 'room-1'})
