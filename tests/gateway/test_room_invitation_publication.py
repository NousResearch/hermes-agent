"""Invitation publication keeps reservation and admission authority consistent."""
import asyncio
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from aiohttp.test_utils import TestClient, TestServer

from gateway import hosted_rooms
from gateway.hosted_room_peer import decode_room_grant
from gateway.platforms.api_server_room_grants import _record_invitation
from gateway.platforms.api_server_run_authority import room_authority, room_namespace, room_run_scope
from gateway.platforms.api_server_run_idempotency import RunIdempotencyStore
from tests.gateway.test_api_server_room_cancellation import _adapter, _headers, _invitation
from tests.gateway.test_room_cancellation_retirement_http import app


def claims():
    return dict(room_id='room-1', home_install_id='home', authority_gateway_id='home',
                authority_epoch=1, member_id='member-1', target_install_id='target',
                target_profile='default', permissions=['dispatch', 'status', 'retire'],
                expires_at=time.time() + 3600)


class CommitProbe:
    """Fault/pause only the physical RunStore commit; all SQL remains real SQLite."""
    def __init__(self, connection, before_commit):
        self.connection, self.before_commit = connection, before_commit

    def __getattr__(self, name):
        return getattr(self.connection, name)

    def commit(self):
        self.before_commit()
        return self.connection.commit()


def test_failed_reservation_keeps_the_previous_floor_and_grant(tmp_path, monkeypatch):
    path = tmp_path / 'grants.db'
    monkeypatch.setattr(hosted_rooms, 'default_db_path', lambda: path)
    store = RunIdempotencyStore(str(tmp_path / 'runs.db'))
    adapter, old = SimpleNamespace(_run_idempotency_store=store), claims()
    try:
        _record_invitation(adapter, old, {})
        with hosted_rooms._transaction(path, immediate=True) as conn:
            conn.execute("""CREATE TRIGGER reject_new_reservation BEFORE INSERT
                ON hosted_room_peer_reservations WHEN NEW.authority_epoch=2
                BEGIN SELECT RAISE(ABORT, 'reservation rollback'); END""")
        with pytest.raises(sqlite3.IntegrityError, match='reservation rollback'):
            _record_invitation(adapter, {**old, 'authority_epoch': 2}, {})
        assert hosted_rooms.peer_room_grant_is_current(path, claims=old)
        assert store.accepts_room_authority(room_authority(old), namespace=room_namespace(old), claims=old)
    finally:
        store.close()


@pytest.mark.parametrize('advanced_member', ['member-1', 'member-2'])
def test_other_connections_cannot_pass_the_publication_window(tmp_path, monkeypatch, advanced_member):
    grant_path, run_path = tmp_path / 'grants.db', str(tmp_path / 'runs.db')
    monkeypatch.setattr(hosted_rooms, 'default_db_path', lambda: grant_path)
    publisher, captured, competing = [RunIdempotencyStore(run_path) for _ in range(3)]
    adapter = SimpleNamespace(_run_idempotency_store=publisher)
    old = claims()
    reached, release, captured_started = threading.Event(), threading.Event(), threading.Event()
    try:
        _record_invitation(adapter, old, {})
        if advanced_member != old['member_id']:
            _record_invitation(adapter, {**old, 'member_id': advanced_member}, {})
        newer = {**old, 'member_id': advanced_member, 'authority_epoch': 2}
        original = publisher._conn

        def pause_commit():
            reached.set()
            assert release.wait(10), 'publication barrier was not released'

        publisher._conn = CommitProbe(original, pause_commit)
        with ThreadPoolExecutor(max_workers=3) as pool:
            publishing = pool.submit(_record_invitation, adapter, newer, {})
            try:
                assert reached.wait(10)
                assert hosted_rooms.peer_room_grant_is_current(grant_path, claims=newer)

                def captured_write():
                    captured_started.set()
                    return captured.reserve(room_run_scope(old), 'room:old:1', 'hash', 'late',
                                            {'status': 'queued'}, room_authority=room_authority(old))

                pending = pool.submit(captured_write)
                unrelated = {**newer, 'home_install_id': 'foreign', 'authority_gateway_id': 'foreign',
                             'authority_epoch': 3, 'member_id': 'other-member'}
                collision = pool.submit(_record_invitation,
                    SimpleNamespace(_run_idempotency_store=competing), unrelated, {})
                assert captured_started.wait(10)
                assert not pending.done()
            finally:
                release.set()
            publishing.result(timeout=10)
            assert pending.result(timeout=10) == ('authority_retired', None)
            with pytest.raises(ValueError, match='authority has already advanced'):
                collision.result(timeout=10)
        assert publisher._conn.execute('SELECT COUNT(*) FROM run_idempotency').fetchone()[0] == 0
        assert hosted_rooms.peer_room_grant_is_current(grant_path, claims=newer)
    finally:
        release.set()
        for store in (publisher, captured, competing):
            store.close()


@pytest.mark.asyncio
async def test_failed_floor_commit_cannot_admit_a_captured_old_request(tmp_path, monkeypatch):
    adapter = _adapter(tmp_path / 'runs.db')
    entered, release = asyncio.Event(), asyncio.Event()
    history = adapter._conversation_history_for_session

    async def paused(*args, **kwargs):
        entered.set()
        await release.wait()
        return await history(*args, **kwargs)

    create = MagicMock(side_effect=AssertionError('old authority executed'))
    monkeypatch.setattr(adapter, '_create_agent', create)
    monkeypatch.setattr(adapter, '_conversation_history_for_session', paused)
    store = adapter._run_idempotency_store
    original = store._conn
    try:
        async with TestClient(TestServer(app(adapter))) as cli:
            grant, body = await _invitation(cli)
            old = decode_room_grant(adapter._room_grant_secret(), grant, permission='status')
            pending = asyncio.create_task(cli.post('/v1/runs', headers=_headers(grant), json=body))
            await asyncio.wait_for(entered.wait(), 10)

            def fail_commit():
                raise sqlite3.OperationalError('simulated RunStore commit failure')

            store._conn = CommitProbe(original, fail_commit)
            response = await cli.post('/v1/room-members/invitations',
                headers={'Authorization': 'Bearer test-room-key'}, json={
                    'room_id': 'room-1', 'home_install_id': 'home', 'authority_gateway_id': 'home',
                    'authority_epoch': 2, 'member_id': 'member-1'})
            assert response.status >= 400
            store._conn = original
            assert store.accepts_room_authority(room_authority(old), namespace=room_namespace(old), claims=old)
            assert not hosted_rooms.peer_room_grant_is_current(hosted_rooms.default_db_path(), claims=old)
            release.set()
            refused = await pending
            assert refused.status == 409
            assert (await refused.json())['error']['code'] == 'run_history_retired'
            assert not create.called
            # Retrying the failed issuance repairs the floor without reviving old work.
            repaired = await cli.post('/v1/room-members/invitations',
                headers={'Authorization': 'Bearer test-room-key'}, json={
                    'room_id': 'room-1', 'home_install_id': 'home', 'authority_gateway_id': 'home',
                    'authority_epoch': 2, 'member_id': 'member-1'})
            assert repaired.status == 201
            assert not store.accepts_room_authority(room_authority(old), namespace=room_namespace(old), claims=old)
    finally:
        release.set()
        store._conn = original
        store.close()
