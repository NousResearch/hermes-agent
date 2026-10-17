"""Deleting a session and recovering API turns stay under SQLite's expression limit."""
import sqlite3

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.session_api_turn import admit_api_turn, recover_api_turns
from gateway.session_ingress_media import _media_root
from hermes_state_runtime import claim_session_input, settle_session_input


def _plant(count):
    """``count`` retained API images, digests ordered so the last sorts last."""
    root = _media_root()
    planted = []
    for index in range(count):
        digest = f'{index:064x}'
        directory = root / digest
        directory.mkdir(parents=True)
        path = directory / f'api_{digest[:32]}.png'
        path.write_bytes(b'img' + index.to_bytes(4, 'big'))
        planted.append((digest, path))
    return planted


def _mention(owner, digest, name):
    owner.db.create_session('keeper', source='cli')
    owner.db.append_message('keeper', 'user', f'notes {digest}/{name}')


def _remaining(planted):
    return [digest for digest, path in planted if path.exists()]


def _settle_text(api, owner, session_id):
    _, _, row = admit_api_turn(
        api, session_id=session_id, request_id='turn-' + session_id,
        user_message='hello', conversation_history=[])
    started = claim_session_input(owner.db, epoch=owner.epoch, session_id=row['target_session_id'])
    settle_session_input(
        owner.db, epoch=owner.epoch, admission_id=row['admission_id'],
        generation=started['generation'], outcome='completed')
    return row


@pytest.mark.asyncio
@pytest.mark.parametrize('count', [10, 1010])
async def test_delete_session_with_many_api_images_succeeds(api, owner, count):
    """The history scan is batched, so a thousand images do not 500 the delete."""
    _settle_text(api, owner, 'chat')
    planted = _plant(count)
    kept_digest, kept_path = planted[-1]
    _mention(owner, kept_digest, kept_path.name)
    app = web.Application()
    app.router.add_delete('/api/sessions/{session_id}', api._handle_delete_session)
    async with TestClient(TestServer(app)) as client:
        deleted = await client.delete('/api/sessions/chat')
        body = await deleted.text()
        assert deleted.status == 200, body
        assert (await deleted.json())['deleted'] is True
    assert owner.db.get_session('chat') is None
    assert _remaining(planted) == [kept_digest]


@pytest.mark.asyncio
async def test_media_index_failure_does_not_commit_the_delete(api, owner, monkeypatch):
    """A history-scan failure leaves the session row in place instead of 500 after it is gone."""
    _settle_text(api, owner, 'chat')
    _plant(1)

    def explode(conn, references, after_id=0):
        raise sqlite3.OperationalError('Expression tree is too large (maximum depth 1000)')

    monkeypatch.setattr('gateway.session_ingress_media._history_mentions', explode)
    app = web.Application()
    app.router.add_delete('/api/sessions/{session_id}', api._handle_delete_session)
    async with TestClient(TestServer(app)) as client:
        deleted = await client.delete('/api/sessions/chat')
        assert deleted.status == 500, await deleted.text()
    assert owner.db.get_session('chat') is not None


def test_startup_recovery_survives_a_thousand_api_images(api, owner, monkeypatch):
    """Image collection must not abort recovery of a queued API turn."""
    _, _, row = admit_api_turn(
        api, session_id='queued', request_id='queued-turn',
        user_message='pending', conversation_history=[])
    assert row['status'] == 'queued'
    planted = _plant(1010)
    kept_digest, kept_path = planted[-1]
    _mention(owner, kept_digest, kept_path.name)
    scheduled = []
    monkeypatch.setattr(owner, '_schedule', lambda ref: scheduled.append(ref.session_id))
    recover_api_turns(api)
    assert scheduled == ['queued']
    assert _remaining(planted) == [kept_digest]


def test_startup_recovery_continues_when_image_collection_fails(api, owner, monkeypatch):
    """A collection error is not a reason to leave queued API turns unscheduled."""
    admit_api_turn(
        api, session_id='queued', request_id='queued-turn',
        user_message='pending', conversation_history=[])
    _plant(1)

    def explode(conn, references, after_id=0):
        raise sqlite3.OperationalError('Expression tree is too large (maximum depth 1000)')

    monkeypatch.setattr('gateway.session_ingress_media._history_mentions', explode)
    scheduled = []
    monkeypatch.setattr(owner, '_schedule', lambda ref: scheduled.append(ref.session_id))
    recover_api_turns(api)
    assert scheduled == ['queued']
