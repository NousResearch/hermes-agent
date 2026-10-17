"""Deleting a session and recovering API turns stay under SQLite's expression limit (#140090)."""
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


async def _delete(api, session_id):
    app = web.Application()
    app.router.add_delete('/api/sessions/{session_id}', api._handle_delete_session)
    async with TestClient(TestServer(app)) as client:
        deleted = await client.delete('/api/sessions/' + session_id)
        return deleted.status, await deleted.text()


def _explode(conn, references, after_id=0):
    raise sqlite3.OperationalError('database disk image is malformed')


@pytest.mark.asyncio
@pytest.mark.parametrize('count', [10, 1010])
async def test_delete_session_with_many_api_images_succeeds(api, owner, count):
    """The history scan's statement does not grow with the image count, so a thousand images
    do not 500 the delete, and only the image a transcript still names stays."""
    _settle_text(api, owner, 'chat')
    planted = _plant(count)
    kept_digest, kept_path = planted[-1]
    _mention(owner, kept_digest, kept_path.name)
    status, body = await _delete(api, 'chat')
    assert status == 200, body
    assert owner.db.get_session('chat') is None
    assert _remaining(planted) == [kept_digest]


@pytest.mark.asyncio
async def test_a_cleanup_failure_after_the_delete_committed_is_not_a_500(api, owner, monkeypatch):
    """The session is already gone when the image sweep runs: its failure is logged and the
    files stay for the next sweep, but the finished delete answers 200."""
    _settle_text(api, owner, 'chat')
    planted = _plant(1)
    monkeypatch.setattr('gateway.session_ingress_media._history_mentions', _explode)
    status, body = await _delete(api, 'chat')
    assert status == 200, body
    assert owner.db.get_session('chat') is None
    assert _remaining(planted) == [planted[0][0]]


def test_startup_recovery_survives_a_thousand_api_images(api, owner, monkeypatch):
    """Image collection at startup must not abort recovery of a queued API turn."""
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


def test_startup_recovery_continues_when_image_collection_fails(api, owner, monkeypatch, caplog):
    """A collection error is logged and skipped, never a reason to leave queued turns unscheduled."""
    admit_api_turn(
        api, session_id='queued', request_id='queued-turn',
        user_message='pending', conversation_history=[])
    _plant(1)
    monkeypatch.setattr('gateway.session_ingress_media._history_mentions', _explode)
    scheduled = []
    monkeypatch.setattr(owner, '_schedule', lambda ref: scheduled.append(ref.session_id))
    recover_api_turns(api)
    assert scheduled == ['queued']
    assert 'API image collection failed' in caplog.text
