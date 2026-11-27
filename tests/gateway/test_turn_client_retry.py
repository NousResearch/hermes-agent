"""The owner's client stamp (``surface_v1.client``) is not part of a turn's exact-retry identity.

An app updated between a send and its retry announces another version, and a reconnect may not
have said hello yet: the retry of the same request id is still the same request and gets the first
row back. Everything else in the payload still decides. Driven through the native socket's real
``prompt.submit``.
"""
from __future__ import annotations

from contextlib import closing
import sqlite3

import pytest

from tests.gateway.test_turn_client_stamp import DESKTOP, _authority, _Client, _rows

UPDATED = dict(DESKTOP, version='2.2.0')
# ``admission_fingerprint`` of ``{'text': 'same words'}`` on target ``s``, intent ``queue``, as the base
# commit (before the stamp existed) stores it: rows without a stamp keep their digests.
UNSTAMPED_DIGEST = '04b1fb6260c482642925cf4f3575925d2b96b74ff758884a8e522e25a00aad1f'


async def _owner(tmp_path, monkeypatch):
    from gateway.config import Platform
    from gateway.session import SessionSource
    from gateway.owner_identity import principal_subject
    from gateway.session_authority import LiveSession

    async def answer(event):
        return 'ok'
    authority = await _authority(tmp_path / 'home', monkeypatch, answer)
    # The session's own user, so the committed payload is the input alone (no operator envelope).
    source = SessionSource(platform=Platform.TELEGRAM, chat_id='c', user_id=principal_subject({'user_id': 'owner'}))
    authority.sessions['s'] = LiveSession(source, 's')
    return authority


async def _client(authority, label):
    client = _Client(authority)
    if label is not None:
        assert 'result' in await client.call('client.capabilities', client=label)
    await client.call('session.resume', session_id='s')
    return client


async def _submit(authority, label, **params):
    client = await _client(authority, label)
    try:
        reply = await client.call('prompt.submit', session_id='s', submission_id='one', **params)
        if 'result' in reply and authority.sessions['s'].task is not None:
            await authority.sessions['s'].task
        return reply
    finally:
        await client.connection.close()


def _digest(authority, request_id):
    with closing(sqlite3.connect(authority.persistence.scope.profile_dir / 'state.db')) as db:
        return db.execute('SELECT payload_digest FROM session_admissions WHERE request_id=?', (request_id,)).fetchone()[0]


@pytest.mark.asyncio
@pytest.mark.parametrize('first, retry', [(DESKTOP, UPDATED), (DESKTOP, None), (None, DESKTOP)],
                         ids=['app-updated', 'retry-unannounced', 'first-unannounced'])
async def test_a_retry_from_another_or_no_label_is_the_same_request(tmp_path, monkeypatch, first, retry):
    authority = await _owner(tmp_path, monkeypatch)
    sent = await _submit(authority, first, text='same words')
    again = await _submit(authority, retry, text='same words')
    assert 'result' in again, again
    assert again['result']['admission_id'] == sent['result']['admission_id']
    rows = _rows(authority)
    assert len(rows) == 1
    assert rows['one'].get('surface_v1') == ({'client': first} if first else None), 'the first stamp is kept'
    assert _digest(authority, 'one') == UNSTAMPED_DIGEST


@pytest.mark.asyncio
@pytest.mark.parametrize('changed', [{'text': 'other words'}, {'text': 'same words', 'surface': 'hud'}],
                         ids=['text', 'surface'])
async def test_a_retry_that_changes_anything_else_is_still_a_conflict(tmp_path, monkeypatch, changed):
    authority = await _owner(tmp_path, monkeypatch)
    assert 'result' in await _submit(authority, DESKTOP, text='same words')
    conflict = await _submit(authority, UPDATED, **changed)
    assert conflict['error']['message'] == 'admission_conflict', conflict


@pytest.mark.asyncio
async def test_an_unstamped_row_keeps_the_digest_it_had_before_the_stamp(tmp_path, monkeypatch):
    """Control: an unannounced submit stores exactly the digest the base commit stores."""
    authority = await _owner(tmp_path, monkeypatch)
    assert 'result' in await _submit(authority, None, text='same words')
    assert 'surface_v1' not in _rows(authority)['one']
    assert _digest(authority, 'one') == UNSTAMPED_DIGEST
