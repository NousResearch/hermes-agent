"""Desktop chooser ``session.export`` through real ``AuthorityConnection.dispatch``.

The chooser sends the eight boolean ``transcript_options`` and expects the filtered
persisted transcript plus ``transcript_version`` 1. A payload without options stays
the old ``/save`` snapshot. Dispatch, ``export_session`` and authorization are real;
the database is a synthetic ``SessionDB`` under the test tmp dir.
"""
import json

import pytest

from gateway.config import Platform
from tests.gateway.test_session_authority_cancel_settlement import _authority

CHOOSER = {
    'include_tools': False,
    'include_reasoning': False,
    'include_images': True,
    'include_videos': True,
    'include_files': True,
    'include_links': True,
    'include_asides': True,
    'collapse_details': True,
}
KEPT = 'kept user text'
SECRET = 'secret tool output'
PRIVATE = 'private system prompt'


def _seed(db):
    db.append_message('s', 'system', PRIVATE)
    db.append_message('s', 'user', KEPT)
    db.append_message('s', 'tool', SECRET, tool_name='lookup')


def _connection(authority, **identity):
    from gateway.session_controls import AuthorityConnection
    merged = {'user_id': 'human'}
    merged.update(identity)
    return AuthorityConnection(authority, object(), merged)


async def _export(authority, *, identity=None, **params):
    conn = _connection(authority, **(identity or {}))
    try:
        return await conn.dispatch({
            'id': 1, 'method': 'session.export',
            'params': {'session_id': 's', **params},
        })
    finally:
        await conn.close()


def _reason(reply):
    return (reply.get('error') or {}).get('data', {}).get('reason')


@pytest.mark.asyncio
async def test_chooser_dispatch_returns_filtered_persisted_transcript(tmp_path, monkeypatch):
    db, authority = _authority(tmp_path, monkeypatch)
    try:
        _seed(db)
        reply = await _export(authority, format='json', transcript_options=CHOOSER)
        assert _reason(reply) is None, reply
        result = reply['result']
        assert result['transcript_version'] == 1
        assert result['format'] == 'json'
        doc = json.loads(result['content'])
        assert [row['text'] for row in doc['messages']] == [KEPT]
        assert SECRET not in result['content'] and PRIVATE not in result['content']
        stored = [row['content'] for row in db.get_messages('s')]
        assert KEPT in stored and SECRET in stored and PRIVATE in stored
    finally:
        db.close()


@pytest.mark.asyncio
async def test_legacy_dispatch_keeps_unfiltered_snapshot_without_version(tmp_path, monkeypatch):
    db, authority = _authority(tmp_path, monkeypatch)
    try:
        _seed(db)
        reply = await _export(authority)
        assert _reason(reply) is None, reply
        result = reply['result']
        assert 'transcript_version' not in result
        assert result['format'] == 'json' and KEPT in result['content']
        assert SECRET in result['content'] and PRIVATE in result['content']
        stored = [row['content'] for row in db.get_messages('s')]
        assert SECRET in stored
    finally:
        db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('platform,identity,extra,reason,stranger', [
    (Platform.LOCAL, {}, {}, 'permission_denied', True),
    (None, {'profile_id': 'foreign'}, {}, 'profile_mismatch', False),
    (None, {'capabilities': ['session:submit']}, {}, 'permission_denied', False),
    (None, {}, {'profile': 'foreign'}, 'profile_mismatch', False),
])
async def test_legacy_dispatch_denies_wrong_owner_profile_and_capability(
        tmp_path, monkeypatch, platform, identity, extra, reason, stranger):
    db, authority = _authority(tmp_path, monkeypatch, platform=platform)
    try:
        if stranger:
            authority.sessions['s'].source.user_id = 'other-human'
        _seed(db)
        reply = await _export(authority, identity=identity, **extra)
        assert SECRET not in json.dumps(reply)
        assert _reason(reply) == reason, reply
        assert 'result' not in reply
    finally:
        db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('platform,identity,extra,reason,stranger', [
    (Platform.LOCAL, {}, {}, 'permission_denied', True),
    (None, {'profile_id': 'foreign'}, {}, 'profile_mismatch', False),
    (None, {'capabilities': ['session:submit']}, {}, 'permission_denied', False),
    (None, {}, {'profile': 'foreign'}, 'profile_mismatch', False),
])
async def test_chooser_dispatch_denies_wrong_owner_profile_and_capability(
        tmp_path, monkeypatch, platform, identity, extra, reason, stranger):
    db, authority = _authority(tmp_path, monkeypatch, platform=platform)
    try:
        if stranger:
            authority.sessions['s'].source.user_id = 'other-human'
        _seed(db)
        reply = await _export(authority, identity=identity, format='json',
                              transcript_options=CHOOSER, **extra)
        assert SECRET not in json.dumps(reply)
        assert _reason(reply) == reason, reply
        assert 'result' not in reply
        assert SECRET in [row['content'] for row in db.get_messages('s')]
    finally:
        db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('options', [
    {**CHOOSER, 'include_secrets': True},
    {key: value for key, value in CHOOSER.items() if key != 'include_tools'},
    {**CHOOSER, 'include_tools': 'no'},
    ['include_tools'],
    None,
])
async def test_invalid_transcript_options_fail_closed(tmp_path, monkeypatch, options):
    db, authority = _authority(tmp_path, monkeypatch)
    try:
        _seed(db)
        reply = await _export(authority, format='json', transcript_options=options)
        assert SECRET not in json.dumps(reply)
        assert _reason(reply) == 'invalid_params', reply
        assert 'result' not in reply
        assert SECRET in [row['content'] for row in db.get_messages('s')]
    finally:
        db.close()
