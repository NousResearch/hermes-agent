"""Canonical stored previews reach the wire without adopting history or leaking other roles."""
import json
from unittest.mock import Mock

import pytest

from tests.gateway.test_session_listing_subagents import _native, _snapshot


async def _dispatch(native, method='session.list', **params):
    reply = await native.dispatch({'id': 1, 'method': method, 'params': params})
    assert 'result' in reply, reply
    return json.loads(json.dumps(reply))['result']


def _seed(db, sid, messages, *, title=None, **kwargs):
    db.create_session(sid, source='tui', model='preview-model', **kwargs)
    if title is not None:
        db.set_session_title(sid, title)
    for role, text in messages:
        db.append_message(sid, role, text)
    db.end_session(sid, 'tui_shutdown')


@pytest.mark.asyncio
async def test_real_dispatch_serializes_first_user_previews_and_exports_consumer_payload(tmp_path, monkeypatch):
    from hermes_state_common import _PREVIEW_MAX_CHARS
    authority, db, native = await _native(tmp_path, monkeypatch)
    authority._schedule = Mock(side_effect=AssertionError('listing must not schedule inference'))
    _seed(db, 'preview', [('system', 'FORBIDDEN_SYSTEM'), ('assistant', 'FORBIDDEN_ASSISTANT'),
                          ('user', 'First user preview'), ('user', 'NOT_THE_FIRST_USER')])
    _seed(db, 'titled', [('user', 'Preview loses to title')], title='Explicit title')
    for role in ('system', 'assistant', 'tool'):
        _seed(db, role, [(role, 'FORBIDDEN_' + role.upper())])
    _seed(db, 'empty', [])
    _seed(db, 'long', [('user', 'x' * 10000)])
    for sid, extra in [('hidden', {}), ('foreign', {'profile_name': 'foreign'}),
                       ('foreign-owner', {'user_id': 'foreign'})]:
        _seed(db, sid, [('user', 'PRIVATE_' + sid)], **extra)
    db.set_session_hidden('hidden', True)
    before = _snapshot(db)
    payload = await _dispatch(native, scope='stored')
    rows = {row['id']: row for row in payload['sessions']}
    assert set(rows) == {'preview', 'titled', 'system', 'assistant', 'tool', 'empty', 'long'}
    assert rows['preview']['preview'] == 'First user preview'
    assert rows['titled']['title'] == 'Explicit title'
    assert rows['titled']['preview'] == 'Preview loses to title'
    assert all(rows[sid]['preview'] == '' for sid in ('system', 'assistant', 'tool', 'empty'))
    assert rows['long']['preview'] == 'x' * _PREVIEW_MAX_CHARS + '...'
    assert all(row['id'] == row['session_id'] == row['resolved_id'] for row in rows.values())
    assert all(isinstance(row['preview'], str) for row in rows.values())
    assert 'FORBIDDEN_' not in json.dumps(payload)
    assert 'PRIVATE_' not in json.dumps(payload)
    assert _snapshot(db) == before
    assert not authority.runner.adapters
    authority._schedule.assert_not_called()
    # The scratch test artifact is copied verbatim into the mounted Ink consumer fixture.
    (tmp_path / 'dispatch.json').write_text(json.dumps({'id': 1, 'result': payload}, indent=2) + '\n')


@pytest.mark.asyncio
@pytest.mark.parametrize('compressed', [False, True])
async def test_closed_canonical_local_preview_keeps_owner_and_tip_identity(tmp_path, monkeypatch, compressed):
    from gateway import session_listing
    _authority, db, native = await _native(tmp_path, monkeypatch)
    _seed(db, 'closed-owner', [('user', 'Closed first user')])
    resumed = await _dispatch(native, 'session.resume', session_id='closed-owner')
    if compressed:
        db.reopen_session('closed-owner')
        assert db.try_acquire_compression_lock('closed-owner', 'preview-fixture')
        try:
            db.publish_compression_child(parent_session_id='closed-owner', child_session_id='closed-tip',
                source='tui', model='preview-model', messages=[{'role': 'user', 'content': 'Closed tip user'}],
                compression_lock_holder='preview-fixture')
        finally:
            db.release_compression_lock('closed-owner', 'preview-fixture')
    else:
        db.reopen_session('closed-owner')
    await _dispatch(native, 'session.detach', session_id='closed-owner',
                    subscription_id=resumed['subscription_id'], end_reason='tui_close')
    original = session_listing._closed_local_session
    projected = []
    def observed(*args):
        result = original(*args)
        if result is not None:
            projected.append(result)
        return result
    monkeypatch.setattr(session_listing, '_closed_local_session', observed)
    before = _snapshot(db)
    rows = (await _dispatch(native, scope='stored'))['sessions']
    assert len(rows) == 1
    row = rows[0]
    assert row['id'] == row['session_id'] == row['resolved_id'] == 'closed-owner'
    assert row['preview'] == ('Closed tip user' if compressed else 'Closed first user')
    assert projected == rows
    assert db.get_compression_tip('closed-owner') == ('closed-tip' if compressed else 'closed-owner')
    assert _snapshot(db) == before


@pytest.mark.asyncio
async def test_compressed_history_uses_existing_storage_preview_and_resolved_tip(tmp_path, monkeypatch):
    _authority, db, native = await _native(tmp_path, monkeypatch)
    _seed(db, 'root', [('user', 'Root user')])
    db.reopen_session('root')
    assert db.try_acquire_compression_lock('root', 'preview-fixture')
    try:
        db.publish_compression_child(parent_session_id='root', child_session_id='tip', source='tui',
            messages=[{'role': 'user', 'content': 'Tip user'}, {'role': 'assistant', 'content': 'FORBIDDEN_TIP'}],
            compression_lock_holder='preview-fixture')
    finally:
        db.release_compression_lock('root', 'preview-fixture')
    db.end_session('tip', 'tui_shutdown')
    before = _snapshot(db)
    rows = (await _dispatch(native, scope='stored'))['sessions']
    assert [(row['id'], row['session_id'], row['resolved_id'], row['preview']) for row in rows] == [
        ('tip', 'tip', 'tip', 'Tip user')]
    assert _snapshot(db) == before


@pytest.mark.parametrize('row_preview,tip_preview,expected', [
    ('Row first user', 'Tip first user', 'Row first user'),
    ('', 'Tip first user', 'Tip first user'), (None, '', ''), (None, None, ''),
])
def test_both_projectors_use_row_then_tip_then_empty(row_preview, tip_preview, expected):
    # The closed branch still executes its real owner/authorization and closed-view checks.
    from types import SimpleNamespace
    from gateway.session_contract import SessionRef
    from gateway.session_listing import _closed_local_session, _picker_row
    row = {'id': 'row', 'preview': row_preview}
    tip = {'id': 'tip', 'source': 'tui', 'chat_id': 'owner', 'preview': tip_preview}
    handle = SimpleNamespace(ref=SessionRef('p', 'owner'), execution_state='idle')
    authority = SimpleNamespace(profile_id='p', sessions={'owner': SimpleNamespace(subscribers={}, detached=True)},
                                _handle=lambda ref: handle, authorize=Mock())
    assert _picker_row(row, tip, 'tip')['preview'] == expected
    assert _closed_local_session(authority, object(), row, tip)['preview'] == expected
    authority.authorize.assert_called_once_with(authority.authorize.call_args.args[0], handle.ref, 'session:read')
