"""Actual authority dispatch paging: eligible ranks, raw windows and nonmutating failure."""
import sqlite3
from itertools import pairwise

import pytest

from tests.gateway.test_session_listing_subagents import _native, _row, _snapshot


async def _call(native, **params):
    return await native.dispatch({'id': 1, 'method': 'session.list', 'params': params})


def _stable(db):
    db._write_sql('UPDATE sessions SET started_at=1000')
    db._write_sql('UPDATE messages SET timestamp=1000')


@pytest.mark.asyncio
@pytest.mark.parametrize('count', [0, 199, 200, 201, 250])
async def test_exact_depth_and_ties(tmp_path, monkeypatch, count):
    from tui_gateway.contracts.canonical_sessions import CanonicalListResult
    _, db, native = await _native(tmp_path, monkeypatch)
    for i in range(count):
        _row(db, f'good-{i:04}')
    _stable(db)
    expected = [f'good-{i:04}' for i in reversed(range(count))]
    before = _snapshot(db)
    default = (await _call(native, scope='stored'))['result']
    assert default['offset'] == 0 and default['limit'] == 200
    assert default['has_more'] == (count > 200)
    actual = []
    for offset in range(0, count + 51, 50):
        result = (await _call(native, scope='stored', offset=offset, limit=50))['result']
        CanonicalListResult.model_validate(result)
        assert [r['id'] for r in result['sessions']] == expected[offset:offset + 50]
        assert result['has_more'] == (count > offset + 50)
        actual.extend(r['id'] for r in result['sessions'])
    assert actual == expected
    assert _snapshot(db) == before


@pytest.mark.asyncio
@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('count', [199, 200, 201])
async def test_denied_first_chunk_and_mixed_boundaries(tmp_path, monkeypatch, enabled, count):
    import gateway.session_listing as listing
    _, db, native = await _native(tmp_path, monkeypatch, enabled=enabled)
    for i in range(200):
        _row(db, f'z-denied-{i:04}', user_id='foreign')
    for i in range(count):
        _row(db, f'good-{i:04}')
    _row(db, 'child', parent='good-0000', source='subagent')
    _row(db, 'inherited', parent='good-0000')
    _row(db, 'internal', source='tool')
    _row(db, 'hidden')
    db.set_session_hidden('hidden', True)
    _row(db, 'foreign-profile', profile_name='foreign')
    _row(db, 'bound-foreign')
    from hermes_state_mutation_binding import BINDING_PREFIX
    db._write_sql('INSERT INTO state_meta(key,value) VALUES(?,?)',
                 (BINDING_PREFIX + 'bound-foreign', '["foreign","foreign"]'))
    _row(db, 'foreign-parent', user_id='foreign')
    _row(db, 'foreign-child', parent='foreign-parent', source='subagent')
    _row(db, 'owned-parent')
    _row(db, 'wrong-owner-child', parent='owned-parent', source='subagent', user_id='foreign')
    from tests.gateway.test_stored_picker_routed_sources import _messaging_row
    _messaging_row(db, 'messaging', '111')
    root = db.get_session('messaging')
    db.create_session('messaging-tip', source='telegram', model='m', user_id='111', parent_session_id='messaging')
    db._write_sql('UPDATE sessions SET session_key=?,chat_id=?,chat_type=?,origin_json=? WHERE id=?',
                 (root['session_key'], '111', 'dm', root['origin_json'], 'messaging-tip'))
    db.append_message('messaging-tip', 'user', 'tip first user')
    db._write_sql("UPDATE sessions SET end_reason='compression' WHERE id='messaging'")
    db.create_session('empty-cron', source='cron')
    _stable(db)
    expected = sorted([f'good-{i:04}' for i in range(count)] + ['owned-parent', 'messaging'] +
                      (['child', 'inherited'] if enabled else []), reverse=True)
    raw_calls, projected, raw_sql_counts = [], [], []
    read_all = db._read_all
    def observed_sql(query, *args, **kwargs):
        rows = read_all(query, *args, **kwargs)
        if 'ORDER BY _effective_last_active DESC' in query:
            raw_sql_counts.append(len(rows))
        return rows
    monkeypatch.setattr(db, '_read_all', observed_sql)
    rich, project = db.list_sessions_rich, listing._stored_picker_row
    def traced(**kwargs):
        rows = rich(**kwargs)
        assert len(rows) == raw_sql_counts[-1]
        raw_calls.append((kwargs['offset'], len(rows)))
        return rows
    def observed(*args):
        projected.append(args[2]['id'])
        return project(*args)
    monkeypatch.setattr(db, 'list_sessions_rich', traced)
    monkeypatch.setattr(listing, '_stored_picker_row', observed)
    before = _snapshot(db)
    for offset, limit in [(0, 200), (198, 1), (199, 2), (200, 200), (9007199254740991, 1)]:
        raw_calls.clear()
        result = (await _call(native, scope='stored', offset=offset, limit=limit))['result']
        assert [r['id'] for r in result['sessions']] == expected[offset:offset + limit]
        assert result['has_more'] == (len(expected) > offset + limit)
        assert raw_calls[0][0] == 0
        assert all(b[0] == a[0] + a[1] for a, b in pairwise(raw_calls))
        assert [c[0] for c in raw_calls[:2]] == [0, 200]
    assert 'z-denied-0199' in projected and 'foreign-profile' in projected
    assert _snapshot(db) == before


@pytest.mark.asyncio
@pytest.mark.parametrize('position', [199, 200, 201])
async def test_aliases_and_denied_lookahead_do_not_consume_ranks(tmp_path, monkeypatch, position):
    _, db, native = await _native(tmp_path, monkeypatch)
    for sid in ['good-a', 'good-b', 'denied']:
        _row(db, sid, user_id='foreign' if sid == 'denied' else None)
    real = db.list_sessions_rich(limit=200, include_subagents=True, compact_rows=True)
    rows = {r['id']: r for r in real}
    raw = [rows['denied']] * position + [rows['good-b'], rows['good-b'], rows['good-a']] + [rows['denied']] * 200
    calls = []
    def chunks(**kw):
        calls.append(kw['offset'])
        return raw[kw['offset']:kw['offset'] + kw['limit']]
    monkeypatch.setattr(db, 'list_sessions_rich', chunks)
    result = (await _call(native, scope='stored', offset=1, limit=1))['result']
    assert [r['id'] for r in result['sessions']] == ['good-a']
    assert result['has_more'] is False and calls == [0, 200, 400]


BAD = [('offset', x) for x in [-1, True, False, 1.5, '1', None, 9007199254740992]] + [
    ('limit', x) for x in [0, 201, True, False, '50', 1.5, None]]


@pytest.mark.asyncio
@pytest.mark.parametrize('key,value', BAD)
async def test_invalid_arguments_fail_before_query(tmp_path, monkeypatch, key, value):
    _, db, native = await _native(tmp_path, monkeypatch)
    before = _snapshot(db)
    def unexpected(**kwargs):
        pytest.fail('invalid request queried storage')
    monkeypatch.setattr(db, 'list_sessions_rich', unexpected)
    reply = await _call(native, scope='stored', **{key: value})
    assert reply['error']['code'] == 4001 and reply['error']['message'] == 'invalid_params'
    assert _snapshot(db) == before


@pytest.mark.asyncio
async def test_storage_error_is_not_partial_success(tmp_path, monkeypatch):
    _, db, native = await _native(tmp_path, monkeypatch)
    for i in range(201):
        _row(db, f'good-{i:04}')
    rich = db.list_sessions_rich
    def failing(**kwargs):
        if kwargs['offset']:
            raise sqlite3.OperationalError('second chunk unavailable')
        return rich(**kwargs)
    monkeypatch.setattr(db, 'list_sessions_rich', failing)
    before = _snapshot(db)
    reply = await _call(native, scope='stored')
    assert 'result' not in reply and reply['error']['message'] == 'storage_unavailable'
    assert _snapshot(db) == before


@pytest.mark.asyncio
async def test_title_and_live_paging(tmp_path, monkeypatch):
    _, db, native = await _native(tmp_path, monkeypatch)
    ids = []
    for i in range(10):
        _row(db, f'live-{i:02}')
        reply = await native.dispatch({'id': i, 'method': 'session.resume', 'params': {'session_id': f'live-{i:02}'}})
        assert 'result' in reply, reply
        db.reopen_session(f'live-{i:02}')
        ids.append(reply['result']['session_id'])
    _row(db, 'hidden-live')
    hidden = await native.dispatch({'id': 20, 'method': 'session.resume', 'params': {'session_id': 'hidden-live'}})
    assert 'result' in hidden
    db.set_session_hidden('hidden-live', True)
    _row(db, 'closed-live')
    closed = await native.dispatch({'id': 21, 'method': 'session.resume', 'params': {'session_id': 'closed-live'}})
    await native.dispatch({'id': 22, 'method': 'session.detach', 'params': {
        'session_id': 'closed-live', 'subscription_id': closed['result']['subscription_id'], 'end_reason': 'tui_close'}})
    from gateway.session_controls import AuthorityConnection
    foreign = AuthorityConnection(native.authority, object(), {'provider': 'local', 'user_id': 'other',
        'profile_id': native.authority.profile_id, 'instance_id': native.authority.instance_id})
    assert (await _call(foreign, offset=5, limit=3))['result']['sessions'] == []
    mismatched = AuthorityConnection(native.authority, object(), {'provider': 'local', 'user_id': 'native',
        'profile_id': 'foreign', 'instance_id': native.authority.instance_id})
    assert (await _call(mismatched, scope='stored'))['error']['message'] == 'profile_mismatch'
    assert (await _call(mismatched))['error']['message'] == 'profile_mismatch'
    _stable(db)
    result = (await _call(native, offset=5, limit=3))['result']
    assert result['scope'] == 'live'
    assert [r['id'] for r in result['sessions']] == sorted(ids, reverse=True)[5:8]
    assert result['has_more'] is True
    assert (await _call(native, offset=7, limit=3))['result']['has_more'] is False
    db.set_session_title(ids[0], 'My chat')
    title = (await _call(native, title='My chat', offset=0, limit=1))['result']
    assert title['offset'] == 0 and title['limit'] == 1 and title['has_more'] is False
    assert len(title['sessions']) == 1
    assert (await _call(native, title='My chat', offset=1))['error']['message'] == 'invalid_params'
