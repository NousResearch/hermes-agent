"""Real authority/SQLite regressions for guarded delegated history."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.gateway.test_stored_picker_routed_sources import _owner


async def _native(tmp_path, monkeypatch, *, enabled=True):
    from gateway.session_controls import AuthorityConnection
    authority, db = await _owner(tmp_path, monkeypatch)
    home = Path(db.db_path).parent
    home.joinpath('config.yaml').write_text(
        '{}\n' if enabled is None else 'sessions:\n  show_subagents: ' + str(enabled).lower() + '\n')
    native = AuthorityConnection(authority, object(), {'provider': 'local', 'user_id': 'native',
        'profile_id': authority.profile_id, 'instance_id': authority.instance_id}, operator=True)
    assert authority._native_legacy_transports[native.actor.transport_id] == native.actor
    assert {'session:read', 'session:create', 'session:control'} <= set(native.actor.capabilities)
    return authority, db, native


def _row(db, sid, *, parent=None, source='cli', **kwargs):
    import os
    from unittest.mock import patch
    from agent.session_source import session_source_for
    # Exercise the real source resolver, independent of the test runner's launch source.
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop('HERMES_SESSION_SOURCE', None)
        if parent and source != 'subagent':
            os.environ['HERMES_SESSION_SOURCE'] = source
        actual_source = session_source_for('subagent') if parent else source
    assert actual_source == source
    db.create_session(sid, source=actual_source, model='m', parent_session_id=parent,
        model_config={'_delegate_from': parent} if parent else None, **kwargs)
    db.append_message(sid, 'user', sid + ' question')
    db.append_message(sid, 'assistant', sid + ' answer')
    db.end_session(sid, 'completed')


def _snapshot(db):
    with db._read_ctx() as conn:
        return {table: [tuple(row) for row in conn.execute('SELECT * FROM ' + table)]
                for table in ('sessions', 'messages', 'state_meta', 'session_admissions', 'worker_executions',
                              'session_turn_leases', 'gateway_routing')}


async def _list(native):
    reply = await native.dispatch({'id': 1, 'method': 'session.list', 'params': {'scope': 'stored'}})
    assert 'result' in reply, reply
    return reply['result']['sessions']


@pytest.mark.asyncio
@pytest.mark.parametrize('enabled', [None, False, True])
async def test_profile_setting_lists_literal_and_inherited_children_read_only(tmp_path, monkeypatch, enabled):
    authority, db, native = await _native(tmp_path, monkeypatch, enabled=enabled)
    _row(db, 'parent')
    _row(db, 'literal', parent='parent', source='subagent')
    _row(db, 'inherited', parent='parent')
    before = _snapshot(db)
    rows = await _list(native)
    ids = [row['id'] for row in rows]
    assert ids.count('parent') == 1
    for sid in ('literal', 'inherited'):
        assert ids.count(sid) == int(bool(enabled))
    assert _snapshot(db) == before
    assert not authority.runner.adapters


@pytest.mark.parametrize('path', [None, '', ':memory:', 'file::memory:?cache=shared', 42])
def test_invalid_db_paths_never_read_config(path, monkeypatch):
    from gateway.session_listing import _show_subagents
    import hermes_cli.session_listing as listing
    def unexpected(*args):
        pytest.fail('non-file DB must not read profile config')
    monkeypatch.setattr(listing, 'show_subagent_sessions', unexpected)
    assert _show_subagents(SimpleNamespace(db_path=path)) is False


@pytest.mark.asyncio
@pytest.mark.parametrize('source', ['subagent', 'cli'])
async def test_resume_child_preserves_provenance_and_end_stamp_on_cold_restore(tmp_path, monkeypatch, source):
    from gateway.config import Platform
    from hermes_state_local import local_receipt
    authority, db, native = await _native(tmp_path, monkeypatch)
    _row(db, 'parent')
    _row(db, 'child', parent='parent', source=source)
    parent, child = db.get_session('parent'), db.get_session('child')
    before_messages = db.get_messages('child')
    for attempt in range(2):
        if attempt:
            previous, db_path = authority, db.db_path
            db.close()
            authority, db, native = await _native(tmp_path, monkeypatch)
            assert authority is not previous
            assert authority.runner is not previous.runner
            assert db.db_path == db_path
        listed = next(row for row in await _list(native) if row['id'] == 'child')
        reply = await native.dispatch({'id': 2, 'method': 'session.resume',
                                       'params': {'session_id': listed['session_id']}})
        assert 'result' in reply, reply
        assert reply['result']['session_id'] == 'child'
        assert [m['content'] for m in reply['result']['messages']] == ['child question', 'child answer']
        saved = db.get_session('child')
        for key in ('source', 'ended_at', 'end_reason', 'parent_session_id', 'model_config'):
            assert saved[key] == child[key]
        assert db.get_session('parent') == parent
        assert db.get_messages('child') == before_messages
        receipt = local_receipt(db, 'child')
        assert receipt['policy']['source'] == 'cli'
        assert not authority.runner.adapters[Platform.LOCAL].policies['child'].model == ''
        assert 'child' in {row['id'] for row in await _list(native)}
    with db._read_ctx() as conn:
        assert not conn.execute('SELECT 1 FROM session_admissions').fetchone()
        assert not conn.execute('SELECT 1 FROM worker_executions').fetchone()


@pytest.mark.asyncio
@pytest.mark.parametrize('denial', ['owner', 'profile', 'binding', 'missing', 'self', 'marker', 'routed', 'lease', 'running'])
async def test_invalid_ancestry_is_hidden_and_resume_refuses_without_writes(tmp_path, monkeypatch, denial):
    import time
    from hermes_state_mutation_binding import BINDING_PREFIX
    authority, db, native = await _native(tmp_path, monkeypatch)
    _row(db, 'parent')
    _row(db, 'child', parent='parent', source='subagent')
    now = time.time()
    faults = {
        'owner': ('UPDATE sessions SET user_id=? WHERE id=?', ('foreign', 'parent')),
        'profile': ('UPDATE sessions SET profile_name=? WHERE id=?', ('foreign', 'parent')),
        'binding': ('INSERT INTO state_meta(key,value) VALUES(?,?)',
                    (BINDING_PREFIX + 'parent', json.dumps([authority.profile_id, 'foreign']))),
        'missing': ('UPDATE sessions SET parent_session_id=?,model_config=? WHERE id=?',
                    (None, json.dumps({'_delegate_from': 'absent'}), 'child')),
        'self': ('UPDATE sessions SET parent_session_id=?,model_config=? WHERE id=?',
                 ('child', json.dumps({'_delegate_from': 'child'}), 'child')),
        'marker': ('UPDATE sessions SET parent_session_id=?,model_config=? WHERE id=?',
                   (None, json.dumps({'_delegate_from': ''}), 'child')),
        'routed': ('UPDATE sessions SET session_key=?,chat_id=?,origin_json=? WHERE id=?',
                   ('foreign-route', 'foreign-chat', '{}', 'child')),
        'lease': ('INSERT INTO session_turn_leases(conversation_id,holder,acquired_at,expires_at) VALUES(?,?,?,?)',
                  ('parent', 'live', now, now + 3600)),
        'running': ('UPDATE sessions SET ended_at=NULL WHERE id=?', ('child',)),
    }
    db._write_sql(*faults[denial])
    # Even a previously saved child binding cannot bypass ancestry validation.
    db._write_sql('INSERT INTO state_meta(key,value) VALUES(?,?)',
        (BINDING_PREFIX + 'child', json.dumps([authority.profile_id, native.actor.subject], separators=(',', ':'))))
    before = _snapshot(db)
    assert 'child' not in {row['id'] for row in await _list(native)}
    reply = await native.dispatch({'id': 2, 'method': 'session.resume', 'params': {'session_id': 'child'}})
    assert 'error' in reply, reply
    assert _snapshot(db) == before


@pytest.mark.asyncio
@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('include_subagents', [False, True])
async def test_client_cannot_override_server_subagent_setting(tmp_path, monkeypatch, enabled, include_subagents):
    _, db, native = await _native(tmp_path, monkeypatch, enabled=enabled)
    _row(db, 'parent')
    _row(db, 'child', parent='parent', source='subagent')
    before = _snapshot(db)
    reply = await native.dispatch({'id': 3, 'method': 'session.list',
        'params': {'scope': 'stored', 'include_subagents': include_subagents}})
    assert reply['error']['code'] == 4001, reply
    assert reply['error']['message'] == 'invalid_params', reply
    assert _snapshot(db) == before


async def _refused_without_mutation(authority, db, native, sid='child'):
    from unittest.mock import Mock
    before = _snapshot(db)
    sessions = dict(authority.sessions)
    adapters = dict(authority.runner.adapters)
    schedule = Mock(side_effect=AssertionError('a rejected resume must not schedule a turn'))
    authority._schedule = schedule
    listed = await native.dispatch({'id': 1, 'method': 'session.list', 'params': {'scope': 'stored'}})
    if native.actor.profile_id != authority.profile_id:
        assert listed['error']['message'] == 'profile_mismatch', listed
    else:
        assert sid not in {row['id'] for row in listed['result']['sessions']}
    reply = await native.dispatch({'id': 2, 'method': 'session.resume', 'params': {'session_id': sid}})
    assert reply['error']['message'] in {
        'not_found', 'permission_denied', 'storage_unavailable', 'runtime_coordination_required',
        'profile_mismatch'}, reply
    assert _snapshot(db) == before
    assert authority.sessions == sessions
    assert authority.runner.adapters == adapters
    schedule.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize('target', ['child', 'parent'])
@pytest.mark.parametrize('fault', ['owner', 'profile', 'binding_owner', 'binding_profile', 'binding_json'])
async def test_foreign_lineage_cannot_bypass_saved_child_binding(tmp_path, monkeypatch, target, fault):
    from hermes_state_mutation_binding import BINDING_PREFIX
    authority, db, native = await _native(tmp_path, monkeypatch)
    _row(db, 'parent')
    _row(db, 'child', parent='parent', source='subagent')
    binding = [authority.profile_id, native.actor.subject]
    db._write_sql('INSERT INTO state_meta(key,value) VALUES(?,?)',
                  (BINDING_PREFIX + 'child', json.dumps(binding)))
    if fault in ('owner', 'profile'):
        field = 'user_id' if fault == 'owner' else 'profile_name'
        db._write_sql('UPDATE sessions SET ' + field + '=? WHERE id=?', ('foreign', target))
    else:
        value = ('{' if fault == 'binding_json' else
                 json.dumps(['foreign', binding[1]] if fault == 'binding_profile' else [binding[0], 'foreign']))
        db._write_sql('INSERT OR REPLACE INTO state_meta(key,value) VALUES(?,?)',
                      (BINDING_PREFIX + target, value))
    await _refused_without_mutation(authority, db, native)


@pytest.mark.asyncio
async def test_foreign_actor_profile_cannot_claim_delegated_history(tmp_path, monkeypatch):
    from gateway.session_controls import AuthorityConnection
    authority, db, _ = await _native(tmp_path, monkeypatch)
    _row(db, 'parent')
    _row(db, 'child', parent='parent', source='subagent')
    foreign = AuthorityConnection(authority, object(), {'provider': 'local', 'user_id': 'native',
        'profile_id': 'foreign', 'instance_id': authority.instance_id}, operator=True)
    await _refused_without_mutation(authority, db, foreign)


@pytest.mark.asyncio
@pytest.mark.parametrize('fault', ['cycle', 'marker_list', 'marker_number', 'marker_object', 'tool', 'oneshot'])
async def test_cycles_malformed_markers_and_internal_sources_are_denied(tmp_path, monkeypatch, fault):
    authority, db, native = await _native(tmp_path, monkeypatch)
    _row(db, 'parent')
    _row(db, 'child', parent='parent', source='subagent')
    if fault == 'cycle':
        _row(db, 'middle', parent='child', source='subagent')
        db._write_sql('UPDATE sessions SET parent_session_id=?,model_config=? WHERE id=?',
                      ('middle', json.dumps({'_delegate_from': 'middle'}), 'parent'))
    elif fault.startswith('marker_'):
        marker = {'marker_list': ['parent'], 'marker_number': 42, 'marker_object': {'id': 'parent'}}[fault]
        db._write_sql('UPDATE sessions SET model_config=? WHERE id=?',
                      (json.dumps({'_delegate_from': marker}), 'child'))
    else:
        db._write_sql('UPDATE sessions SET source=? WHERE id=?', (fault, 'child'))
    await _refused_without_mutation(authority, db, native)


@pytest.mark.asyncio
@pytest.mark.parametrize('hops', [32, 33])
async def test_delegation_ancestry_hop_boundary(tmp_path, monkeypatch, hops):
    authority, db, native = await _native(tmp_path, monkeypatch)
    _row(db, 'root')
    parent = 'root'
    for hop in range(hops):
        sid = 'child' if hop == hops - 1 else f'ancestor-{hop}'
        _row(db, sid, parent=parent, source='subagent')
        parent = sid
    if hops == 33:
        await _refused_without_mutation(authority, db, native)
    else:
        before = _snapshot(db)
        assert 'child' in {row['id'] for row in await _list(native)}
        assert _snapshot(db) == before
        reply = await native.dispatch({'id': 2, 'method': 'session.resume', 'params': {'session_id': 'child'}})
        assert reply['result']['session_id'] == 'child', reply
        assert db.get_session('child')['ended_at'] is not None


@pytest.mark.asyncio
@pytest.mark.parametrize('target', ['child', 'parent'])
@pytest.mark.parametrize('kind,status', [
    ('admission', 'queued'), ('admission', 'started'), ('admission', 'unknown'),
    ('worker', 'registered'), ('worker', 'running'), ('worker', 'unknown')])
async def test_active_runtime_states_block_list_and_resume(tmp_path, monkeypatch, target, kind, status):
    authority, db, native = await _native(tmp_path, monkeypatch)
    _row(db, 'parent')
    _row(db, 'child', parent='parent', source='subagent')
    if kind == 'admission':
        db._write_sql("""INSERT INTO session_admissions(admission_id,request_id,principal_id,
            target_session_id,lineage_json,payload_json,payload_digest,intent,status,owner_epoch,generation)
            VALUES(?,?,?,?,?,?,?,'queue',?,?,?)""",
            ('admission', 'request', native.actor.subject, target, '[]', '{}', 'digest', status, authority.epoch, 1))
    else:
        db._write_sql("""INSERT INTO worker_executions(execution_id,session_id,kind,owner_epoch,
            generation,status,adoption_digest) VALUES(?,?,'child',?,?,?,?)""",
            ('execution', target, authority.epoch, 1, status, 'digest'))
    await _refused_without_mutation(authority, db, native)


@pytest.mark.asyncio
@pytest.mark.parametrize('target', ['child', 'parent'])
@pytest.mark.parametrize('state', ['alive', 'unknown', 'malformed', 'stale'])
async def test_active_registry_is_read_only_and_fail_closed(tmp_path, monkeypatch, target, state):
    import os
    from hermes_cli import active_sessions
    authority, db, native = await _native(tmp_path, monkeypatch)
    _row(db, 'parent')
    _row(db, 'child', parent='parent', source='subagent')
    path = Path(db.db_path).parent / 'runtime' / 'active_sessions.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {'lease_id': 'test', 'session_id': target, 'pid': os.getpid()}
    if state == 'unknown':
        entry['process_start_time'] = 1.0
        monkeypatch.setattr(active_sessions, '_own_start_time', lambda: None)
    elif state == 'stale':
        entry['process_start_time'] = 1.0  # A mismatched process identity, not this live writer.
    path.write_text('{' if state == 'malformed' else json.dumps({'entries': [entry]}))
    before = path.read_bytes()
    if state == 'stale':
        snapshot = _snapshot(db)
        assert 'child' in {row['id'] for row in await _list(native)}
        assert _snapshot(db) == snapshot
    else:
        await _refused_without_mutation(authority, db, native)
    assert path.read_bytes() == before


@pytest.mark.asyncio
@pytest.mark.parametrize('receipt', ['policy', 'legacy'])
@pytest.mark.parametrize('fault', ['json', 'shape', 'owner', 'profile', 'route', 'missing'])
async def test_malformed_adopted_receipts_cannot_resume(tmp_path, monkeypatch, receipt, fault):
    from hermes_state_local import POLICY_PREFIX
    from hermes_state_local_migration import LEGACY_PREFIX
    authority, db, native = await _native(tmp_path, monkeypatch)
    _row(db, 'parent')
    _row(db, 'child', parent='parent', source='subagent')
    reply = await native.dispatch({'id': 1, 'method': 'session.resume', 'params': {'session_id': 'child'}})
    assert 'result' in reply, reply
    key = (POLICY_PREFIX if receipt == 'policy' else LEGACY_PREFIX) + 'child'
    with db._read_ctx() as conn:
        saved = json.loads(conn.execute('SELECT value FROM state_meta WHERE key=?', (key,)).fetchone()[0])
    if fault == 'missing':
        db._write_sql('DELETE FROM state_meta WHERE key=?', (key,))
    else:
        if fault in ('owner', 'profile', 'route'):
            saved[{'owner': 'principal_id', 'profile': 'profile_id', 'route': 'route'}[fault]] = 'foreign'
        value = '{' if fault == 'json' else '[]' if fault == 'shape' else json.dumps(saved)
        db._write_sql('UPDATE state_meta SET value=? WHERE key=?', (value, key))
    authority.sessions.clear()
    await _refused_without_mutation(authority, db, native)


@pytest.mark.asyncio
@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('include_subagents', [False, True])
async def test_legacy_dispatch_rejects_subagent_override(tmp_path, monkeypatch, enabled, include_subagents):
    from tui_gateway import server
    _, db, _ = await _native(tmp_path, monkeypatch, enabled=enabled)
    _row(db, 'parent')
    _row(db, 'child', parent='parent', source='subagent')
    before = _snapshot(db)
    reply = server.handle_request({'id': 3, 'method': 'session.list',
                                   'params': {'include_subagents': include_subagents}})
    assert reply['error']['code'] == 4000, reply
    assert 'include_subagents' in reply['error']['message'], reply
    assert _snapshot(db) == before
