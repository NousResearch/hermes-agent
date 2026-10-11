"""Pre-authority rows never grant ownership to the first remote claimant."""
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.asyncio
async def test_legacy_history_adoption_is_native_owned_atomic_and_retryable(tmp_path, monkeypatch):
    from gateway import run
    from gateway.config import GatewayConfig, Platform
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_contract import Principal
    from gateway.session_local_title import resolve_titled_session
    from hermes_state_local import local_receipt
    from hermes_state_runtime import RuntimeStoreError

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    config = {'platform_toolsets': {'cli': []}}
    monkeypatch.setattr(run, '_load_gateway_config', lambda: config)
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    db = store._db
    # The old CLI persisted these columns, not a fabricated canonical receipt.
    db.create_session('old-cli', source='cli', model='old-model', cwd=str(tmp_path))
    db.set_session_title('old-cli', 'My retained work')
    db.append_message('old-cli', role='user', content='OLD_QUESTION')
    db.append_message('old-cli', role='assistant', content='OLD_ANSWER')
    db.end_session('old-cli', 'cli_exit')
    original = db.get_session('old-cli')
    runner = SimpleNamespace(session_store=store, _session_db=db, adapters={}, _draining=False)
    authority = await initialize_session_authority(runner, profile_id=str(tmp_path), instance_id='first')
    caps = frozenset({'session:create', 'session:read', 'session:submit', 'session:control'})
    remote = Principal('remote', str(tmp_path), caps, 'remote-socket')
    native = Principal('native', str(tmp_path), caps, 'native-socket')
    authority._native_legacy_transports = {native.transport_id: native}
    with pytest.raises(RuntimeStoreError, match='not_found'):
        resolve_titled_session(authority, remote, 'My retained work')
    from gateway.session_mutations import mutate_session
    from gateway.session_contract import SessionRef
    with pytest.raises(RuntimeStoreError, match='permission_denied'):
        await mutate_session(authority, remote, SessionRef(authority.profile_id, 'old-cli'),
            {'session_id': 'old-cli', 'request_id': 'steal', 'expected_revision': original['runtime_revision'],
             'operation': 'rename', 'payload': {'title': 'stolen'}})
    assert db.get_session('old-cli') == original
    ref = resolve_titled_session(authority, native, 'My retained work')
    assert ref == resolve_titled_session(authority, native, 'old-cli')
    snapshot = await authority.attach(native, ref)
    assert [m['content'] for m in snapshot.history] == ['OLD_QUESTION', 'OLD_ANSWER']
    receipt = local_receipt(db, ref.session_id)
    assert ref.session_id == 'old-cli'
    assert receipt['entry']['session_id'] == 'old-cli'
    assert db.get_session('old-cli')['ended_at'] is None
    assert db.get_session('old-cli')['title'] == original['title']
    assert db.get_session('old-cli')['parent_session_id'] == original['parent_session_id']
    policy = runner.adapters[Platform.LOCAL].policies[ref.session_id]
    assert policy.model == 'old-model' and policy.cwd == str(tmp_path)
    config['platform_toolsets']['cli'] = ['terminal']
    authority.sessions.clear()
    await authority.attach(native, ref)
    assert runner.adapters[Platform.LOCAL].policies[ref.session_id] == policy
    history = [{'role': 'user', 'content': 'COMPRESSED'}, {'role': 'assistant', 'content': 'RETAINED'}]
    assert db.try_acquire_compression_lock(ref.session_id, 'fixture')
    try:
        db.publish_compression_child(parent_session_id=ref.session_id, child_session_id='compressed',
            source='cli', messages=history, compression_lock_holder='fixture')
    finally:
        db.release_compression_lock(ref.session_id, 'fixture')
    snapshot = await authority.attach(native, ref)
    assert [m['content'] for m in snapshot.history] == ['COMPRESSED', 'RETAINED']
    assert resolve_titled_session(authority, native, 'old-cli') == ref
    assert local_receipt(db, ref.session_id)['policy'] == receipt['policy']
    with pytest.raises(RuntimeStoreError, match='permission_denied'):
        await authority.attach(remote, ref)
    foreign = Principal(native.subject, '/foreign', caps, native.transport_id)
    with pytest.raises(RuntimeStoreError, match='profile_mismatch'):
        await authority.attach(foreign, ref)
    db._write_sql('UPDATE state_meta SET value=? WHERE key=?', ('{}', 'gateway.local_policy.v1:' + ref.session_id))
    authority.sessions.clear()
    with pytest.raises(RuntimeStoreError, match='storage_unavailable'):
        resolve_titled_session(authority, native, 'old-cli')


@pytest.mark.asyncio
async def test_legacy_adoption_refuses_live_owner_and_foreign_profile_without_writes(tmp_path, monkeypatch):
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_contract import Principal
    from gateway.session_local_title import resolve_titled_session
    from hermes_cli.active_sessions import try_acquire_active_session, release_active_session
    from hermes_state_runtime import RuntimeStoreError

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'platform_toolsets': {'cli': []}})
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    db = store._db
    db.create_session('old-live', source='cli', model='old', cwd=str(tmp_path))
    db.create_session('foreign', source='cli', model='old', cwd=str(tmp_path), profile_name='other')
    runner = SimpleNamespace(session_store=store, _session_db=db, adapters={}, _draining=False)
    authority = await initialize_session_authority(runner, profile_id=str(tmp_path), instance_id='first')
    native = Principal('native', str(tmp_path), frozenset({'session:read', 'session:create'}), 'socket')
    authority._native_legacy_transports = {native.transport_id: native}
    lease, refusal = try_acquire_active_session(session_id='old-live', surface='cli', registry_home=Path(db.db_path).parent, config={})
    assert lease is not None, refusal
    from hermes_cli.active_sessions import active_session_liveness_guard
    with active_session_liveness_guard('old-live', registry_home=Path(db.db_path).parent) as active:
        assert active
    try:
        with pytest.raises(RuntimeStoreError, match='runtime_coordination_required'):
            resolve_titled_session(authority, native, 'old-live')
        with pytest.raises(RuntimeStoreError, match='not_found'):
            resolve_titled_session(authority, native, 'foreign')
        with db._read_ctx() as conn:
            assert not conn.execute("SELECT 1 FROM state_meta WHERE key LIKE 'gateway.local_policy.v1:%'").fetchall()
    finally:
        release_active_session(lease)
    ref = resolve_titled_session(authority, native, 'old-live')
    db.set_session_title(ref.session_id, 'Bot Chat')
    from hermes_state_local_lineage import reset_local_target
    from hermes_state_local import local_receipt
    receipt = local_receipt(db, ref.session_id)
    entry = dict(receipt['entry'], session_id='reset-tip')
    reset_local_target(db, epoch=authority.epoch, parent_session_id=ref.session_id, entry=entry)
    from gateway.session_bot import _target
    submitter = Principal(native.subject, native.profile_id, native.capabilities | {'session:submit'}, native.transport_id)
    assert _target(authority, submitter)[2].session_id == 'reset-tip'


@pytest.mark.asyncio
async def test_upgraded_oneshot_and_tool_history_resumes_by_id_and_title_but_stays_out_of_pickers(tmp_path, monkeypatch):
    """main's `chat -q` stored ``oneshot`` rows (``--source tool`` stored ``tool``) and printed
    `hermes --resume <id>` for each. After the upgrade, the native owner adopts them by id and by
    title like ``cli`` history, and they keep their label so human pickers still hide them."""
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_controls import AuthorityConnection

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'platform_toolsets': {'cli': []}})
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    db = store._db
    for sid, source in (('old-oneshot', 'oneshot'), ('old-tool', 'tool')):
        db.create_session(sid, source=source, model='old-model', cwd=str(tmp_path))
        db.set_session_title(sid, sid + ' title')
        db.append_message(sid, role='user', content=sid.upper())
        db.end_session(sid, 'cli_exit')
    runner = SimpleNamespace(session_store=store, _session_db=db, adapters={}, _draining=False,
                             _cached_agent_for=lambda route: None)
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    authority = await initialize_session_authority(runner, profile_id=str(tmp_path), instance_id='first')
    native = AuthorityConnection(authority, object(), {'provider': 'local', 'user_id': 'native',
        'profile_id': str(tmp_path), 'instance_id': 'first'})
    by_id = await native.dispatch({'id': 1, 'method': 'session.resume', 'params': {'session_id': 'old-oneshot'}})
    assert by_id.get('result', {}).get('session_id') == 'old-oneshot', by_id
    assert [m['content'] for m in by_id['result']['messages']] == ['OLD-ONESHOT']
    by_title = await native.dispatch({'id': 2, 'method': 'session.resume', 'params': {'title': 'old-tool title'}})
    assert by_title['result']['session_id'] == 'old-tool', by_title
    # Every turn refreshes the routing peer through the transport (``local``); the adopted label survives it.
    source = authority.sessions['old-oneshot'].source
    store._record_gateway_session_peer('old-oneshot', authority.sessions['old-oneshot'].route, source)
    assert {db.get_session(s)['source'] for s in ('old-oneshot', 'old-tool')} == {'oneshot', 'tool'}
    listed = await native.dispatch({'id': 3, 'method': 'session.list', 'params': {}})
    assert listed['result']['sessions'] == [], listed
