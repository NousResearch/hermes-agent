"""Pre-authority rows never grant ownership to the first remote claimant."""
import json
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
    # A code folder under focus: the row's saved cwd must not adopt it into the coding posture.
    (tmp_path / 'pyproject.toml').write_text('[project]\nname = "old"\n')
    config = {'platform_toolsets': {'cli': []}, 'agent': {'coding_context': 'focus'}}
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
    # Adoption is a mount: main's end stamp and source label stay until the first admitted turn.
    assert (db.get_session('old-cli')['ended_at'], db.get_session('old-cli')['end_reason']) == (original['ended_at'], 'cli_exit')
    assert db.get_session('old-cli')['source'] == 'cli'
    assert db.get_session('old-cli')['title'] == original['title']
    assert db.get_session('old-cli')['parent_session_id'] == original['parent_session_id']
    policy = runner.adapters[Platform.LOCAL].policies[ref.session_id]
    assert policy.model == 'old-model' and policy.cwd == str(tmp_path)
    assert not policy.coding_posture and 'coding' not in policy.toolsets
    config['platform_toolsets']['cli'] = ['terminal']
    authority.sessions.clear()
    await authority.attach(native, ref)
    assert runner.adapters[Platform.LOCAL].policies[ref.session_id] == policy
    history = [{'role': 'user', 'content': 'COMPRESSED'}, {'role': 'assistant', 'content': 'RETAINED'}]
    # Compression runs inside a turn, and the first admitted turn reopens the adopted row.
    from gateway.session_local_recovery import reopen_local_session
    reopen_local_session(authority, ref)
    assert db.get_session('old-cli')['end_reason'] is None
    assert db.try_acquire_compression_lock(ref.session_id, 'fixture')
    try:
        db.publish_compression_child(parent_session_id=ref.session_id, child_session_id='compressed',
            source='cli', messages=history, compression_lock_holder='fixture')
    finally:
        db.release_compression_lock(ref.session_id, 'fixture')
    snapshot = await authority.attach(native, ref)
    assert [m['content'] for m in snapshot.history if not m.get('archived')] == ['COMPRESSED', 'RETAINED']
    assert [m['content'] for m in snapshot.history if m.get('archived')] == ['OLD_QUESTION', 'OLD_ANSWER']
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
    assert (await _target(authority, submitter))[2].session_id == 'reset-tip'


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


@pytest.mark.asyncio
async def test_upgraded_desktop_history_resumes_by_id(tmp_path, monkeypatch):
    """main's Desktop stored its chats with ``source='desktop'``. After the upgrade the native owner
    adopts them by id like ``gui`` history, including the ones stamped with the signed-in Nous
    account. A served remote backend's login stamp (``hermes serve`` basic auth) adopts for that same
    login only. Any other identity still needs its own authority."""
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_controls import AuthorityConnection

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'platform_toolsets': {'gui': []}})
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    db = store._db
    for sid, user_id in (('old-desktop', None), ('old-desktop-account', 'nous:account'),
                         ('old-desktop-remote', 'basic:hermes')):
        db.create_session(sid, source='desktop', model='old-model', cwd=str(tmp_path), user_id=user_id)
        db.append_message(sid, role='user', content=sid.upper())
        db.end_session(sid, 'session_switch')
    runner = SimpleNamespace(session_store=store, _session_db=db, adapters={}, _draining=False,
                             _cached_agent_for=lambda route: None)
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    authority = await initialize_session_authority(runner, profile_id=str(tmp_path), instance_id='first')
    native = AuthorityConnection(authority, object(), {'provider': 'local', 'user_id': 'native',
        'profile_id': str(tmp_path), 'instance_id': 'first'})
    resumed = await native.dispatch({'id': 1, 'method': 'session.resume', 'params': {'session_id': 'old-desktop'}})
    assert resumed.get('result', {}).get('session_id') == 'old-desktop', resumed
    assert [m['content'] for m in resumed['result']['messages']] == ['OLD-DESKTOP']
    from gateway.config import Platform
    assert runner.adapters[Platform.LOCAL].policies['old-desktop'].platform == 'desktop'
    stamped = await native.dispatch({'id': 2, 'method': 'session.resume', 'params': {'session_id': 'old-desktop-account'}})
    assert stamped.get('result', {}).get('session_id') == 'old-desktop-account', stamped
    assert [m['content'] for m in stamped['result']['messages']] == ['OLD-DESKTOP-ACCOUNT']
    # A served remote backend stamped its basic-auth login on the row at creation (#139818). Hermes
    # is single-user and sign-in is owner-only, so that login is the machine owner: the owner's own
    # native ticket (native Desktop, the TUI) adopts it once its serve is gone (gateway.owner_identity).
    caps = ['session:create', 'session:read', 'session:submit', 'session:control', 'session:approve',
            'session:respond']
    from gateway.owner_identity import host_owner_subject
    owner = AuthorityConnection(authority, object(), {'provider': 'local', 'user_id': host_owner_subject(),
        'profile_id': str(tmp_path), 'instance_id': 'first', 'capabilities': caps})
    remote = await owner.dispatch({'id': 5, 'method': 'session.resume',
                                   'params': {'session_id': 'old-desktop-remote'}})
    assert remote.get('result', {}).get('session_id') == 'old-desktop-remote', remote
    assert [m['content'] for m in remote['result']['messages']] == ['OLD-DESKTOP-REMOTE']
    # Any other identity, or a Nous stamp on a non-Desktop row, still needs its own authority.
    for sid, source, user_id in (('old-desktop-foreign', 'desktop', 'telegram:42'),
                                 ('old-cli-account', 'cli', 'nous:account')):
        db.create_session(sid, source=source, model='old-model', cwd=str(tmp_path), user_id=user_id)
        db.append_message(sid, role='user', content=sid.upper())
        db.end_session(sid, 'session_switch')
        refused = await native.dispatch({'id': 3, 'method': 'session.resume', 'params': {'session_id': sid}})
        assert 'error' in refused, refused
        assert db.get_session(sid)['user_id'] == user_id


@pytest.mark.asyncio
async def test_native_host_adopts_row_stamped_by_a_served_login(tmp_path, monkeypatch):
    """#139818: after its ``hermes serve`` process is gone, an unrouted Desktop row stamped with a
    ``<provider>:<user>`` login is adoptable by the host's own native viewer WHEN a live registered
    provider in this home positively attests that exact user. The cross-login hole stays closed:
    a login the backend no longer serves (no registered provider / wrong user / messaging prefix)
    is still refused without a write."""
    import secrets

    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_controls import AuthorityConnection
    from gateway.session_local_migration import served_login_attests
    from hermes_cli.dashboard_auth.registry import register_provider, get_provider
    import plugins.dashboard_auth.basic as basic_plugin

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'platform_toolsets': {'gui': []}})
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    db = store._db

    rows = (
        ('served-row', 'basic:hermes'),      # the login this backend serves
        ('other-user', 'basic:someone'),     # a basic login that is NOT the configured user
        ('gone-login', 'oidc:carol'),        # an issuer-less label for an unregistered provider
        ('foreign-im', 'telegram:42'),       # a messaging platform's real foreign identity
    )
    for sid, uid in rows:
        db.create_session(sid, source='desktop', model='old-model', cwd=str(tmp_path), user_id=uid)
        db.append_message(sid, role='user', content=sid.upper())
        db.end_session(sid, 'session_switch')

    provider = basic_plugin.BasicAuthProvider(
        username='hermes', password_hash=basic_plugin.hash_password('pw'),
        secret=secrets.token_bytes(32))
    register_provider(provider, scope=str(tmp_path))
    try:
        # The registry lookup is scoped to this home and fails closed elsewhere.
        assert served_login_attests('basic:hermes', profile_home=str(tmp_path)) is True
        assert served_login_attests('basic:someone', profile_home=str(tmp_path)) is False
        assert served_login_attests('oidc:carol', profile_home=str(tmp_path)) is False
        assert served_login_attests('telegram:42', profile_home=str(tmp_path)) is False
        assert served_login_attests('basic:hermes', profile_home=str(tmp_path / 'other')) is False

        runner = SimpleNamespace(session_store=store, _session_db=db, adapters={}, _draining=False,
                                 _cached_agent_for=lambda route: None)
        runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
        authority = await initialize_session_authority(runner, profile_id=str(tmp_path),
                                                       instance_id='first')
        # The host's own native viewer: no login subject (provider == 'local' bootstrap identity).
        native = AuthorityConnection(authority, object(), {'provider': 'local', 'user_id': 'native',
            'profile_id': str(tmp_path), 'instance_id': 'first'})

        adopted = await native.dispatch({'id': 1, 'method': 'session.resume',
                                         'params': {'session_id': 'served-row'}})
        assert adopted.get('result', {}).get('session_id') == 'served-row', adopted
        bound = db.get_session('served-row')
        assert bound['session_key'] == 'agent:main:local:dm:served-row', bound
        # The row is rebound to the host native peer's authenticated subject (the namespaced local
        # bootstrap receipt), and its served-login stamp plus route columns are canonicalised.
        assert bound['user_id'] == native.actor.subject == 'auth:v1:["local","","native"]', bound
        assert bound['chat_id'] == 'served-row' and bound['origin_json'], bound

        # Every stamp the live provider cannot positively attest is still refused and untouched.
        for sid, uid in (('other-user', 'basic:someone'), ('gone-login', 'oidc:carol'),
                         ('foreign-im', 'telegram:42')):
            refused = await native.dispatch({'id': 2, 'method': 'session.resume',
                                             'params': {'session_id': sid}})
            assert refused.get('error', {}).get('message') == 'not_found', (sid, refused)
            untouched = db.get_session(sid)
            assert untouched['session_key'] is None and untouched['user_id'] == uid, (sid, untouched)
    finally:
        # The registry is process-global; drop the test provider from every slot so it cannot leak
        # to another test (the basic provider registered with a scope lives in the scoped overlay).
        from hermes_cli.dashboard_auth import registry as _registry
        with _registry._lock:
            _registry._providers.pop('basic', None)
            for overlay in list(_registry._scoped_providers.values()):
                overlay.pop('basic', None)
        assert get_provider('basic', scope=str(tmp_path)) is None


@pytest.mark.asyncio
async def test_native_host_can_delete_a_served_login_row_after_adoption(tmp_path, monkeypatch):
    """#139818 acceptance matrix cell 1 (delete leg): before adoption a bare ``basic:<user>`` row
    is refused by ``authorize_history`` with ``permission_denied`` (the host could not even clean
    the ghost row up); after the host native viewer's adoption commits, the row is native-owned and
    canonical-routed, so the same viewer's ``session.mutate delete`` retires it — no 403."""
    import secrets

    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_controls import AuthorityConnection
    from hermes_cli.dashboard_auth.registry import register_provider, get_provider
    from hermes_state_mutation_binding import authorize_history
    import plugins.dashboard_auth.basic as basic_plugin

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'platform_toolsets': {'gui': []}})
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    db = store._db
    db.create_session('served-row', source='desktop', model='old-model',
                      cwd=str(tmp_path), user_id='basic:hermes')
    db.append_message('served-row', role='user', content='HELLO')
    db.end_session('served-row', 'session_switch')

    provider = basic_plugin.BasicAuthProvider(
        username='hermes', password_hash=basic_plugin.hash_password('pw'),
        secret=secrets.token_bytes(32))
    register_provider(provider, scope=str(tmp_path))
    try:
        runner = SimpleNamespace(session_store=store, _session_db=db, adapters={}, _draining=False,
                                 _cached_agent_for=lambda route: None)
        runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
        authority = await initialize_session_authority(runner, profile_id=str(tmp_path),
                                                       instance_id='first')
        native = AuthorityConnection(authority, object(), {'provider': 'local', 'user_id': 'native',
            'profile_id': str(tmp_path), 'instance_id': 'first'})

        # Pre-adoption: the foreign-login stamp refuses the host viewer's delete at the authorizer.
        with db._read_ctx() as conn:
            with pytest.raises(Exception, match='permission_denied'):
                authorize_history(conn, native.actor, 'served-row', claim=True)

        adopted = await native.dispatch({'id': 1, 'method': 'session.resume',
                                         'params': {'session_id': 'served-row'}})
        assert adopted.get('result', {}).get('session_id') == 'served-row', adopted

        deleted = await native.dispatch({'id': 2, 'method': 'session.mutate', 'params': {
            'session_id': 'served-row', 'request_id': 'delete-served-row',
            'expected_revision': 0, 'expected_generation': 0,
            'operation': 'delete', 'payload': {}}})
        assert 'error' not in deleted, deleted
        assert deleted['result'].get('deleted_ids') == ['served-row'], deleted
        assert db.get_session('served-row') is None
    finally:
        from hermes_cli.dashboard_auth import registry as _registry
        with _registry._lock:
            _registry._providers.pop('basic', None)
            for overlay in list(_registry._scoped_providers.values()):
                overlay.pop('basic', None)
        assert get_provider('basic', scope=str(tmp_path)) is None
