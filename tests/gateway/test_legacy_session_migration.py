"""Pre-authority rows never grant ownership to the first remote claimant."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from persistence.conversation.lineage import CompressionChild
from persistence.conversation.messages import NewMessage
from persistence_assembly import store_location
from tests.fixtures.storage_holder import policy_key, raw_holder
from tests.fixtures.storage_read import session_row
from tests.fixtures.storage_seed import seed_messages, seed_session


@pytest.mark.asyncio
async def test_legacy_history_adoption_is_native_owned_atomic_and_retryable(tmp_path, monkeypatch):
    from gateway import run
    from gateway.config import GatewayConfig, Platform
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_contract import SessionActor
    from gateway.session_local_title import resolve_titled_session
    from persistence.core.errors import AuthorityRefusal

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    # A code folder under focus: the row's saved cwd must not adopt it into the coding posture.
    (tmp_path / 'pyproject.toml').write_text('[project]\nname = "old"\n')
    config = {'platform_toolsets': {'cli': []}, 'agent': {'coding_context': 'focus'}}
    monkeypatch.setattr(run, '_load_gateway_config', lambda: config)
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    db = store.persistence
    # The old CLI persisted these columns, not a fabricated canonical receipt.
    seed_session(db, 'old-cli', 'cli', model='old-model', cwd=str(tmp_path), title='My retained work')
    seed_messages(db, 'old-cli', {'role': 'user', 'content': 'OLD_QUESTION'},
                  {'role': 'assistant', 'content': 'OLD_ANSWER'})
    db.sessions.end('old-cli', 'cli_exit')
    original = session_row(db, 'old-cli')
    runner = SimpleNamespace(session_store=store, _session_db=db, adapters={}, _draining=False)
    authority = await initialize_session_authority(runner, profile_id=str(tmp_path), instance_id='first')
    caps = frozenset({'session:create', 'session:read', 'session:submit', 'session:control'})
    remote = SessionActor('remote', str(tmp_path), caps, 'remote-socket')
    native = SessionActor('native', str(tmp_path), caps, 'native-socket')
    authority._native_legacy_transports = {native.transport_id: native}
    with pytest.raises(AuthorityRefusal, match='not_found'):
        resolve_titled_session(authority, remote, 'My retained work')
    from gateway.session_mutations import mutate_session
    from gateway.session_contract import SessionRef
    with pytest.raises(AuthorityRefusal, match='permission_denied'):
        await mutate_session(authority, remote, SessionRef(authority.profile_id, 'old-cli'),
            {'session_id': 'old-cli', 'request_id': 'steal', 'expected_revision': original['runtime_revision'],
             'operation': 'rename', 'payload': {'title': 'stolen'}})
    assert session_row(db, 'old-cli') == original
    ref = resolve_titled_session(authority, native, 'My retained work')
    assert ref == resolve_titled_session(authority, native, 'old-cli')
    snapshot = await authority.attach(native, ref)
    assert [m['content'] for m in snapshot.history] == ['OLD_QUESTION', 'OLD_ANSWER']
    receipt = db.bindings.receipt(ref.session_id)
    assert ref.session_id == 'old-cli'
    assert receipt.entry['session_id'] == 'old-cli'
    # Adoption is a mount: main's end stamp and source label stay until the first admitted turn.
    assert (session_row(db, 'old-cli')['ended_at'], session_row(db, 'old-cli')['end_reason']) == (original['ended_at'], 'cli_exit')
    assert session_row(db, 'old-cli')['source'] == 'cli'
    assert session_row(db, 'old-cli')['title'] == original['title']
    assert session_row(db, 'old-cli')['parent_session_id'] == original['parent_session_id']
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
    assert session_row(db, 'old-cli')['end_reason'] is None
    assert db.coordination.try_acquire_compression_lock(session_id=ref.session_id, holder='fixture')
    try:
        db.lineage.publish_compression_child(ref.session_id, CompressionChild(session_id='compressed', source='cli'),
            [NewMessage.from_live(m) for m in history], compression_lock_holder='fixture')
    finally:
        db.coordination.release_compression_lock(session_id=ref.session_id, holder='fixture')
    snapshot = await authority.attach(native, ref)
    assert [m['content'] for m in snapshot.history if not m.get('archived')] == ['COMPRESSED', 'RETAINED']
    assert [m['content'] for m in snapshot.history if m.get('archived')] == ['OLD_QUESTION', 'OLD_ANSWER']
    assert resolve_titled_session(authority, native, 'old-cli') == ref
    assert db.bindings.receipt(ref.session_id).policy == receipt.policy
    with pytest.raises(AuthorityRefusal, match='permission_denied'):
        await authority.attach(remote, ref)
    foreign = SessionActor(native.subject, '/foreign', caps, native.transport_id)
    with pytest.raises(AuthorityRefusal, match='profile_mismatch'):
        await authority.attach(foreign, ref)
    raw_holder(db)._execute_write(lambda conn: conn.execute(
        'UPDATE state_meta SET value=? WHERE key=?', ('{}', policy_key(ref.session_id))))
    authority.sessions.clear()
    with pytest.raises(AuthorityRefusal, match='storage_unavailable'):
        resolve_titled_session(authority, native, 'old-cli')


@pytest.mark.asyncio
async def test_legacy_adoption_refuses_live_owner_and_foreign_profile_without_writes(tmp_path, monkeypatch):
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_contract import SessionActor
    from gateway.session_local_title import resolve_titled_session
    from hermes_cli.active_sessions import try_acquire_active_session, release_active_session
    from persistence.core.errors import AuthorityRefusal

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'platform_toolsets': {'cli': []}})
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    db = store.persistence
    seed_session(db, 'old-live', 'cli', model='old', cwd=str(tmp_path))
    seed_session(db, 'foreign', 'cli', model='old', cwd=str(tmp_path), profile_name='other')
    runner = SimpleNamespace(session_store=store, _session_db=db, adapters={}, _draining=False)
    authority = await initialize_session_authority(runner, profile_id=str(tmp_path), instance_id='first')
    native = SessionActor('native', str(tmp_path), frozenset({'session:read', 'session:create'}), 'socket')
    authority._native_legacy_transports = {native.transport_id: native}
    lease, refusal = try_acquire_active_session(session_id='old-live', surface='cli', registry_home=Path(store_location(db.scope)).parent, config={})
    assert lease is not None, refusal
    from hermes_cli.active_sessions import active_session_liveness_guard
    with active_session_liveness_guard('old-live', registry_home=Path(store_location(db.scope)).parent) as active:
        assert active
    try:
        with pytest.raises(AuthorityRefusal, match='runtime_coordination_required'):
            resolve_titled_session(authority, native, 'old-live')
        with pytest.raises(AuthorityRefusal, match='not_found'):
            resolve_titled_session(authority, native, 'foreign')
        with raw_holder(db)._read_ctx() as conn:
            assert not conn.execute("SELECT 1 FROM state_meta WHERE key LIKE ?", (policy_key() + '%',)).fetchall()
    finally:
        release_active_session(lease)
    ref = resolve_titled_session(authority, native, 'old-live')
    db.sessions.set_title(ref.session_id, 'Bot Chat')
    receipt = db.bindings.receipt(ref.session_id)
    entry = dict(receipt.entry, session_id='reset-tip')
    db.bindings.reset_target(epoch=authority.epoch, parent_session_id=ref.session_id, entry=entry)
    from gateway.session_bot import _target
    submitter = SessionActor(native.subject, native.profile_id, native.capabilities | {'session:submit'}, native.transport_id)
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
    db = store.persistence
    for sid, source in (('old-oneshot', 'oneshot'), ('old-tool', 'tool')):
        seed_session(db, sid, source, model='old-model', cwd=str(tmp_path), title=sid + ' title')
        seed_messages(db, sid, {'role': 'user', 'content': sid.upper()})
        db.sessions.end(sid, 'cli_exit')
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
    assert {session_row(db, s)['source'] for s in ('old-oneshot', 'old-tool')} == {'oneshot', 'tool'}
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
    db = store.persistence
    for sid, user_id in (('old-desktop', None), ('old-desktop-account', 'nous:account'),
                         ('old-desktop-remote', 'basic:hermes')):
        seed_session(db, sid, 'desktop', model='old-model', cwd=str(tmp_path), user_id=user_id)
        seed_messages(db, sid, {'role': 'user', 'content': sid.upper()})
        db.sessions.end(sid, 'session_switch')
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
        seed_session(db, sid, source, model='old-model', cwd=str(tmp_path), user_id=user_id)
        seed_messages(db, sid, {'role': 'user', 'content': sid.upper()})
        db.sessions.end(sid, 'session_switch')
        refused = await native.dispatch({'id': 3, 'method': 'session.resume', 'params': {'session_id': sid}})
        assert 'error' in refused, refused
        assert session_row(db, sid)['user_id'] == user_id
