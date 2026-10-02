"""GUI plumbing follows model routing only at an authorized idle resume boundary."""
import copy
import json
from types import SimpleNamespace

import pytest
import pytest_asyncio

from gateway.session_contract import Principal
from gateway.session_local import create_local_session
from gateway.session_local_plumbing import refresh_on_resume
from hermes_state_local import local_receipt
from hermes_state_runtime import RuntimeStoreError, admit_session_input, claim_session_input


@pytest_asyncio.fixture
async def plumbing(tmp_path, monkeypatch):
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway import run
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    config = {'model': {'default': 'before', 'provider': 'custom', 'base_url': 'http://127.0.0.1:1/v1', 'api_key': 'private-before'},
        'platform_toolsets': {'cli': []}, 'agent': {'system_prompt': 'frozen prompt', 'reasoning_effort': 'low'},
        'mcp_servers': {'old': {'command': 'frozen-mcp'}},
        'terminal': {'docker_env': {'KEPT': 'private-terminal'}}}
    monkeypatch.setattr(run, '_load_gateway_config', lambda *args: copy.deepcopy(config))
    monkeypatch.setattr(run, '_resolve_gateway_model', lambda cfg: cfg['model']['default'])
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    evictions = []
    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False,
                             _evict_cached_agent=lambda route: evictions.append(route))
    authority = await initialize_session_authority(runner, profile_id=str(tmp_path), instance_id='owner')
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)
    actor = Principal('human', authority.profile_id, frozenset({'session:create', 'session:read', 'session:control'}), 'viewer')
    try:
        yield SimpleNamespace(config=config, authority=authority, actor=actor, evictions=evictions, home=tmp_path)
    finally:
        store._db.close()


def create(p, **options):
    return create_local_session(p.authority, p.actor, {'request_id': 'room', 'source': 'gui', 'room_plumbing': True, **options})


@pytest.mark.asyncio
async def test_idle_refresh_changes_runtime_only_and_preserves_private_custody_and_prefix(plumbing):
    p = plumbing
    ref = create(p)
    p.authority.db.append_message(ref.session_id, 'user', 'retained history')
    p.authority.db._write_sql('UPDATE sessions SET system_prompt=? WHERE id=?', ('cached-prefix', ref.session_id))
    before = local_receipt(p.authority.db, ref.session_id)
    refresh_on_resume(p.authority, p.actor, ref)
    assert local_receipt(p.authority.db, ref.session_id) == before and not p.evictions
    p.config['model'] = dict(reversed(list(p.config['model'].items())))
    refresh_on_resume(p.authority, p.actor, ref)
    assert local_receipt(p.authority.db, ref.session_id) == before and not p.evictions
    p.config['model'].update(default='after', base_url='http://127.0.0.1:2/v1', api_key='private-after')
    p.config['platform_toolsets']['cli'] = ['terminal']
    p.config['agent'].update(system_prompt='must not replace prefix', reasoning_effort='high')
    p.config['mcp_servers'] = {'new': {'command': 'must-not-apply'}}
    p.authority._local_config_secrets.clear()  # cold recovery with the old runtime key rotated
    refresh_on_resume(p.authority, p.actor, ref)
    after = local_receipt(p.authority.db, ref.session_id)
    assert after['policy']['model'] == 'after'
    assert after['policy']['toolsets'] == before['policy']['toolsets']
    assert after['policy']['terminal_json'] == before['policy']['terminal_json']
    assert after['policy']['request_json'] == before['policy']['request_json']
    cfg = json.loads(after['policy']['config_json'])
    assert cfg['agent']['system_prompt'] == 'frozen prompt' and cfg['agent']['reasoning_effort'] == 'high'
    assert cfg['mcp_servers'] == json.loads(before['policy']['config_json'])['mcp_servers']
    encoded = json.dumps(after)
    assert 'private-before' not in encoded and 'private-after' not in encoded
    from gateway.session_policy import restore_policy
    assert restore_policy(after['policy']).config(p.authority)['model']['api_key'] == 'private-after'
    assert p.authority.db.get_session(ref.session_id)['system_prompt'] == 'cached-prefix'
    assert p.authority.db.get_messages_as_conversation(ref.session_id)[0]['content'] == 'retained history'
    assert len(p.evictions) == 1
    refresh_on_resume(p.authority, p.actor, ref)
    assert len(p.evictions) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('status', ['queued', 'started', 'unknown'])
async def test_pending_execution_keeps_original_policy_and_agent(plumbing, status):
    p = plumbing
    ref = create(p)
    admit_session_input(p.authority.db, epoch=p.authority.epoch, principal_id=p.actor.subject,
        session_id=ref.session_id, request_id='active', payload={'text': 'must keep policy'})
    if status != 'queued':
        row = claim_session_input(p.authority.db, epoch=p.authority.epoch, session_id=ref.session_id)
        if status == 'unknown':
            p.authority.db._write_sql("UPDATE session_admissions SET status='unknown' WHERE admission_id=?", (row['admission_id'],))
    before = local_receipt(p.authority.db, ref.session_id)
    p.config['model']['default'] = 'later'
    refresh_on_resume(p.authority, p.actor, ref)
    assert local_receipt(p.authority.db, ref.session_id) == before and not p.evictions


@pytest.mark.asyncio
async def test_read_only_foreign_and_ordinary_sessions_cannot_refresh(plumbing):
    p = plumbing
    ref = create(p)
    before = local_receipt(p.authority.db, ref.session_id)
    p.config['model']['default'] = 'later'
    reader = Principal(p.actor.subject, p.actor.profile_id, frozenset({'session:read'}), 'reader')
    refresh_on_resume(p.authority, reader, ref)
    assert local_receipt(p.authority.db, ref.session_id) == before
    foreign = Principal('other', p.actor.profile_id, p.actor.capabilities, 'foreign')
    with pytest.raises(RuntimeStoreError, match='permission_denied'):
        refresh_on_resume(p.authority, foreign, ref)
    ordinary = create_local_session(p.authority, p.actor, {'request_id': 'ordinary', 'source': 'gui'})
    frozen = local_receipt(p.authority.db, ordinary.session_id)
    p.config['model']['default'] = 'newest'
    refresh_on_resume(p.authority, p.actor, ordinary)
    assert local_receipt(p.authority.db, ordinary.session_id) == frozen and not p.evictions


@pytest.mark.asyncio
@pytest.mark.parametrize('options', [{'room_plumbing': 'true'}, {'source': 'cli'}, {'model': 'override'},
    {'toolsets': []}, {'ignore_rules': True}, {'yolo': True}])
async def test_room_plumbing_does_not_bypass_source_or_launch_policy(plumbing, options):
    with pytest.raises(RuntimeStoreError, match='invalid_params'):
        create(plumbing, **options)
    assert plumbing.authority.db.session_count() == 0


@pytest.mark.asyncio
async def test_marker_survives_cold_owner_and_legacy_gui_adoption(plumbing, monkeypatch):
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_local_title import resolve_titled_session
    from gateway.session_local_migration import bind_native_transport
    p = plumbing
    ref = create(p)
    p.authority.db.create_session('legacy-room', source='gui', model='obsolete-provider-model',
                                   model_config={'room_plumbing': True})
    p.authority.db.set_session_title('legacy-room', 'Group: old · thread')
    p.authority.db.set_session_hidden('legacy-room', True)
    p.authority.db.append_message('legacy-room', 'user', 'old room history')
    bind_native_transport(p.authority, p.actor, {'provider': 'local', 'profile_id': p.actor.profile_id,
                                               'instance_id': p.authority.instance_id})
    adopted = resolve_titled_session(p.authority, p.actor, 'Group: old · thread')
    old = local_receipt(p.authority.db, adopted.session_id)
    assert old['policy']['model'] == 'before'
    assert json.loads(old['policy']['request_json'])['room_plumbing'] is True
    store = SessionStore(p.home / 'sessions', GatewayConfig())
    evictions = []
    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False,
                             _evict_cached_agent=lambda route: evictions.append(route))
    try:
        p.config['model'].update(default='after-restart', api_key='rotated-after-restart')
        cold = await initialize_session_authority(runner, profile_id=p.actor.profile_id, instance_id='next-owner')
        for owned in (ref, adopted):
            refresh_on_resume(cold, p.actor, owned)
            saved = local_receipt(cold.db, owned.session_id)
            assert saved['policy']['model'] == 'after-restart'
            assert json.loads(saved['policy']['request_json'])['room_plumbing'] is True
        assert cold.db.get_messages_as_conversation(adopted.session_id)[0]['content'] == 'old room history'
        assert len(evictions) == 2
    finally:
        store._db.close()


@pytest.mark.asyncio
async def test_failed_policy_write_preserves_original_receipt_and_live_cache(plumbing, monkeypatch):
    p = plumbing
    ref = create(p)
    before = local_receipt(p.authority.db, ref.session_id)
    p.config['model']['default'] = 'not-committed'
    original = p.authority.db._execute_write
    def abort(callback, **kwargs):
        def write(conn):
            callback(conn)
            raise RuntimeError('transaction failed')
        return original(write, **kwargs)
    monkeypatch.setattr(p.authority.db, '_execute_write', abort)
    with pytest.raises(RuntimeError, match='transaction failed'):
        refresh_on_resume(p.authority, p.actor, ref)
    assert local_receipt(p.authority.db, ref.session_id) == before
    assert not p.evictions
