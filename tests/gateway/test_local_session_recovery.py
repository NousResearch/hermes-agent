"""Local creation is durable before publication and restart never replays a claim."""
import json
from types import SimpleNamespace

import pytest


@pytest.mark.asyncio
async def test_local_create_receipt_survives_cold_authority_atomically(tmp_path, monkeypatch):
    from gateway.config import GatewayConfig, Platform
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_contract import Principal
    from gateway.session_local import create_local_session
    from gateway import run
    from hermes_state_runtime import RuntimeStoreError

    config = {'platform_toolsets': {'cli': []}}
    monkeypatch.setattr(run, '_load_gateway_config', lambda: config)
    def runner():
        store = SessionStore(tmp_path / 'sessions', GatewayConfig())
        return SimpleNamespace(session_store=store, _session_db=store._db,
                               adapters={}, _draining=False)
    first = runner()
    authority = await initialize_session_authority(first, profile_id='fixture', instance_id='first')
    actor = Principal('owner', 'fixture', frozenset({'session:create', 'session:read'}), 'socket')
    params = {'request_id': 'retry', 'source': 'gui', 'cwd': str(tmp_path), 'model': 'frozen', 'toolsets': []}
    ref = create_local_session(authority, actor, params)
    policy = first.adapters[Platform.LOCAL].policies[authority.sessions[ref.session_id].source.chat_id]
    second = runner()
    cold = await initialize_session_authority(second, profile_id='fixture', instance_id='second')
    assert ref.session_id in cold.sessions, 'durable creation was lost with process-local registry'
    config['platform_toolsets']['cli'] = ['terminal']
    assert create_local_session(cold, actor, params) == ref
    assert second.adapters[Platform.LOCAL].policies[cold.sessions[ref.session_id].source.chat_id] == policy
    with pytest.raises(RuntimeStoreError, match='invalid_params'):
        create_local_session(cold, actor, {**params, 'source': 'tui'})
    foreign = Principal('other', 'fixture', actor.capabilities, 'other-socket')
    with pytest.raises(RuntimeStoreError, match='permission_denied'):
        await cold.resolve(foreign, ref)

    db = cold.db
    write = db._execute_write
    def abort(callback, **kwargs):
        def fail(conn):
            callback(conn)
            raise RuntimeError('injected transaction abort')
        return write(fail, **kwargs)
    before = db.session_count()
    monkeypatch.setattr(db, '_execute_write', abort)
    with pytest.raises(RuntimeError, match='injected transaction abort'):
        create_local_session(cold, actor, {**params, 'request_id': 'aborted'})
    monkeypatch.setattr(db, '_execute_write', write)
    assert db.session_count() == before
    assert set(cold.sessions) == {ref.session_id}
    with db._read_ctx() as conn:
        policies = conn.execute("SELECT value FROM state_meta WHERE key LIKE 'gateway.local_policy.v1:%'").fetchall()
    assert len(policies) == 1
    assert json.loads(policies[0][0])['principal_id'] == actor.subject
    from hermes_state_runtime import admit_session_input, list_session_admissions
    admit_session_input(db, epoch=cold.epoch, principal_id=actor.subject,
                        session_id=ref.session_id, request_id='preclaim', payload={'text': 'never run'})
    db._write_sql('UPDATE state_meta SET value=? WHERE key=?',
                  ('{}', 'gateway.local_policy.v1:' + ref.session_id))
    await cold._drain(ref)
    assert list_session_admissions(db, session_id=ref.session_id, pending_only=False)[0]['status'] == 'queued'


@pytest.mark.platforms("linux")
def test_real_daemon_recovers_only_authorized_never_started_local_work(tmp_path):
    from tests.gateway.fixtures.local_recovery_probe import probe
    print(json.dumps(probe(tmp_path)))


def test_messaging_session_history_follows_the_compression_tip(tmp_path, monkeypatch):
    """The authority keeps a messaging session's admission identity at the compression root,
    but attach/resume history must come from the current physical transcript."""
    from gateway.config import Platform
    from gateway.session_authority import LiveSession, SessionAuthority
    from gateway.session_contract import SessionRef
    from gateway.session_local_recovery import local_history
    from hermes_state import SessionDB
    from hermes_state_runtime import begin_runtime_epoch
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    with SessionDB(db_path=tmp_path / 'state.db') as db:
        db.create_session('root', source='telegram', session_key='agent:telegram:dm:1')
        db.append_message('root', 'user', 'BEFORE_COMPRESSION')
        assert db.try_acquire_compression_lock('root', 'holder')
        db.publish_compression_child(parent_session_id='root', child_session_id='child', source='telegram',
                                     messages=[{'role': 'user', 'content': 'SUMMARY'}], compression_lock_holder='holder')
        db.append_message('child', 'user', 'AFTER_COMPRESSION')
        authority = SessionAuthority(SimpleNamespace(_draining=False, config=SimpleNamespace(multiplex_profiles=False)),
                                     profile_id='owned', instance_id='current', db=db,
                                     epoch=begin_runtime_epoch(db, instance_id='current'))
        source = SimpleNamespace(platform=Platform.TELEGRAM, user_id='human')
        authority.sessions['root'] = LiveSession(source, 'agent:telegram:dm:1')
        history = local_history(authority, SessionRef('owned', 'root'))
        contents = [m['content'] for m in history]
        assert 'AFTER_COMPRESSION' in contents
        assert 'BEFORE_COMPRESSION' not in contents
        # Snapshot rows carry their durable row id (a session.mutate rewind's target_message_id),
        # under the public TranscriptMessage name, never the internal ``_row_id``.
        assert [m['row_id'] for m in history] == [m['id'] for m in db.get_messages('child')]
        assert not any('_row_id' in m for m in history)


@pytest.mark.asyncio
async def test_branch_and_local_entries_keep_the_routing_prune_sweep_running(tmp_path, monkeypatch):
    """Every producer of canonical routing entries (local create, API bind, ``session.mutate``
    branch) stamps the store clock: one aware-UTC entry makes ``prune_old_entries`` raise against
    its naive cutoff, and the watcher swallows that, so no stale route is ever pruned again."""
    from datetime import timedelta
    from gateway.config import GatewayConfig, Platform
    from gateway.session import SessionSource, SessionStore
    from gateway.session_api import bind_api_session
    from gateway.session_authority import initialize_session_authority
    from gateway.session_contract import Principal
    from gateway.session_local import create_local_session
    from gateway.session_mutations import mutate_session
    from gateway import run

    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'platform_toolsets': {'cli': []}})
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False)
    authority = await initialize_session_authority(runner, profile_id='fixture', instance_id='first')
    actor = Principal('owner', 'fixture', frozenset({'session:create', 'session:read', 'session:control'}), 'socket')
    ref = create_local_session(authority, actor, {'request_id': 'r', 'source': 'gui', 'cwd': str(tmp_path),
                                                  'model': 'frozen', 'toolsets': []})
    bind_api_session(authority, 'api-owner')
    branched = await mutate_session(authority, actor, ref, dict(session_id=ref.session_id, request_id='branch',
        expected_revision=0, expected_generation=0, operation='branch', payload={}))
    telegram = store.get_or_create_session(
        SessionSource(platform=Platform.TELEGRAM, chat_id='7', user_id='7', chat_type='dm'))
    # Cold store: only the durable index feeds the sweep.
    cold = SessionStore(tmp_path / 'sessions', GatewayConfig())
    with cold._lock:
        cold._ensure_loaded_locked()
        assert any(e.session_id == branched['branched_session_id'] for e in cold._entries.values())
        cold._entries[telegram.session_key].updated_at -= timedelta(days=90)
    assert cold.prune_old_entries(30) == 1
    assert cold.lookup_by_session_key(telegram.session_key) is None


@pytest.mark.asyncio
async def test_delete_retires_owner_receipts_and_never_resurrects_the_id(tmp_path, monkeypatch):
    """The deletion fence takes the local policy receipt (a cron policy's ``request_json`` holds
    the job and raw prompt) and the API binding + declared conversation key with the transcript;
    a later create/cron/API retry of the same deterministic id is refused instead of recreating it."""
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_api import bind_api_session
    from gateway.session_authority import initialize_session_authority
    from gateway.session_contract import Principal
    from gateway.session_local import create_local_session
    from gateway.session_mutations import mutate_session
    from gateway import run
    from hermes_state_runtime import RuntimeStoreError

    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'platform_toolsets': {'cli': []}})
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False)
    authority = await initialize_session_authority(runner, profile_id='fixture', instance_id='first')
    actor = Principal('owner', 'fixture', frozenset({'session:create', 'session:read', 'session:control'}), 'socket')
    params = {'request_id': 'r', 'source': 'gui', 'cwd': str(tmp_path), 'model': 'frozen', 'toolsets': []}
    local = create_local_session(authority, actor, params)
    api = bind_api_session(authority, 'api-owner', declared_key='conversation')
    for ref in (local, api):
        await mutate_session(authority, actor, ref, dict(session_id=ref.session_id, request_id='delete',
            expected_revision=0, expected_generation=0, operation='delete', payload={}))
    with authority.db._read_ctx() as conn:
        left = [k for (k,) in conn.execute("SELECT key FROM state_meta WHERE key LIKE 'gateway.local_policy.v1:%' "
                                           "OR key LIKE 'gateway.api.%'")]
    assert left == []
    with pytest.raises(RuntimeStoreError, match='not_found'):
        create_local_session(authority, actor, params)
    with pytest.raises(RuntimeStoreError, match='not_found'):
        bind_api_session(authority, 'api-owner')
    assert authority.db.get_session(local.session_id) is None
    assert authority.db.get_session('api-owner') is None
