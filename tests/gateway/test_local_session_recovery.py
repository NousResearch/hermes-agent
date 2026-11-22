"""Local creation is durable before publication and restart never replays a claim."""
import json
from types import SimpleNamespace

import pytest

from tests.fixtures.storage_holder import policy_key, raw_holder
from tests.fixtures.storage_read import transcript_rows
from tests.fixtures.storage_seed import seed_messages, seed_session
from tests.gateway.fixtures.storage_handles import close_storage_handles, storage_handle


@pytest.mark.asyncio
async def test_local_create_receipt_survives_cold_authority_atomically(tmp_path, monkeypatch):
    from gateway.config import GatewayConfig, Platform
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_contract import SessionActor
    from gateway.session_local import create_local_session
    from gateway import run
    from persistence.core.errors import AuthorityRefusal

    config = {'platform_toolsets': {'cli': []}}
    monkeypatch.setattr(run, '_load_gateway_config', lambda: config)
    def runner():
        store = SessionStore(tmp_path / 'sessions', GatewayConfig())
        return SimpleNamespace(session_store=store, _session_db=store.persistence,
                               adapters={}, _draining=False)
    first = runner()
    authority = await initialize_session_authority(first, profile_id='fixture', instance_id='first')
    actor = SessionActor('owner', 'fixture', frozenset({'session:create', 'session:read'}), 'socket')
    params = {'request_id': 'retry', 'source': 'gui', 'cwd': str(tmp_path), 'model': 'frozen', 'toolsets': []}
    ref = create_local_session(authority, actor, params)
    policy = first.adapters[Platform.LOCAL].policies[authority.sessions[ref.session_id].source.chat_id]
    second = runner()
    cold = await initialize_session_authority(second, profile_id='fixture', instance_id='second')
    assert ref.session_id in cold.sessions, 'durable creation was lost with process-local registry'
    config['platform_toolsets']['cli'] = ['terminal']
    assert create_local_session(cold, actor, params) == ref
    assert second.adapters[Platform.LOCAL].policies[cold.sessions[ref.session_id].source.chat_id] == policy
    with pytest.raises(AuthorityRefusal, match='invalid_params'):
        create_local_session(cold, actor, {**params, 'source': 'tui'})
    foreign = SessionActor('other', 'fixture', actor.capabilities, 'other-socket')
    with pytest.raises(AuthorityRefusal, match='permission_denied'):
        await cold.resolve(foreign, ref)

    db, holder = cold.persistence, raw_holder(cold.persistence)
    write = holder._execute_write
    def abort(callback, **kwargs):
        def fail(conn):
            callback(conn)
            raise RuntimeError('injected transaction abort')
        return write(fail, **kwargs)
    before = db.sessions.count_sessions()
    monkeypatch.setattr(holder, '_execute_write', abort)
    with pytest.raises(RuntimeError, match='injected transaction abort'):
        create_local_session(cold, actor, {**params, 'request_id': 'aborted'})
    monkeypatch.setattr(holder, '_execute_write', write)
    assert db.sessions.count_sessions() == before
    assert set(cold.sessions) == {ref.session_id}
    with raw_holder(db)._read_ctx() as conn:
        policies = conn.execute('SELECT value FROM state_meta WHERE key LIKE ?', (policy_key() + '%',)).fetchall()
    assert len(policies) == 1
    assert json.loads(policies[0][0])['principal_id'] == actor.subject
    db.admissions.admit(epoch=cold.epoch, principal_id=actor.subject,
                        session_id=ref.session_id, request_id='preclaim', payload={'text': 'never run'})
    raw_holder(db)._execute_write(lambda conn: conn.execute('UPDATE state_meta SET value=? WHERE key=?',
                                                            ('{}', policy_key(ref.session_id))))
    await cold._drain(ref)
    assert db.admissions.list(session_id=ref.session_id, pending_only=False)[0].status == 'queued'


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
    from persistence.conversation.lineage import CompressionChild
    from persistence.conversation.messages import NewMessage
    from persistence_assembly import open_persistence, profile_scope
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    with open_persistence(profile_scope(home=tmp_path)) as db:
        seed_session(db, 'root', 'telegram', session_key='agent:telegram:dm:1')
        seed_messages(db, 'root', {'role': 'user', 'content': 'BEFORE_COMPRESSION'})
        assert db.coordination.try_acquire_compression_lock(session_id='root', holder='holder')
        db.lineage.publish_compression_child('root', CompressionChild(session_id='child', source='telegram'),
                                             [NewMessage.from_live({'role': 'user', 'content': 'SUMMARY'})],
                                             compression_lock_holder='holder')
        seed_messages(db, 'child', {'role': 'user', 'content': 'AFTER_COMPRESSION'})
        authority = SessionAuthority(SimpleNamespace(_draining=False, config=SimpleNamespace(multiplex_profiles=False)),
                                     profile_id='owned', instance_id='current',
                                     epoch=db.runtime.begin_epoch(instance_id='current'), persistence=storage_handle(tmp_path))
        source = SimpleNamespace(platform=Platform.TELEGRAM, user_id='human')
        authority.sessions['root'] = LiveSession(source, 'agent:telegram:dm:1')
        history = local_history(authority, SessionRef('owned', 'root'))
        contents = [m['content'] for m in history]
        assert 'AFTER_COMPRESSION' in contents
        # The root's turn is still shown, marked archived: display history, never a live row.
        assert [m['content'] for m in history if m.get('archived')] == ['BEFORE_COMPRESSION']
        # Snapshot rows carry their durable row id (a session.mutate rewind's target_message_id),
        # under the public TranscriptMessage name, never the internal ``_row_id``.
        assert [m['row_id'] for m in history if not m.get('archived')] == [m['id'] for m in transcript_rows(db, 'child')]
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
    from gateway.session_contract import SessionActor
    from gateway.session_local import create_local_session
    from gateway.session_mutations import mutate_session
    from gateway import run

    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'platform_toolsets': {'cli': []}})
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    runner = SimpleNamespace(session_store=store, _session_db=store.persistence, adapters={}, _draining=False)
    authority = await initialize_session_authority(runner, profile_id='fixture', instance_id='first')
    actor = SessionActor('owner', 'fixture', frozenset({'session:create', 'session:read', 'session:control'}), 'socket')
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
    from gateway.session_contract import SessionActor
    from gateway.session_local import create_local_session
    from gateway.session_mutations import mutate_session
    from gateway import run
    from persistence.core.errors import AuthorityRefusal

    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'platform_toolsets': {'cli': []}})
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    runner = SimpleNamespace(session_store=store, _session_db=store.persistence, adapters={}, _draining=False)
    authority = await initialize_session_authority(runner, profile_id='fixture', instance_id='first')
    actor = SessionActor('owner', 'fixture', frozenset({'session:create', 'session:read', 'session:control'}), 'socket')
    params = {'request_id': 'r', 'source': 'gui', 'cwd': str(tmp_path), 'model': 'frozen', 'toolsets': []}
    local = create_local_session(authority, actor, params)
    api = bind_api_session(authority, 'api-owner', declared_key='conversation')
    for ref in (local, api):
        await mutate_session(authority, actor, ref, dict(session_id=ref.session_id, request_id='delete',
            expected_revision=0, expected_generation=0, operation='delete', payload={}))
    with raw_holder(authority.persistence)._read_ctx() as conn:
        left = [k for (k,) in conn.execute('SELECT key FROM state_meta WHERE key LIKE ? OR key LIKE ?',
                                           (policy_key() + '%', 'gateway.api.%'))]
    assert left == []
    with pytest.raises(AuthorityRefusal, match='not_found'):
        create_local_session(authority, actor, params)
    with pytest.raises(AuthorityRefusal, match='not_found'):
        bind_api_session(authority, 'api-owner')
    assert authority.persistence.sessions.get(local.session_id) is None
    assert authority.persistence.sessions.get('api-owner') is None
