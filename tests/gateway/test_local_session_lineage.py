"""A local conversation keeps one admission owner across physical transcript boundaries."""
import json
from types import SimpleNamespace

import pytest
from tests.fixtures.storage_holder import policy_key, raw_holder
from tests.fixtures.storage_read import session_row
from tests.fixtures.storage_seed import seed_session


@pytest.mark.asyncio
async def test_local_lineage_transitions_preserve_owner_or_roll_back(tmp_path, monkeypatch):
    from gateway import run
    from gateway.config import GatewayConfig, Platform
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_contract import SessionActor, SessionRef
    from gateway.session_local import create_local_session
    from persistence.conversation.lineage import CompressionChild
    from persistence.conversation.messages import NewMessage
    from persistence.core.errors import AuthorityRefusal

    monkeypatch.setattr(run, '_load_gateway_config', dict)
    def runner():
        store = SessionStore(tmp_path / 'sessions', GatewayConfig())
        return SimpleNamespace(session_store=store, _session_db=store.persistence,
                               adapters={}, _draining=False)
    first = runner()
    authority = await initialize_session_authority(first, profile_id='fixture', instance_id='first')
    actor = SessionActor('owner', 'fixture', frozenset({'session:create', 'session:read'}), 'socket')
    params = dict(request_id='lineage', source='gui', cwd=str(tmp_path), model='frozen', toolsets=[])
    ref = create_local_session(authority, actor, params)
    db = authority.persistence
    live = authority.sessions[ref.session_id]
    original = db.bindings.receipt(ref.session_id)
    queued, _ = db.admissions.admit(epoch=authority.epoch, principal_id=actor.subject,
                                    session_id=ref.session_id, request_id='pending', payload={'text': 'pending'})
    history = [{'role': 'user', 'content': 'retained'}, {'role': 'assistant', 'content': 'history'}]
    def publish(parent, child):
        db.lineage.publish_compression_child(parent, CompressionChild(session_id=child, source='gui'),
                                             [NewMessage.from_live(m) for m in history],
                                             compression_lock_holder='fixture')
    def compress(parent, child):
        assert db.coordination.try_acquire_compression_lock(session_id=parent, holder='fixture')
        try:
            publish(parent, child)
        finally:
            db.coordination.release_compression_lock(session_id=parent, holder='fixture')
    holder = raw_holder(db)
    publish_write = holder._execute_write
    def abort_publication(callback, **kwargs):
        def fail(conn):
            callback(conn)
            raise RuntimeError('publication abort')
        return publish_write(fail, **kwargs)
    assert db.coordination.try_acquire_compression_lock(session_id=ref.session_id, holder='fixture')
    try:
        monkeypatch.setattr(holder, '_execute_write', abort_publication)
        with pytest.raises(RuntimeError, match='publication abort'):
            publish(ref.session_id, 'aborted-child')
        monkeypatch.setattr(holder, '_execute_write', publish_write)
    finally:
        db.coordination.release_compression_lock(session_id=ref.session_id, holder='fixture')
    assert db.sessions.get('aborted-child') is None
    assert db.bindings.receipt(ref.session_id) == original
    assert db.sessions.get(ref.session_id).end_reason is None
    compress(ref.session_id, 'compressed')
    second = runner()
    cold = await initialize_session_authority(second, profile_id='fixture', instance_id='second')
    assert ref.session_id in cold.sessions, 'atomic compression lost the private local owner'
    assert create_local_session(cold, actor, params) == ref
    snapshot = await cold.attach(actor, ref)
    assert [(m['role'], m['content']) for m in snapshot.history] == [(m['role'], m['content']) for m in history]
    assert snapshot.pending[0].admission_id == queued.admission_id
    assert second.session_store._entries[live.route].session_id == 'compressed'
    assert db.bindings.receipt(ref.session_id).policy == original.policy
    write = raw_holder(db)._execute_write
    def abort(callback, **kwargs):
        def fail(conn):
            callback(conn)
            raise RuntimeError('transaction abort')
        return write(fail, **kwargs)
    # Use the owning SessionStore's store: reset must not publish any in-memory target on failure.
    db = cold.persistence
    write = (holder := raw_holder(second.session_store._persistence_for_key(live.route)))._execute_write
    before = db.bindings.receipt(ref.session_id)
    monkeypatch.setattr(holder, '_execute_write', abort)
    with pytest.raises(RuntimeError, match='transaction abort'):
        second.session_store.reset_session(live.route)
    monkeypatch.setattr(holder, '_execute_write', write)
    assert db.bindings.receipt(ref.session_id) == before
    assert second.session_store._entries[live.route].session_id == 'compressed'
    reset = second.session_store.reset_session(live.route)
    assert reset.session_id != 'compressed'
    assert session_row(db, reset.session_id)['cwd'] == params['cwd']
    assert session_row(db, reset.session_id)['model'] == params['model']
    third = runner()
    recovered = await initialize_session_authority(third, profile_id='fixture', instance_id='third')
    snapshot = await recovered.attach(actor, ref)
    assert snapshot.history == ()  # reset is a deliberate history boundary, not a fork
    assert snapshot.pending[0].admission_id == queued.admission_id
    assert create_local_session(recovered, actor, params) == ref
    policy = third.adapters[Platform.LOCAL].policies[live.source.chat_id]
    assert policy.cwd == params['cwd'] and policy.source == params['source']
    # An interrupted claim remains owned by the logical conversation, even after another rotation.
    claimed = recovered.persistence.admissions.claim(epoch=recovered.epoch, session_id=ref.session_id)
    compress(reset.session_id, 'compressed-again')
    fourth = runner()
    restarted = await initialize_session_authority(fourth, profile_id='fixture', instance_id='fourth')
    assert restarted.persistence.admissions.get(admission_id=claimed.admission_id).status == 'unknown'
    with pytest.raises(AuthorityRefusal, match='unknown_execution'):
        restarted.persistence.admissions.claim(epoch=restarted.epoch, session_id=ref.session_id)
    assert len(restarted.persistence.admissions.list(session_id=ref.session_id)) == 1
    receipt = json.loads(db.state_meta.get(policy_key(ref.session_id)))
    db.state_meta.set(policy_key(ref.session_id), json.dumps(dict(receipt, profile_id='foreign')))
    with pytest.raises(AuthorityRefusal, match='storage_unavailable'):
        compress('compressed-again', 'foreign-child')
    assert db.sessions.get('foreign-child') is None
    db.state_meta.set(policy_key(ref.session_id), json.dumps(receipt))
    foreign = SessionActor('foreign', 'fixture', actor.capabilities, 'foreign')
    with pytest.raises(AuthorityRefusal, match='permission_denied'):
        await restarted.attach(foreign, ref)
    wrong_profile = SessionActor('owner', 'other-profile', actor.capabilities, 'other-profile')
    with pytest.raises(AuthorityRefusal, match='profile_mismatch'):
        await restarted.attach(wrong_profile, ref)
    seed_session(db, 'unrelated-fork', 'gui', parent_session_id='compressed-again',
                 model_config={'_branched_from': 'compressed-again'}, chat_id=live.source.chat_id,
                 user_id=actor.subject, session_key=live.route)
    with pytest.raises(AuthorityRefusal, match='storage_unavailable'):
        await restarted.attach(actor, SessionRef('fixture', 'unrelated-fork'))
    # Copying the receipt onto a fork does not turn it into the canonical owner.
    db.state_meta.set(policy_key('unrelated-fork'), db.state_meta.get(policy_key(ref.session_id)))
    with pytest.raises(AuthorityRefusal, match='storage_unavailable'):
        await restarted.attach(actor, SessionRef('fixture', 'unrelated-fork'))
    # Route reuse by a native reset must not pick a departed logical owner's generation.
    from gateway.run_turn_runner import TurnRunner
    from gateway.session import SessionSource
    from gateway.session_authority import LiveSession
    restarted.sessions[reset.session_id] = LiveSession(None, live.route)
    native = SimpleNamespace(session_id=reset.session_id, session_key=live.route,
                             source=SessionSource(Platform.TELEGRAM, 'native'))
    turn = TurnRunner(fourth, native)
    assert turn._approval_owner[1] == reset.session_id


@pytest.mark.platforms("linux")
def test_real_local_lineage_survives_cold_daemon(tmp_path):
    from tests.gateway.fixtures.local_lineage_probe import probe
    print(json.dumps(probe(tmp_path)))
