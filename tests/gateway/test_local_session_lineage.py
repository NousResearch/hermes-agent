"""A local conversation keeps one admission owner across physical transcript boundaries."""
import json
from types import SimpleNamespace

import pytest


@pytest.mark.asyncio
async def test_local_lineage_transitions_preserve_owner_or_roll_back(tmp_path, monkeypatch):
    from gateway import run
    from gateway.config import GatewayConfig, Platform
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_contract import Principal, SessionRef
    from gateway.session_local import create_local_session
    from hermes_state_local import local_receipt
    from hermes_state_runtime import (RuntimeStoreError, admit_session_input, claim_session_input,
                                      get_session_admission, list_session_admissions)

    monkeypatch.setattr(run, '_load_gateway_config', lambda: {})
    def runner():
        store = SessionStore(tmp_path / 'sessions', GatewayConfig())
        return SimpleNamespace(session_store=store, _session_db=store._db,
                               adapters={}, _draining=False)
    first = runner()
    authority = await initialize_session_authority(first, profile_id='fixture', instance_id='first')
    actor = Principal('owner', 'fixture', frozenset({'session:create', 'session:read'}), 'socket')
    params = dict(request_id='lineage', source='gui', cwd=str(tmp_path), model='frozen', toolsets=[])
    ref = create_local_session(authority, actor, params)
    db = authority.db
    live = authority.sessions[ref.session_id]
    original = local_receipt(db, ref.session_id)
    queued = admit_session_input(db, epoch=authority.epoch, principal_id=actor.subject,
                                session_id=ref.session_id, request_id='pending', payload={'text': 'pending'})
    history = [{'role': 'user', 'content': 'retained'}, {'role': 'assistant', 'content': 'history'}]
    def compress(parent, child):
        assert db.try_acquire_compression_lock(parent, 'fixture')
        try:
            db.publish_compression_child(parent_session_id=parent, child_session_id=child,
                                         source='gui', messages=history, compression_lock_holder='fixture')
        finally:
            db.release_compression_lock(parent, 'fixture')
    publish_row = db._publish_child_session_row
    def abort_publication(*args, **kwargs):
        publish_row(*args, **kwargs)
        raise RuntimeError('publication abort')
    monkeypatch.setattr(db, '_publish_child_session_row', abort_publication)
    with pytest.raises(RuntimeError, match='publication abort'):
        compress(ref.session_id, 'aborted-child')
    monkeypatch.setattr(db, '_publish_child_session_row', publish_row)
    assert db.get_session('aborted-child') is None
    assert local_receipt(db, ref.session_id) == original
    assert db.get_session(ref.session_id)['end_reason'] is None
    compress(ref.session_id, 'compressed')
    second = runner()
    cold = await initialize_session_authority(second, profile_id='fixture', instance_id='second')
    assert ref.session_id in cold.sessions, 'atomic compression lost the private local owner'
    assert create_local_session(cold, actor, params) == ref
    snapshot = await cold.attach(actor, ref)
    assert [(m['role'], m['content']) for m in snapshot.history] == [(m['role'], m['content']) for m in history]
    assert snapshot.pending[0].admission_id == queued['admission_id']
    assert second.session_store._entries[live.route].session_id == 'compressed'
    assert local_receipt(db, ref.session_id)['policy'] == original['policy']
    write = db._execute_write
    def abort(callback, **kwargs):
        def fail(conn):
            callback(conn)
            raise RuntimeError('transaction abort')
        return write(fail, **kwargs)
    # Use the owning SessionStore DB: reset must not publish any in-memory target on failure.
    db = cold.db
    write = db._execute_write
    before = local_receipt(db, ref.session_id)
    monkeypatch.setattr(db, '_execute_write', abort)
    with pytest.raises(RuntimeError, match='transaction abort'):
        second.session_store.reset_session(live.route)
    monkeypatch.setattr(db, '_execute_write', write)
    assert local_receipt(db, ref.session_id) == before
    assert second.session_store._entries[live.route].session_id == 'compressed'
    reset = second.session_store.reset_session(live.route)
    assert reset.session_id != 'compressed'
    assert db.get_session(reset.session_id)['cwd'] == params['cwd']
    assert db.get_session(reset.session_id)['model'] == params['model']
    third = runner()
    recovered = await initialize_session_authority(third, profile_id='fixture', instance_id='third')
    snapshot = await recovered.attach(actor, ref)
    assert snapshot.history == ()  # reset is a deliberate history boundary, not a fork
    assert snapshot.pending[0].admission_id == queued['admission_id']
    assert create_local_session(recovered, actor, params) == ref
    policy = third.adapters[Platform.LOCAL].policies[live.source.chat_id]
    assert policy.cwd == params['cwd'] and policy.source == params['source']
    # An interrupted claim remains owned by the logical conversation, even after another rotation.
    claimed = claim_session_input(recovered.db, epoch=recovered.epoch, session_id=ref.session_id)
    compress(reset.session_id, 'compressed-again')
    fourth = runner()
    restarted = await initialize_session_authority(fourth, profile_id='fixture', instance_id='fourth')
    assert get_session_admission(restarted.db, admission_id=claimed['admission_id'])['status'] == 'unknown'
    with pytest.raises(RuntimeStoreError, match='unknown_execution'):
        claim_session_input(restarted.db, epoch=restarted.epoch, session_id=ref.session_id)
    assert len(list_session_admissions(restarted.db, session_id=ref.session_id)) == 1
    receipt = local_receipt(db, ref.session_id)
    db.set_meta('gateway.local_policy.v1:' + ref.session_id, json.dumps(dict(receipt, profile_id='foreign')))
    with pytest.raises(RuntimeStoreError, match='storage_unavailable'):
        compress('compressed-again', 'foreign-child')
    assert db.get_session('foreign-child') is None
    db.set_meta('gateway.local_policy.v1:' + ref.session_id, json.dumps(receipt))
    foreign = Principal('foreign', 'fixture', actor.capabilities, 'foreign')
    with pytest.raises(RuntimeStoreError, match='permission_denied'):
        await restarted.attach(foreign, ref)
    wrong_profile = Principal('owner', 'other-profile', actor.capabilities, 'other-profile')
    with pytest.raises(RuntimeStoreError, match='profile_mismatch'):
        await restarted.attach(wrong_profile, ref)
    db.create_session('unrelated-fork', source='gui', parent_session_id='compressed-again',
                      model_config={'_branched_from': 'compressed-again'}, chat_id=live.source.chat_id,
                      user_id=actor.subject, session_key=live.route)
    with pytest.raises(RuntimeStoreError, match='storage_unavailable'):
        await restarted.attach(actor, SessionRef('fixture', 'unrelated-fork'))
    # Copying the receipt onto a fork does not turn it into the canonical owner.
    db.set_meta('gateway.local_policy.v1:unrelated-fork', json.dumps(local_receipt(db, ref.session_id)))
    with pytest.raises(RuntimeStoreError, match='storage_unavailable'):
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


@pytest.mark.linux_only
def test_real_local_lineage_survives_cold_daemon(tmp_path):
    from tests.gateway.fixtures.local_lineage_probe import probe
    print(json.dumps(probe(tmp_path)))
