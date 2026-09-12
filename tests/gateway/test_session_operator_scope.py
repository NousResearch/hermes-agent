"""Operator scope changes access, not durable authorship or execution fences."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from gateway.session_contract import Principal, Submission
from gateway.session_controls import AuthorityConnection
from hermes_state_runtime import RuntimeStoreError, list_session_admissions


@pytest.mark.asyncio
async def test_operators_share_local_sessions_without_reassigning_receipts(tmp_path, monkeypatch):
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_local import create_local_session
    from gateway.session_local_title import resolve_titled_session
    from gateway.session_mutations import mutate_session
    from gateway.session_group_controls import _profiles

    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'platform_toolsets': {'cli': []}})
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False)
    authority = await initialize_session_authority(runner, profile_id=str(tmp_path), instance_id='fixture')
    caps = frozenset({'session:create', 'session:read', 'session:submit', 'session:control', 'session:operator'})
    native = Principal('uid:1000', str(tmp_path), caps, 'native')
    remote = Principal('auth:v1:remote', str(tmp_path), caps, 'remote')
    executed = []

    async def execute(authority, ref, row):
        executed.append((row['principal_id'], row['payload']['text']))
        return 'fixture result'

    monkeypatch.setattr('gateway.session_finite.execute_finite_admission', execute)
    for owner, peer in ((native, remote), (remote, native)):
        ref = create_local_session(authority, owner, {'request_id': owner.subject, 'source': 'gui',
            'cwd': str(tmp_path), 'model': 'fixture', 'toolsets': []})
        authority.db.append_message(ref.session_id, 'user', 'retained history')
        snapshot = await authority.attach(peer, ref)
        assert snapshot.history[0]['content'] == 'retained history'
        assert authority.sessions[ref.session_id].source.user_id == owner.subject
        edit = dict(session_id=ref.session_id, request_id='rename', expected_revision=0,
                    operation='rename', payload={'title': 'Bot Chat'})
        await mutate_session(authority, peer, ref, edit)
        assert resolve_titled_session(authority, peer, 'Bot Chat') == ref
        assert _profiles(authority, peer, tmp_path, {})['profiles'][0]['canonical_session']['id'] == ref.session_id
        with pytest.raises(RuntimeStoreError, match='revision_conflict'):
            await mutate_session(authority, peer, ref, {**edit, 'request_id': 'stale'})
        with pytest.raises(RuntimeStoreError, match='stale_generation'):
            await authority.interrupt(peer, ref, -1)
        for actor in (owner, peer):
            await authority.submit(actor, Submission('same-input', ref, {'text': actor.subject}, 'queue'))
        await authority.sessions[ref.session_id].task
        rows = list_session_admissions(authority.db, session_id=ref.session_id, pending_only=False)
        assert [row['principal_id'] for row in rows] == [owner.subject, peer.subject]
        assert all(row['status'] == 'terminal' for row in rows)
        assert executed[-2:] == [(owner.subject, owner.subject), (peer.subject, peer.subject)]
        for denied in (replace(peer, profile_id='foreign'), replace(peer, capabilities=caps - {'session:read'}),
                       replace(peer, capabilities=caps - {'session:operator'})):
            with pytest.raises(RuntimeStoreError):
                await authority.attach(denied, ref)
        await authority.detach(peer, snapshot.subscription_id)
        # Free the unique title before testing the inverse direction.
        authority.db.set_session_title(ref.session_id, 'Finished ' + owner.subject)
    # A disconnected operator's accepted turn must survive a new authority epoch.
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)
    receipt = await authority.submit(peer, Submission('restart', ref, {'text': 'after restart'}, 'queue'))
    runner.adapters = {}
    cold = await initialize_session_authority(runner, profile_id=str(tmp_path), instance_id='restarted')
    await cold._drain(ref)
    assert (await cold.receipt(peer, ref, receipt.admission_id)).status == 'terminal'
    assert executed[-1] == (peer.subject, 'after restart')
    assert (await cold.attach(peer, ref)).history[0]['content'] == 'retained history'
    store._db.close()


@pytest.mark.asyncio
async def test_identity_fields_cannot_grant_operator_or_first_claim(tmp_path, monkeypatch):
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_local import create_local_session

    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'platform_toolsets': {'cli': []}})
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False,
                             _cached_agent_for=lambda route: None)
    authority = await initialize_session_authority(runner, profile_id=str(tmp_path), instance_id='fixture')
    owner = Principal('owner', str(tmp_path), frozenset({'session:create'}), 'owner')
    ref = create_local_session(authority, owner, {'request_id': 'owner', 'source': 'gui',
        'cwd': str(tmp_path), 'model': 'fixture', 'toolsets': []})
    identity = {'user_id': 'other', 'provider': 'local', 'operator': True,
                'profile_id': str(tmp_path), 'instance_id': 'fixture', 'native_bootstrap': True,
                'capabilities': ['session:operator', 'session:read', 'session:create']}
    connection = AuthorityConnection(authority, object(), identity)
    assert 'session:operator' not in connection.actor.capabilities
    with pytest.raises(RuntimeStoreError, match='permission_denied'):
        await authority.attach(connection.actor, ref)
    authority.db.create_session('unowned', source='cli')
    operator = replace(connection.actor, capabilities=connection.actor.capabilities | {'session:operator'})
    from gateway.session_local_migration import resolve_local_target
    with pytest.raises(RuntimeStoreError):
        resolve_local_target(authority, operator, 'unowned')
    trusted = AuthorityConnection(authority, object(), {'user_id': 'verified'}, operator=True)
    assert 'session:operator' in trusted.actor.capabilities
    resumed = await trusted.dispatch({'id': 1, 'method': 'session.resume',
                                      'params': {'session_id': ref.session_id}})
    assert resumed['result']['session_id'] == ref.session_id
    listing = await trusted.dispatch({'id': 2, 'method': 'session.list', 'params': {}})
    assert ref.session_id in {s['id'] for s in listing['result']['sessions']}
    denied = await trusted.dispatch({'id': 3, 'method': 'prompt.submit', 'params': {
        'session_id': ref.session_id, 'input_id': 'forged', 'text': 'no',
        'local_operator_v1': {'principal_id': 'owner'}}})
    assert denied['error']['message'] == 'invalid_params'
    for changed in ({'instance_id': 'foreign'}, {'capabilities': ['worker:adopt']}):
        restricted = AuthorityConnection(authority, object(), {'user_id': 'verified', **changed}, operator=True)
        with pytest.raises(RuntimeStoreError):
            await authority.attach(restricted.actor, ref)
        await restricted.close()
    await trusted.close()
    await connection.close()
    store._db.close()
