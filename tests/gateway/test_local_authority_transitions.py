"""Local policy publication and detached lifecycle respect the durable FIFO."""
import asyncio
from dataclasses import asdict, replace
import json
from types import SimpleNamespace

import pytest

from gateway.session_contract import Submission


async def local_session(tmp_path, monkeypatch, source='cli'):
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_controls import AuthorityConnection
    from gateway.session_local import create_local_session
    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'platform_toolsets': {'cli': [], 'acp': []}})
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False,
        _cached_agent_for=lambda route: None, _adapter_for_source=lambda source: None,
        _evict_cached_agent=lambda route: None)
    authority = await initialize_session_authority(runner, profile_id=str(tmp_path), instance_id='fixture')
    connection = AuthorityConnection(authority, object(), {'user_id': 'owner'})
    ref = create_local_session(authority, connection.actor, {'request_id': 'create', 'source': source,
        'cwd': str(tmp_path), 'model': 'old', 'toolsets': []})
    return authority, connection, ref


@pytest.mark.asyncio
async def test_detached_acp_session_ends_when_running_work_settles(tmp_path, monkeypatch):
    owner, connection, ref = await local_session(tmp_path, monkeypatch, 'acp')
    running, release = asyncio.Event(), asyncio.Event()
    async def execute(authority, ref, row):
        running.set()
        await release.wait()
        return 'done'
    monkeypatch.setattr('gateway.session_finite.execute_finite_admission', execute)
    snapshot = await owner.attach(connection.actor, ref)
    try:
        await owner.submit(connection.actor, Submission('turn', ref, {'text': 'work'}, 'queue'))
        await asyncio.wait_for(running.wait(), 10)
        await owner.detach(connection.actor, snapshot.subscription_id)
        assert owner.db.get_session(ref.session_id)['ended_at'] is None
        release.set()
        await owner.sessions[ref.session_id].task
        assert owner.db.get_session(ref.session_id)['end_reason'] == 'acp_disconnect'
    finally:
        release.set()
        await connection.close()
        owner.db.close()


@pytest.mark.asyncio
async def test_model_commit_and_live_publication_hold_admission_and_claim_gate(tmp_path, monkeypatch):
    from gateway import session_mutations
    from gateway.session_policy import restore_policy
    from hermes_state_local import local_receipt
    from hermes_state_runtime import admit_session_input
    owner, connection, ref = await local_session(tmp_path, monkeypatch)
    live = owner.sessions[ref.session_id]
    committed, publish, attempted = asyncio.Event(), asyncio.Event(), asyncio.Event()
    original_to_thread = asyncio.to_thread
    async def barrier(function, *args, **kwargs):
        result = await original_to_thread(function, *args, **kwargs)
        if function is session_mutations.mutate_runtime_session:
            committed.set()
            await publish.wait()
        return result
    async def prepare(authority, live, payload, prepared):
        old = restore_policy(prepared['snapshot']['receipt']['policy'])
        config = old.config()
        config['model'] = {'default': 'new', 'provider': 'fixture'}
        return {**prepared, 'policy': asdict(replace(old, model='new', config_json=json.dumps(config)))}
    seen = []
    async def execute(authority, ref, row):
        seen.append(local_receipt(authority.db, ref.session_id)['policy']['model'])
        return 'done'
    monkeypatch.setattr('gateway.session_mutation_model.prepare_model', prepare)
    monkeypatch.setattr('gateway.session_finite.execute_finite_admission', execute)
    monkeypatch.setattr(asyncio, 'to_thread', barrier)
    admit_session_input(owner.db, epoch=owner.epoch, principal_id=connection.actor.subject,
        session_id=ref.session_id, request_id='queued', payload={'text': 'queued'})
    handle = owner._handle(ref)
    task = asyncio.create_task(session_mutations.mutate_session(owner, connection.actor, ref,
        dict(session_id=ref.session_id, request_id='model', operation='model', payload={'model': 'new'},
             expected_revision=handle.revision, expected_generation=handle.execution_generation)))
    submit = None
    try:
        await asyncio.wait_for(committed.wait(), 10)
        async def send():
            attempted.set()
            return await owner.submit(connection.actor, Submission('follower', ref, {'text': 'next'}, 'queue'))
        submit = asyncio.create_task(send())
        owner._schedule(ref)
        await attempted.wait()
        await asyncio.sleep(0)
        publish.set()
        await task
        await submit
        await live.task
        assert seen == ['new', 'new']
    finally:
        publish.set()
        await asyncio.gather(task, *([submit] if submit else []), return_exceptions=True)
        await connection.close()
        owner.db.close()


@pytest.mark.asyncio
async def test_provider_switch_drops_launch_key_but_retains_config_secrets(tmp_path, monkeypatch):
    from gateway.session_policy import build_policy, bind_launch_key, launch_key
    from gateway.session_policy_credentials import recover_config_secrets
    from gateway.session_mutation_model import prepare_model
    owner, connection, ref = await local_session(tmp_path, monkeypatch)
    secrets = {('providers', 'old', 'api_key'): 'config-secret'}
    policy = build_policy({'source': 'cli', 'cwd': str(tmp_path), 'model': 'old', 'provider': 'old'},
                          {'model': {'default': 'old', 'provider': 'old'},
                           'providers': {'old': {'api_key': 'config-secret'}}}, private_secrets=secrets)
    policy = bind_launch_key(owner, ref.session_id, policy, 'launch-secret', config_secrets=secrets)
    owner.runner._resolve_session_agent_runtime = lambda **kwargs: ('old', {'api_key': 'launch-secret'})
    monkeypatch.setattr('hermes_cli.model_switch.switch_model', lambda **kwargs: SimpleNamespace(
        success=True, new_model='new', target_provider='new-provider', base_url=None, provider_changed=True))
    try:
        prepared = await prepare_model(owner, owner.sessions[ref.session_id], {'model': 'new'},
            {'snapshot': {'receipt': {'session_id': ref.session_id, 'policy': asdict(policy)}}})
        changed = type(policy)(**prepared['policy'])
        assert launch_key(owner, changed) is None
        assert recover_config_secrets(owner, changed) == secrets
        assert launch_key(owner, policy) == 'launch-secret'
    finally:
        await connection.close()
        owner.db.close()


@pytest.mark.asyncio
async def test_branch_route_uses_store_clock_and_deleted_policy_is_retired(tmp_path, monkeypatch):
    from gateway.session_mutations import mutate_session
    from hermes_state_local import POLICY_PREFIX
    owner, connection, ref = await local_session(tmp_path, monkeypatch)
    try:
        result = await mutate_session(owner, connection.actor, ref,
            dict(session_id=ref.session_id, request_id='branch', operation='branch', payload={},
                 expected_revision=0, expected_generation=0))
        assert owner.runner.session_store.prune_old_entries(1) == 0
        child = result['branched_session_id']
        owner.db.delete_session(child)
        with owner.db._read_ctx() as conn:
            assert conn.execute('SELECT 1 FROM state_meta WHERE key=?', (POLICY_PREFIX + child,)).fetchone() is None
    finally:
        await connection.close()
        owner.db.close()
