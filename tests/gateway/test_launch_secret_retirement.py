"""Deleting a local session drops the launch key and frozen config secrets it held in memory."""
from types import SimpleNamespace

import pytest


@pytest.mark.asyncio
async def test_delete_prunes_launch_and_config_secrets(tmp_path, monkeypatch):
    from gateway import run, session_mutations
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_controls import AuthorityConnection
    from gateway.session_local import create_local_session
    config = {'platform_toolsets': {'cli': []}, 'model': {
        'provider': 'custom', 'api_key': 'config-secret', 'base_url': 'http://127.0.0.1:9/v1'}}
    monkeypatch.setattr(run, '_load_gateway_config', lambda *a, **k: config)
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False,
                             _cached_agent_for=lambda route: None, _adapter_for_source=lambda source: None,
                             _evict_cached_agent=lambda route: None)
    owner = await initialize_session_authority(runner, profile_id=str(tmp_path), instance_id='fixture')
    connection = AuthorityConnection(owner, object(), {'user_id': 'owner'})
    def create(request_id):
        return create_local_session(owner, connection.actor, {'request_id': request_id, 'source': 'cli',
            'cwd': str(tmp_path), 'model': 'm', 'toolsets': [], 'api_key': 'launch-' + request_id})
    try:
        doomed, kept = create('doomed'), create('kept')
        assert len(owner._local_launch_keys) == 2 and len(owner._local_config_secrets) == 2
        handle = owner._handle(doomed)
        await session_mutations.mutate_session(owner, connection.actor, doomed, dict(
            session_id=doomed.session_id, request_id='delete', operation='delete', payload={},
            expected_revision=handle.revision, expected_generation=handle.execution_generation))
        assert owner.db.get_session(doomed.session_id) is None
        assert 'launch-doomed' not in owner._local_launch_keys.values()
        assert list(owner._local_launch_keys.values()) == ['launch-kept']
        assert len(owner._local_config_secrets) == 1
        assert all(kept.session_id in ref for ref in owner._local_config_secrets)
    finally:
        await connection.close()
        owner.db.close()


async def _branching_owner(tmp_path, monkeypatch):
    """An owner whose sessions carry a config secret, with the mutate and next-turn key helpers."""
    from gateway import run, session_mutations
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_controls import AuthorityConnection
    from gateway.session_policy import launch_key, policy_for_source
    config = {'platform_toolsets': {'cli': []}, 'model': {
        'provider': 'custom', 'api_key': 'config-secret', 'base_url': 'http://127.0.0.1:9/v1'}}
    monkeypatch.setattr(run, '_load_gateway_config', lambda *a, **k: config)
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False,
                             _cached_agent_for=lambda route: None, _evict_cached_agent=lambda route: None,
                             _set_session_reasoning_override=lambda route, value: None)
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    owner = await initialize_session_authority(runner, profile_id=str(tmp_path), instance_id='fixture')
    connection = AuthorityConnection(owner, object(), {'user_id': 'owner'})

    async def mutate(ref, operation, payload=None, request_id=None):
        handle = owner._handle(ref)
        return await session_mutations.mutate_session(owner, connection.actor, ref, dict(
            session_id=ref.session_id, request_id=request_id or operation + ref.session_id, operation=operation,
            payload=payload or {}, expected_revision=handle.revision,
            expected_generation=handle.execution_generation))

    def key(ref):
        return launch_key(owner, policy_for_source(runner, owner.sessions[ref.session_id].source))
    return owner, connection, mutate, key


def _create(owner, connection, tmp_path, api_key):
    from gateway.session_local import create_local_session
    return create_local_session(owner, connection.actor, {'request_id': 'parent', 'source': 'cli',
        'cwd': str(tmp_path), 'model': 'm', 'toolsets': [], **({'api_key': api_key} if api_key else {})})


@pytest.mark.asyncio
@pytest.mark.parametrize('first', ['parent', 'branch'])
async def test_branches_keep_the_secrets_their_parent_bound_until_the_last_delete(tmp_path, monkeypatch, first):
    """A branch's policy names its parent's launch-key slot and config-secret reference: deleting
    either side keeps them for the survivors (#134889 RV-13). A delete releases only its own
    sessions' slots, so the parent's go with the parent's delete when no branch is left, else at restart."""
    from gateway.session_contract import SessionRef
    owner, connection, mutate, key = await _branching_owner(tmp_path, monkeypatch)
    try:
        parent = _create(owner, connection, tmp_path, 'launch-parent')
        child = SessionRef(parent.profile_id, (await mutate(parent, 'branch'))['branched_session_id'])
        grandchild = SessionRef(parent.profile_id, (await mutate(child, 'branch'))['branched_session_id'])
        secrets = dict(owner._local_config_secrets)
        live = [parent, child, grandchild]
        for ref in list(live) if first == 'parent' else live[::-1]:
            await mutate(ref, 'delete')
            live.remove(ref)
            assert [key(survivor) for survivor in live] == ['launch-parent'] * len(live)
            assert owner._local_config_secrets == secrets or not live
        left = [*owner._local_launch_keys, *owner._local_config_secrets]
        assert all(parent.session_id in ref for ref in left) and bool(left) == (first == 'parent')
    finally:
        await connection.close()
        owner.db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('api_key', [None, 'launch-parent'], ids=['config-secret', 'launch-key'])
async def test_a_branch_rebinds_after_its_parent_is_deleted(tmp_path, monkeypatch, api_key):
    """A branch whose deleted parent's secrets it still uses rebinds them under its own name (here a
    reasoning change) and keeps running; its own delete releases what it bound."""
    import copy
    from gateway import run
    from gateway.session_contract import SessionRef
    from gateway.session_policy import policy_for_source
    owner, connection, mutate, key = await _branching_owner(tmp_path, monkeypatch)
    try:
        parent = _create(owner, connection, tmp_path, api_key)
        child = SessionRef(parent.profile_id, (await mutate(parent, 'branch'))['branched_session_id'])
        await mutate(parent, 'delete')
        rotated = copy.deepcopy(run._load_gateway_config())
        rotated['model']['api_key'] = 'rotated-config-secret'  # only the frozen copy still answers
        monkeypatch.setattr(run, '_load_gateway_config', lambda *a, **k: rotated)
        await mutate(child, 'reasoning', {'effort': 'high'})
        assert key(child) == api_key
        assert policy_for_source(owner.runner, owner.sessions[child.session_id].source).config(owner)['model'][
            'api_key'] == 'config-secret'
        await mutate(child, 'delete')
        assert not owner.sessions
        assert all(child.session_id not in ref for ref in (*owner._local_launch_keys, *owner._local_config_secrets))
    finally:
        await connection.close()
        owner.db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('operation', ['reasoning', 'delete'])
async def test_an_unpublished_edit_keeps_the_secret_it_reuses(tmp_path, monkeypatch, operation):
    """Another session's edit or delete while an edit commits never revokes the secret that edit
    rebound into (here the parent's own, which a deleted branch's descendant also names), or the
    committed policy names a slot that is gone."""
    import asyncio
    import copy
    import threading
    from gateway import run, session_mutations
    from gateway.session_contract import SessionRef
    from gateway.session_policy import policy_for_source
    owner, connection, mutate, _ = await _branching_owner(tmp_path, monkeypatch)
    entered, release, pending = threading.Event(), threading.Event(), None

    def policy(ref):
        return policy_for_source(owner.runner, owner.sessions[ref.session_id].source)

    async def reasoning(ref, effort):
        return await mutate(ref, 'reasoning', {'effort': effort}, request_id='reasoning-' + effort)
    try:
        parent = _create(owner, connection, tmp_path, None)
        await reasoning(parent, 'high')
        frozen = policy(parent).config_secret_ref
        child = SessionRef(parent.profile_id, (await mutate(parent, 'branch'))['branched_session_id'])
        grandchild = SessionRef(parent.profile_id, (await mutate(child, 'branch'))['branched_session_id'])
        await mutate(child, 'delete')  # the grandchild still names the parent's `high` secret
        await reasoning(parent, 'low')
        rotated = copy.deepcopy(run._load_gateway_config())
        rotated['model']['api_key'] = 'rotated-config-secret'  # only the frozen copy still answers
        monkeypatch.setattr(run, '_load_gateway_config', lambda *a, **k: rotated)
        commit = session_mutations.mutate_runtime_session

        def held_commit(*args, **kwargs):
            if kwargs['request_id'] == 'reasoning-again' and not kwargs.get('_prepare_only'):
                entered.set()  # prepared and bound, not yet committed or published
                assert release.wait(10)
            return commit(*args, **kwargs)
        monkeypatch.setattr(session_mutations, 'mutate_runtime_session', held_commit)
        pending = asyncio.create_task(mutate(parent, 'reasoning', {'effort': 'high'}, request_id='reasoning-again'))
        assert await asyncio.to_thread(entered.wait, 10)
        await mutate(grandchild, operation, {'effort': 'medium'} if operation == 'reasoning' else None)
        release.set()
        await pending
        assert policy(parent).config_secret_ref == frozen
        assert policy(parent).config(owner)['model']['api_key'] == 'config-secret'
    finally:
        release.set()
        if pending is not None:
            await asyncio.gather(pending, return_exceptions=True)
        await connection.close()
        owner.db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('api_key', ['launch-parent', None], ids=['launch-key', 'config-secret'])
async def test_a_prune_keeps_the_parents_secret_for_a_branch_not_yet_published(tmp_path, monkeypatch, api_key):
    """A prune of a branch never releases its deleted parent's secrets: a grandchild committed off the
    branch but not yet published (absent from ``authority.sessions``) still names them."""
    import asyncio
    import copy
    import threading
    from gateway import run, session_mutations
    from gateway.session_contract import SessionRef
    from gateway.session_policy import policy_for_source, rebind_launch_key
    from gateway.session_prune import _apply
    owner, connection, mutate, key = await _branching_owner(tmp_path, monkeypatch)
    committed, release, pending = threading.Event(), threading.Event(), None
    try:
        parent = _create(owner, connection, tmp_path, api_key)
        child = SessionRef(parent.profile_id, (await mutate(parent, 'branch'))['branched_session_id'])
        await mutate(parent, 'delete')
        if api_key:
            rebind_launch_key(owner, child, api_key)  # the resume re-supply, which main also honours
        owner.db.end_session(child.session_id, 'tui_shutdown')
        rotated = copy.deepcopy(run._load_gateway_config())
        rotated['model']['api_key'] = 'rotated-config-secret'  # only the frozen copy still answers
        monkeypatch.setattr(run, '_load_gateway_config', lambda *a, **k: rotated)
        commit = session_mutations.mutate_runtime_session

        def held_publication(*args, **kwargs):
            result = commit(*args, **kwargs)
            if kwargs['request_id'] == 'held-branch' and not kwargs.get('_prepare_only'):
                committed.set()  # committed, not yet restored into the owner
                assert release.wait(10)
            return result
        monkeypatch.setattr(session_mutations, 'mutate_runtime_session', held_publication)
        pending = asyncio.create_task(mutate(child, 'branch', request_id='held-branch'))
        assert await asyncio.to_thread(committed.wait, 10)
        assert (await _apply(owner, {'older_than_days': None, 'chat_id': child.session_id}, None))['deleted'] == 1
        release.set()
        grandchild = SessionRef(parent.profile_id, (await pending)['branched_session_id'])
        if api_key:
            assert key(grandchild) == api_key
        else:
            assert policy_for_source(owner.runner, owner.sessions[grandchild.session_id].source).config(
                owner)['model']['api_key'] == 'config-secret'
    finally:
        release.set()
        if pending is not None:
            await asyncio.gather(pending, return_exceptions=True)
        await connection.close()
        owner.db.close()
