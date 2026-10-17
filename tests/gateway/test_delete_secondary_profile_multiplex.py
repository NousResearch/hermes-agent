"""A multiplexed secondary profile's delete retires routing under one owner (#138826)."""
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from tests.gateway.test_session_authorities_multiplex import _reserve_homes, _runner


@pytest.mark.asyncio
async def test_delete_of_secondary_profile_session_does_not_409(tmp_path, monkeypatch):
    from gateway import run
    from gateway.run import _profile_runtime_scope
    from gateway.run_runtime import initialize_gateway_runtime
    from gateway.runtime_ownership import process_ownership
    from gateway.session_contract import Principal
    from gateway.session_local import create_local_session
    from gateway.session_mutations import mutate_session

    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'platform_toolsets': {'cli': []}})
    root, homes = _reserve_homes(tmp_path, monkeypatch, names=('alpha',))
    process_ownership.reserve([home for _, home in homes])
    runner = None
    try:
        runner = _runner(root, homes)
        await initialize_gateway_runtime(runner)
        _name, home = homes[1]
        authority = runner.session_authorities.for_home(home)
        assert authority is not runner.session_authority  # a secondary, never the launch authority
        assert runner.session_store._routing_db.db_path != authority.db.db_path  # the bug's premise
        actor = Principal('owner', authority.profile_id,
                          frozenset({'session:create', 'session:read', 'session:control'}), 'socket')
        with _profile_runtime_scope(home, {}):
            ref = create_local_session(authority, actor, {'request_id': 'r', 'source': 'gui',
                                                          'cwd': str(tmp_path), 'model': 'frozen', 'toolsets': []})
            # A messaging key routed at this session, published to the launch store's durable
            # index — the shape that survives deletes a user actually observes (#138862).
            from gateway.session import SessionEntry
            store = runner.session_store
            now = datetime.now(UTC)
            store._entries['agent:main:telegram:111'] = SessionEntry(
                session_key='agent:main:telegram:111', session_id=ref.session_id,
                created_at=now, updated_at=now)
            with store._lock:
                store._save()
            result = await mutate_session(authority, actor, ref, dict(
                session_id=ref.session_id, request_id='delete', expected_revision=0,
                expected_generation=0, operation='delete', payload={}))
        assert result.get('deleted_ids') == [ref.session_id]
        assert authority.db.get_session(ref.session_id) is None
        assert runner.session_store.lookup_by_session_id(ref.session_id) is None
        # The secondary's routes live in the launch home's shared index: the delete drops its
        # durable row there too, not only at the next whole-index save (a crash in between kept
        # a messaging route to a deleted session).
        routing_db = runner.session_store._routing_db
        assert not routing_db._read_all(
            'SELECT session_key FROM gateway_routing WHERE json_extract(entry_json, ?) = ?',
            ('$.session_id', ref.session_id))
    finally:
        for a in list(getattr(runner, 'session_authorities', None) or []):
            a.db.close()
        for _, home in homes:
            process_ownership.release(home)


@pytest.mark.asyncio
async def test_a_stale_routing_save_cannot_restore_a_deleted_secondary_route(tmp_path, monkeypatch):
    """A whole-index snapshot taken before a secondary profile's delete, saved after it, must not
    put the deleted session's route back in the launch home's shared index: the deletion fence
    has to live in the routing database, not only in the secondary's own store."""
    from gateway import run
    from gateway.run import _profile_runtime_scope
    from gateway.run_runtime import initialize_gateway_runtime
    from gateway.runtime_ownership import process_ownership
    from gateway.session import SessionEntry
    from gateway.session_contract import Principal
    from gateway.session_local import create_local_session
    from gateway.session_mutations import mutate_session

    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'platform_toolsets': {'cli': []}})
    root, homes = _reserve_homes(tmp_path, monkeypatch, names=('alpha',))
    process_ownership.reserve([home for _, home in homes])
    runner = None
    try:
        runner = _runner(root, homes)
        await initialize_gateway_runtime(runner)
        _name, home = homes[1]
        authority = runner.session_authorities.for_home(home)
        actor = Principal('owner', authority.profile_id,
                          frozenset({'session:create', 'session:read', 'session:control'}), 'socket')
        store = runner.session_store
        key = 'agent:main:telegram:222'
        with _profile_runtime_scope(home, {}):
            ref = create_local_session(authority, actor, {'request_id': 'r', 'source': 'gui',
                                                          'cwd': str(tmp_path), 'model': 'frozen', 'toolsets': []})
            now = datetime.now(UTC)
            store._entries[key] = SessionEntry(session_key=key, session_id=ref.session_id,
                                               created_at=now, updated_at=now)
            with store._lock:
                stale = store._snapshot_routing_locked()  # a worker's snapshot, saved late
            await mutate_session(authority, actor, ref, dict(
                session_id=ref.session_id, request_id='delete', expected_revision=0,
                expected_generation=0, operation='delete', payload={}))
            store._persist_routing_data(*stale)
        routing_db = store._routing_db
        assert key not in routing_db.load_gateway_routing_entries(scope=store._routing_scope())
        assert authority.db.get_session(ref.session_id) is None
    finally:
        for a in list(getattr(runner, 'session_authorities', None) or []):
            a.db.close()
        for _, home in homes:
            process_ownership.release(home)


def test_runtime_owns_authority_requires_runner_membership():
    from gateway.session_mutations import _runtime_owns_authority
    launch, secondary = SimpleNamespace(runner=None), SimpleNamespace(runner=None)
    runner = SimpleNamespace(session_authorities=[launch, secondary], session_authority=launch)
    launch.runner = runner
    secondary.runner = runner
    assert _runtime_owns_authority(launch)
    assert _runtime_owns_authority(secondary)
    # An authority the runner does not hold keeps the coordination refusal (fail-closed).
    foreign = SimpleNamespace(runner=SimpleNamespace(session_authorities=[object()], session_authority=object()))
    assert not _runtime_owns_authority(foreign)
    assert not _runtime_owns_authority(SimpleNamespace(runner=None))
    # A registry-less runner owns exactly its single launch authority.
    legacy = SimpleNamespace(runner=None)
    legacy_runner = SimpleNamespace(session_authority=legacy, session_authorities=None)
    legacy.runner = legacy_runner
    assert _runtime_owns_authority(legacy)
    assert not _runtime_owns_authority(SimpleNamespace(runner=legacy_runner))


def test_a_failed_shared_fence_fails_the_canonical_delete_not_the_prune(monkeypatch):
    """The shared-index fence must be durable before a delete reports success: a canonical
    delete refuses ``storage_unavailable`` (its exact retry repeats the fence); a bulk prune,
    which has no retry receipt, keeps its committed result and only logs."""
    from gateway.session_mutations import retire_live_sessions
    from hermes_state_runtime import RuntimeStoreError

    for target in ('gateway.session_policy.release_launch_secrets',
                   'gateway.session_local_mcp.release_editor_mcp',
                   'gateway.session_worker_construct.release_session_prefill'):
        monkeypatch.setattr(target, lambda *a, **k: None)

    def broken(ids):
        raise OSError('disk gone')
    store = SimpleNamespace(retire_runtime_sessions=lambda ids: None, fence_retired_routes=broken,
                            _routing_db=SimpleNamespace(db_path='/launch/state.db'))
    authority = SimpleNamespace(runner=SimpleNamespace(session_store=store), profile_id='p',
                                db=SimpleNamespace(db_path='/secondary/state.db'), sessions={})
    with pytest.raises(RuntimeStoreError) as raised:
        retire_live_sessions(authority, ['gone'], require_fence=True)
    assert raised.value.reason == 'storage_unavailable'
    retire_live_sessions(authority, ['gone'])
