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
        # The retire clears the in-memory entry but not the launch store's durable row: the
        # fence marker lives in the owner store, and the launch table is rewritten only by the
        # next whole-index save. Pin both halves — a live path always republishes on its next
        # save, while an offline CLI delete, with no live store, leaves its row behind for good.
        routing_db = runner.session_store._routing_db
        seated = {r['session_key'] for r in routing_db._read_all(
            'SELECT session_key FROM gateway_routing WHERE json_extract(entry_json, ?) = ?',
            ('$.session_id', ref.session_id))}
        assert 'agent:main:telegram:111' in seated
        with runner.session_store._lock:
            runner.session_store._save()
        assert not routing_db._read_all(
            'SELECT session_key FROM gateway_routing WHERE json_extract(entry_json, ?) = ?',
            ('$.session_id', ref.session_id)), 'the next whole-index save republishes the retire'
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
