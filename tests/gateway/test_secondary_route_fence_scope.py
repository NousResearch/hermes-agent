"""A multiplexed secondary's delete fences its routes in the shared index, not the launch profile.

The launch home's ``state.db`` is both the launch profile's session store and the routing index
every served profile shares. A secondary profile's delete or prune fences its routes there, so a
whole-index snapshot taken before it and saved after it cannot restore them (#138826). That fence
was the launch profile's own session-deletion marker, so the launch profile then refused the id
for good: ``POST /api/sessions`` answered 409 ``session_deleted`` for a session it never had.
"""
import json
from datetime import UTC, datetime

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from tests.gateway.test_session_authorities_multiplex import _reserve_homes, _runner

_KEY = 'route-fence-key'
_AUTH = {'Authorization': 'Bearer ' + _KEY}


async def _delete(authority, actor, ref):
    from gateway.session_mutations import mutate_session
    await mutate_session(authority, actor, ref, dict(
        session_id=ref.session_id, request_id='delete', expected_revision=0,
        expected_generation=0, operation='delete', payload={}))


async def _prune(authority, actor, ref):
    from gateway.session_prune import _apply
    authority.db.end_session(ref.session_id, 'test')
    await _apply(authority, {'older_than_days': None}, None)  # the owner-run `hermes sessions prune`


@pytest.mark.asyncio
@pytest.mark.parametrize('retire', [_delete, _prune], ids=['delete', 'prune'])
async def test_a_secondary_delete_leaves_the_id_free_in_the_launch_profile(tmp_path, monkeypatch, retire):
    from gateway import run
    from gateway.config import PlatformConfig
    from gateway.platforms.api_server import APIServerAdapter
    from gateway.run import _profile_runtime_scope
    from gateway.run_runtime import initialize_gateway_runtime
    from gateway.runtime_ownership import process_ownership
    from gateway.session import SessionEntry
    from gateway.session_contract import Principal
    from gateway.session_local import create_local_session
    from hermes_state_mutation_retirement import prune_retired_tombstones

    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'platform_toolsets': {'cli': []}})
    root, homes = _reserve_homes(tmp_path, monkeypatch, names=('alpha',))
    process_ownership.reserve([home for _, home in homes])
    runner = adapter = None
    try:
        runner = _runner(root, homes)
        await initialize_gateway_runtime(runner)
        launch, home = runner.session_authority, homes[1][1]
        alpha = runner.session_authorities.for_home(home)
        store = runner.session_store
        assert store._routing_db.db_path == launch.db.db_path != alpha.db.db_path  # the shared index
        actor = Principal('owner', alpha.profile_id,
                          frozenset({'session:create', 'session:read', 'session:control', 'session:operator'}),
                          'socket')
        key = 'agent:alpha:telegram:333'
        with _profile_runtime_scope(home, {}):
            ref = create_local_session(alpha, actor, {'request_id': 'r', 'source': 'gui',
                                                      'cwd': str(tmp_path), 'model': 'frozen', 'toolsets': []})
            now = datetime.now(UTC)
            store._entries[key] = SessionEntry(session_key=key, session_id=ref.session_id,
                                               created_at=now, updated_at=now)
            with store._lock:
                stale = store._snapshot_routing_locked()  # a worker's snapshot, saved late
            await retire(alpha, actor, ref)
            store._persist_routing_data(*stale)
        assert alpha.db.get_session(ref.session_id) is None

        def durable():
            return store._routing_db.load_gateway_routing_entries(scope=store._routing_scope())
        assert key not in durable(), 'a stale whole-index save restored the deleted route'

        # The launch profile never had this session: its own create must not see a deletion.
        adapter = APIServerAdapter(PlatformConfig(enabled=True))
        adapter._api_key, adapter.gateway_runner, adapter._session_db = _KEY, runner, launch.db
        app = web.Application()
        app.router.add_post('/api/sessions', adapter._handle_create_session)
        async with TestClient(TestServer(app)) as client:
            created = await client.post('/api/sessions', json={'id': ref.session_id}, headers=_AUTH)
            assert created.status == 201, await created.text()

        # With the launch profile now holding a session by that name, and past the retry horizon,
        # alpha's deleted route stays fenced: the stale save also wrote it into the sessions.json
        # mirror a restart imports, and a late per-entry save can still carry it.
        launch.db._execute_write(lambda conn: prune_retired_tombstones(conn, older_than=0))
        assert key in json.loads((store.sessions_dir / 'sessions.json').read_text())
        with store._lock:
            store._import_legacy_sessions_json(True)
        assert key not in store._entries, 'the sessions.json mirror restored the deleted route'
        store._routing_db.save_gateway_routing_entry(key, json.dumps(stale[0][key]), scope=store._routing_scope())
        assert key not in durable(), 'a late per-entry save restored the deleted route'
        # The launch profile's own route to its session of that name saves.
        own = 'agent:main:telegram:444'
        store._entries[own] = SessionEntry(session_key=own, session_id=ref.session_id,
                                           created_at=now, updated_at=now)
        with store._lock:
            store._save()
        assert own in durable(), "the launch profile's route to its own session was refused"
    finally:
        if adapter is not None:
            adapter._response_store.close()
            adapter._run_idempotency_store.close()
        for a in list(getattr(runner, 'session_authorities', None) or []):
            a.db.close()
        for _, home in homes:
            process_ownership.release(home)
