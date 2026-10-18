"""Retirement is keyed by (profile, id): a multiplexed secondary's delete or prune retires the id
there, never in the launch profile.

The launch home's ``state.db`` is both the launch profile's session store and the routing index
every served profile shares. A secondary profile's delete or prune fences its routes there, so a
whole-index snapshot taken before it and saved after it cannot restore them (#138826). That fence
was the launch profile's own session-deletion marker, so the launch profile then refused the id
for good: ``POST /api/sessions`` answered 409 ``session_deleted`` for a session it never had, and
after a prune the refusal never aged out.
"""
import json
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from tests.gateway.test_session_authorities_multiplex import _reserve_homes, _runner

_KEY = 'route-fence-key'
_AUTH = {'Authorization': 'Bearer ' + _KEY}


@asynccontextmanager
async def _served(tmp_path, monkeypatch):
    """A multiplexed gateway: the default (launch) profile serving ``alpha``, real state.db files."""
    from gateway import run
    from gateway.config import PlatformConfig
    from gateway.platforms.api_server import APIServerAdapter
    from gateway.run_runtime import initialize_gateway_runtime
    from gateway.runtime_ownership import process_ownership
    from gateway.session_contract import Principal

    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'platform_toolsets': {'cli': []}})
    root, homes = _reserve_homes(tmp_path, monkeypatch, names=('alpha',))
    process_ownership.reserve([home for _, home in homes])
    runner = adapter = None
    try:
        runner = _runner(root, homes)
        await initialize_gateway_runtime(runner)
        launch, home = runner.session_authority, homes[1][1]
        alpha = runner.session_authorities.for_home(home)
        assert runner.session_store._routing_db.db_path == launch.db.db_path != alpha.db.db_path
        actor = Principal('owner', alpha.profile_id, frozenset(
            {'session:create', 'session:read', 'session:control', 'session:operator'}), 'socket')
        adapter = APIServerAdapter(PlatformConfig(enabled=True))
        adapter._api_key, adapter.gateway_runner, adapter._session_db = _KEY, runner, launch.db
        yield SimpleNamespace(runner=runner, launch=launch, alpha=alpha, home=home, actor=actor,
                              store=runner.session_store, adapter=adapter)
    finally:
        if adapter is not None:
            adapter._response_store.close()
            adapter._run_idempotency_store.close()
        for a in list(getattr(runner, 'session_authorities', None) or []):
            a.db.close()
        for _, home in homes:
            process_ownership.release(home)


async def _delete(served, ref):
    from gateway.session_mutations import mutate_session
    await mutate_session(served.alpha, served.actor, ref, dict(
        session_id=ref.session_id, request_id='delete', expected_revision=0,
        expected_generation=0, operation='delete', payload={}))


async def _prune(served, ref):
    from gateway.session_prune import _apply
    served.alpha.db.end_session(ref.session_id, 'test')
    await _apply(served.alpha, {'older_than_days': None}, None)  # the owner-run `hermes sessions prune`


def _entry(key, sid):
    from gateway.session import SessionEntry
    now = datetime.now(UTC)
    return SessionEntry(session_key=key, session_id=sid, created_at=now, updated_at=now)


def _durable(store):
    return store._routing_db.load_gateway_routing_entries(scope=store._routing_scope())


def _age_retirements(*dbs, days=8):
    """Move every retirement marker and routing fence *days* into the past: the wall clock passing
    the horizon, with no sweep run anywhere."""
    from hermes_state_mutation_retirement import RETIRED_PREFIX, ROUTE_FENCE_PREFIX
    for db in dbs:
        db._execute_write(lambda conn: conn.execute(
            "UPDATE state_meta SET value=json_set(value, '$.retired_at', json_extract(value, '$.retired_at') - ?) "
            "WHERE (key GLOB ? OR key GLOB ?) AND json_valid(value)",
            (days * 86400, RETIRED_PREFIX + '*', ROUTE_FENCE_PREFIX + '*')))


def _past_the_horizon(*dbs):
    """The wall clock passes the horizon (every fence aged), then each store sweeps its tombstones."""
    from hermes_state_mutation_retirement import prune_retired_tombstones
    _age_retirements(*dbs)
    for db in dbs:
        db._execute_write(lambda conn: prune_retired_tombstones(conn, older_than=0))


def _conflict(db, sid):
    from hermes_state_mutation_retirement import create_conflict
    with db._read_ctx() as conn:
        return create_conflict(conn, sid)


async def _create(served, sid):
    app = web.Application()
    app.router.add_post('/api/sessions', served.adapter._handle_create_session)
    async with TestClient(TestServer(app)) as client:
        created = await client.post('/api/sessions', json={'id': sid}, headers=_AUTH)
        return created.status, await created.text()


@pytest.mark.asyncio
@pytest.mark.parametrize('retire', [_delete, _prune], ids=['delete', 'prune'])
async def test_a_secondary_retirement_leaves_the_id_free_in_the_launch_profile(tmp_path, monkeypatch, retire):
    from gateway.run import _profile_runtime_scope
    from gateway.session_api import bind_api_session
    from gateway.session_local import create_local_session

    async with _served(tmp_path, monkeypatch) as served:
        store = served.store
        own, launch_keyed = 'agent:alpha:telegram:333', 'agent:main:telegram:333'
        with _profile_runtime_scope(served.home, {}):
            ref = create_local_session(served.alpha, served.actor, {
                'request_id': 'r', 'source': 'gui', 'cwd': str(tmp_path), 'model': 'frozen', 'toolsets': []})
            sid = ref.session_id
            store._entries[own] = _entry(own, sid)
            store._entries[launch_keyed] = _entry(launch_keyed, sid)  # a route keyed in the launch namespace
            with store._lock:
                stale = store._snapshot_routing_locked()  # a worker's snapshot, saved late
            await retire(served, ref)
            store._persist_routing_data(*stale)
        assert served.alpha.db.get_session(sid) is None
        assert own not in _durable(store) and launch_keyed not in _durable(store), \
            'a stale whole-index save restored the deleted route'

        # The launch profile never had this session: its own create and bind see no deletion.
        assert _conflict(served.launch.db, sid) is None
        status, text = await _create(served, sid)
        assert status == 201, text
        bind_api_session(served.launch, sid)  # the X-Hermes-Session-Id chat's bind
        fence = 'gateway.retired_route.v1.alpha:' + sid  # alpha's (profile, id) fence in the shared index
        assert served.launch.db.get_meta(fence) is not None
        # The secondary itself still refuses its own retired id within the horizon.
        assert _conflict(served.alpha.db, sid) == ('session_deleted', 'Session was deleted; its id cannot be reused')

        _past_the_horizon(served.alpha.db, served.launch.db)
        if retire is _delete:  # a user delete stays final in alpha, and so does its routing fence
            assert _conflict(served.alpha.db, sid) is not None
            assert served.launch.db.get_meta(fence) is not None
            store._routing_db.save_gateway_routing_entry(own, json.dumps(stale[0][own]), scope=store._routing_scope())
            assert own not in _durable(store), 'a late per-entry save restored the deleted route'
        else:  # a prune is not a delete: alpha may reuse the id, and the shared fence ages out with it
            assert _conflict(served.alpha.db, sid) is None
            assert served.launch.db.get_meta(fence) is None


@pytest.mark.asyncio
async def test_a_launch_delete_fences_only_the_launch_profiles_routes(tmp_path, monkeypatch):
    """The launch profile deleting its ``user-42`` fences its own routes to that id in the shared
    index, not ``alpha``'s route to alpha's own ``user-42``."""
    async with _served(tmp_path, monkeypatch) as served:
        store = served.store
        for db in (served.launch.db, served.alpha.db):
            db.create_session('user-42', source='cli')
        launch_key, alpha_key = 'agent:main:telegram:1', 'agent:alpha:telegram:1'
        store._entries[launch_key] = _entry(launch_key, 'user-42')
        store._entries[alpha_key] = _entry(alpha_key, 'user-42')
        with store._lock:
            stale = store._snapshot_routing_locked()
        served.launch.db.delete_session('user-42')
        store._persist_routing_data(*stale)
        assert launch_key not in _durable(store), "a stale save restored the launch profile's deleted route"
        assert alpha_key in _durable(store), "alpha's route to its own session was refused"
        assert _conflict(served.alpha.db, 'user-42') == ('session_exists', 'Session already exists')


@pytest.mark.asyncio
async def test_a_shared_fence_an_earlier_release_wrote_still_guards_every_route(tmp_path, monkeypatch):
    """An earlier release fenced a secondary's delete in the shared index with the launch profile's
    own marker (``retired_session.v1.<id>``, only ``retired_at``), which can't be told apart from a
    launch delete. Read through the new code it keeps both meanings: it still refuses every route to
    the id (a sessions.json mirror from before the delete must not bring one back), and the launch
    profile keeps refusing the id as deleted."""
    from hermes_state_mutation_retirement import RETIRED_PREFIX
    async with _served(tmp_path, monkeypatch) as served:
        store = served.store
        served.launch.db.set_meta(RETIRED_PREFIX + 'gone', json.dumps({'retired_at': time.time() - 30 * 86400}))
        _past_the_horizon(served.launch.db)
        key = 'agent:alpha:telegram:9'
        store._entries[key] = _entry(key, 'gone')
        with store._lock:
            store._save()
        assert key not in _durable(store)
        assert key not in json.loads((store.sessions_dir / 'sessions.json').read_text())
        store._save_sessions_json({key: store._entries.pop(key).to_dict()})  # an earlier release's unfiltered mirror
        with store._lock:
            store._import_legacy_sessions_json(True)
        assert key not in store._entries, 'the sessions.json mirror restored a fenced route'
        assert _conflict(served.launch.db, 'gone') == ('session_deleted', 'Session was deleted; its id cannot be reused')


def _mirror(store):
    return json.loads((store.sessions_dir / 'sessions.json').read_text())


def _restart(store):
    """The next gateway start's routing load: state.db rows, then the sessions.json import."""
    with store._lock:
        store._entries.clear()
        store._loaded = False
        store._ensure_loaded_locked()


async def _pruned_alpha_route(served, tmp_path):
    """``(key, sid, stale)``: alpha's routed session, pruned through the owner path after a
    worker took a whole-index snapshot that still names its route."""
    from gateway.run import _profile_runtime_scope
    from gateway.session_local import create_local_session
    store, key = served.store, 'agent:alpha:telegram:333'
    with _profile_runtime_scope(served.home, {}):
        ref = create_local_session(served.alpha, served.actor, {
            'request_id': 'r', 'source': 'gui', 'cwd': str(tmp_path), 'model': 'frozen', 'toolsets': []})
        store._entries[key] = _entry(key, ref.session_id)
        with store._lock:
            stale = store._snapshot_routing_locked()
        await _prune(served, ref)
    return key, ref.session_id, stale


@pytest.mark.asyncio
async def test_a_delayed_snapshot_cannot_restore_a_pruned_route_once_its_fence_ages_out(tmp_path, monkeypatch):
    """The delayed snapshot is filtered in ``gateway_routing``, and in its sessions.json mirror too:
    otherwise, once the prune's shared fence aged out, the next start imported the mirror's route
    (its session row is gone, and the stale pass keeps routes without a row). The id itself is free
    again: alpha recreating it routes as usual."""
    async with _served(tmp_path, monkeypatch) as served:
        store = served.store
        key, sid, stale = await _pruned_alpha_route(served, tmp_path)
        store._persist_routing_data(*stale)
        assert key not in _durable(store)
        assert key not in _mirror(store), 'the sessions.json mirror kept the pruned route'
        _past_the_horizon(served.alpha.db, served.launch.db)
        _restart(store)
        assert key not in store._entries, 'a pruned route came back on startup after its fence aged out'

        served.alpha.db.create_session(sid, source='telegram')
        store._entries[key] = _entry(key, sid)
        with store._lock:
            store._save()
        assert key in _durable(store) and key in _mirror(store), 'the recreated id was refused a route'
        _restart(store)
        assert store._entries[key].session_id == sid


@pytest.mark.asyncio
async def test_a_pruned_id_routes_again_after_the_owner_alone_sweeps_past_the_horizon(tmp_path, monkeypatch):
    """Only the owner's own housekeeping runs (the launch profile's auto-prune is off by default):
    once the horizon passes, the shared fence of alpha's prune no longer refuses the route of a
    session alpha recreates with that id, though nothing swept the launch database."""
    async with _served(tmp_path, monkeypatch) as served:
        store = served.store
        key, sid, _stale = await _pruned_alpha_route(served, tmp_path)
        _age_retirements(served.alpha.db, served.launch.db)
        served.alpha.db.prune_sessions(older_than_days=90)  # alpha's auto-prune pass: its own db only
        assert _conflict(served.alpha.db, sid) is None

        served.alpha.db.create_session(sid, source='telegram')
        store._entries[key] = _entry(key, sid)
        with store._lock:
            store._save()
        assert key in _durable(store), 'an expired shared fence refused the recreated route'
        assert key in _mirror(store), 'an expired shared fence kept the recreated route out of sessions.json'
        _restart(store)
        assert store._entries[key].session_id == sid


@pytest.mark.asyncio
async def test_the_owners_next_prune_reclaims_its_expired_shared_fences(tmp_path, monkeypatch):
    """An owner's prune that fences new ids in the shared index also reclaims the fences there that
    aged out, so the index does not grow with every prune while the launch profile never sweeps.
    A delete's fence stays."""
    from gateway.run import _profile_runtime_scope
    from gateway.session_local import create_local_session
    async with _served(tmp_path, monkeypatch) as served:
        _key, sid, _stale = await _pruned_alpha_route(served, tmp_path)
        with _profile_runtime_scope(served.home, {}):
            kept = create_local_session(served.alpha, served.actor, {
                'request_id': 'k', 'source': 'gui', 'cwd': str(tmp_path), 'model': 'frozen', 'toolsets': []})
            await _delete(served, kept)
        _age_retirements(served.alpha.db, served.launch.db)
        with _profile_runtime_scope(served.home, {}):
            later = create_local_session(served.alpha, served.actor, {
                'request_id': 'l', 'source': 'gui', 'cwd': str(tmp_path), 'model': 'frozen', 'toolsets': []})
            await _prune(served, later)
        fence = 'gateway.retired_route.v1.{}:{}'.format
        assert served.launch.db.get_meta(fence('alpha', sid)) is None, 'an expired shared fence was never reclaimed'
        assert served.launch.db.get_meta(fence('main', sid)) is None
        assert served.launch.db.get_meta(fence('alpha', later.session_id)) is not None
        assert served.launch.db.get_meta(fence('alpha', kept.session_id)) is not None, "a delete's fence aged out"


@pytest.mark.asyncio
async def test_a_secondary_delete_fence_outlives_the_horizon_while_a_prune_fence_ages_out(tmp_path, monkeypatch):
    """A delete's shared fence has no ``expires``: 8 days later, with no sweep run, it still refuses
    alpha's route and the launch-keyed one, both to a per-entry save and to the sessions.json import
    on the next start (#138826). A prune's fence aged the same way no longer fences."""
    from gateway.run import _profile_runtime_scope
    from gateway.session_local import create_local_session
    async with _served(tmp_path, monkeypatch) as served:
        store = served.store
        pruned_key, pruned, _stale = await _pruned_alpha_route(served, tmp_path)
        keys = ('agent:alpha:telegram:44', 'agent:main:telegram:44')
        with _profile_runtime_scope(served.home, {}):
            ref = create_local_session(served.alpha, served.actor, {
                'request_id': 'd', 'source': 'gui', 'cwd': str(tmp_path), 'model': 'frozen', 'toolsets': []})
            for key in keys:
                store._entries[key] = _entry(key, ref.session_id)
            with store._lock:
                stale = store._snapshot_routing_locked()
            await _delete(served, ref)
        _age_retirements(served.alpha.db, served.launch.db)
        for key in keys:
            store._routing_db.save_gateway_routing_entry(key, json.dumps(stale[0][key]), scope=store._routing_scope())
            assert key not in _durable(store), f'an aged delete fence let a per-entry save restore {key}'
        store._save_sessions_json({key: stale[0][key] for key in keys})  # a mirror written before the delete
        _restart(store)
        for key in keys:
            assert key not in store._entries, f'an aged delete fence let the sessions.json import restore {key}'

        store._routing_db.save_gateway_routing_entry(
            pruned_key, json.dumps(_entry(pruned_key, pruned).to_dict()), scope=store._routing_scope())
        assert pruned_key in _durable(store), 'a prune fence aged the same way still fenced its route'


@pytest.mark.asyncio
async def test_a_start_within_the_horizon_strips_a_pruned_route_from_an_older_mirror(tmp_path, monkeypatch):
    """A sessions.json an earlier release wrote unfiltered (or one left by a crash before the next
    save) still names the pruned route. The first start while the fence holds skips it and rewrites
    the mirror, so the route cannot come back once the fence ages out."""
    async with _served(tmp_path, monkeypatch) as served:
        store = served.store
        key, _sid, stale = await _pruned_alpha_route(served, tmp_path)
        store._save_sessions_json(stale[0])
        _restart(store)
        assert key not in store._entries
        assert key not in _mirror(store), 'the start kept the pruned route in the mirror'
        _past_the_horizon(served.alpha.db, served.launch.db)
        _restart(store)
        assert key not in store._entries, 'a pruned route came back on startup after its fence aged out'


@pytest.mark.asyncio
async def test_deleting_the_launch_session_keeps_an_earlier_releases_shared_fence_on_every_route(tmp_path, monkeypatch):
    """Before the upgrade, alpha deleted its ``user-42`` while the launch profile had a live
    ``user-42``: the shared index got the launch store's unscoped marker, and no (profile, id)
    fence. Deleting the launch session after the upgrade rewrites that marker. It must stay
    unscoped, or alpha's deleted route comes back through a late per-entry save or the
    sessions.json import on the next start."""
    from hermes_state_mutation_retirement import RETIRED_PREFIX
    async with _served(tmp_path, monkeypatch) as served:
        store, key = served.store, 'agent:alpha:telegram:42'
        served.launch.db.create_session('user-42', source='cli')
        old = json.dumps({'retired_at': time.time() - 86400})  # an earlier release's marker
        served.launch.db.set_meta(RETIRED_PREFIX + 'user-42', old)
        served.alpha.db.set_meta(RETIRED_PREFIX + 'user-42', old)
        stale = _entry(key, 'user-42').to_dict()  # alpha's route, from a snapshot taken before its delete

        served.launch.db.delete_session('user-42')
        assert served.launch.db.get_session('user-42') is None
        store._routing_db.save_gateway_routing_entry(key, json.dumps(stale), scope=store._routing_scope())
        assert key not in _durable(store), "a per-entry save restored alpha's deleted route"
        store._save_sessions_json({key: stale})
        _restart(store)
        assert key not in store._entries, "the sessions.json import restored alpha's deleted route"
        _past_the_horizon(served.launch.db)
        assert served.launch.db.get_meta(RETIRED_PREFIX + 'user-42') is not None, 'the delete shortened the marker'


@pytest.mark.asyncio
@pytest.mark.parametrize('earlier', [None, {'expires': True}], ids=['fresh', 'earlier-sweep'])
async def test_deleting_the_launch_session_writes_a_launch_only_marker(tmp_path, monkeypatch, earlier):
    """A launch delete with no earlier marker, or over an earlier release's sweep marker (only the
    launch profile's own sweep wrote one: an earlier release fenced a secondary's id permanently),
    writes the launch profile's own marker. Alpha's route to its own ``user-42`` still saves."""
    from hermes_state_mutation_retirement import RETIRED_PREFIX
    async with _served(tmp_path, monkeypatch) as served:
        store, key = served.store, 'agent:alpha:telegram:42'
        for db in (served.launch.db, served.alpha.db):
            db.create_session('user-42', source='cli')
        if earlier is not None:
            served.launch.db.set_meta(RETIRED_PREFIX + 'user-42', json.dumps({'retired_at': time.time(), **earlier}))
        served.launch.db.delete_session('user-42')
        marker = json.loads(served.launch.db.get_meta(RETIRED_PREFIX + 'user-42'))
        assert marker.get('own') is True and 'expires' not in marker, marker
        entry = _entry(key, 'user-42').to_dict()
        store._routing_db.save_gateway_routing_entry(key, json.dumps(entry), scope=store._routing_scope())
        assert key in _durable(store), "the launch delete fenced alpha's route"
        store._save_sessions_json({key: entry})
        _restart(store)
        assert store._entries[key].session_id == 'user-42'
