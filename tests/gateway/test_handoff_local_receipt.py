"""A local chat handed off to a messaging lane keeps opening there (#143286).

``/handoff`` repoints the platform's route at the local session (``SessionStore.switch_session``),
which rewrites the row's routing identity. Its local receipt must go with that write, or the row
matches neither lane: every open, pin or rename refused ``storage_unavailable`` (``stored identity
mismatch``) and its title (a Bot Chat's registry lookup) answered ``not_found``. A store an older
build left that way heals when the gateway starts.
"""
import json
from types import SimpleNamespace

import pytest

from tests.gateway.test_authority_cold_route_recovery import _Runner, _source


@pytest.fixture(autouse=True)
def _home(tmp_path, monkeypatch):
    from gateway import run
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(run, '_load_gateway_config', lambda *_path: {'platform_toolsets': {'cli': []}, 'model': {'default': 'fixture'}})


def _compress(authority, sid):
    """Two real compression splits of the local chat, as its turns publish them."""
    db, live = authority.db, authority.sessions[sid]
    store = authority.runner.session_store
    for parent, child in ((sid, sid + '-c1'), (sid + '-c1', sid + '-c2')):
        store._record_gateway_session_peer(parent, live.route, live.source)
        assert db.try_acquire_compression_lock(parent, 'fixture')
        db.publish_compression_child(parent_session_id=parent, child_session_id=child, source='local',
                                     messages=[{'role': 'user', 'content': 'kept'}], compression_lock_holder='fixture')
        db.release_compression_lock(parent, 'fixture')
        store._record_gateway_session_peer(child, live.route, live.source)


async def _handoff(runner, authority, sid, source):
    """The handoff watcher's repoint (``GatewayStartupMixin._handoff_repoint``) on the real store."""
    from gateway.run_startup import GatewayStartupMixin
    from gateway.session import AsyncSessionStore
    from hermes_state import AsyncSessionDB
    route = runner.session_store._generate_session_key(source)
    gateway = SimpleNamespace(_session_db=AsyncSessionDB(authority.db), session_authority=authority,
                              async_session_store=AsyncSessionStore(runner.session_store))
    await GatewayStartupMixin._handoff_repoint(gateway, SimpleNamespace(source=source), route, sid)
    return route


@pytest.mark.asyncio
@pytest.mark.parametrize('history', ['plain', 'compressed', 'reset'])
@pytest.mark.parametrize('handed_off_by', ['this build', 'an older build'])
async def test_a_handed_off_bot_chat_opens_on_its_lane_after_a_restart(tmp_path, handed_off_by, history):
    """``reset``: a ``/new`` left the chat's first segment behind and its receipt names the new one;
    the conversation's own id (what a platform ``/resume <title>`` or a resumed CLI hands off) moves."""
    from gateway.config import Platform
    from gateway.session_authority import initialize_session_authority
    from gateway.session_contract import Principal
    from gateway.session_local import create_local_session
    from gateway.session_local_recovery import transcript_target
    from gateway.session_local_title import resolve_titled_session
    from hermes_state_local import POLICY_PREFIX
    from hermes_state_local_migration import LEGACY_PREFIX

    runner = _Runner(tmp_path)
    warm = await initialize_session_authority(runner, profile_id='fixture', instance_id='warm')
    owner = Principal('owner', 'fixture', frozenset({'session:create', 'session:read', 'session:control'}), 'socket')
    ref = create_local_session(warm, owner, {'request_id': 'bot-chat', 'source': 'gui', 'cwd': str(tmp_path),
                                             'model': 'frozen', 'toolsets': []})
    warm.db.set_session_title(ref.session_id, 'Bot Chat')
    if history == 'reset':
        runner.session_store.reset_session(warm.sessions[ref.session_id].route)
    if history == 'compressed':
        _compress(warm, ref.session_id)
        runner = _Runner(tmp_path)  # a restart records the steps on the receipt (#140060)
        warm = await initialize_session_authority(runner, profile_id='fixture', instance_id='adopted')
        warm.authorize(owner, ref, 'session:read')  # the receipt now names the tip
    keys = (POLICY_PREFIX + ref.session_id, LEGACY_PREFIX + ref.session_id)
    with warm.db._read_ctx() as conn:
        receipt = conn.execute('SELECT key, value FROM state_meta WHERE key IN (?,?)', keys).fetchall()
    assert receipt

    telegram = _source()
    route = await _handoff(runner, warm, ref.session_id, telegram)
    if handed_off_by == 'an older build':
        warm.db._execute_write(lambda conn: conn.executemany(
            'INSERT OR REPLACE INTO state_meta(key, value) VALUES(?,?)', [tuple(row) for row in receipt]))
    else:
        # The Desktop that ran /handoff still holds the chat open.
        warm.authorize(owner, ref, 'session:control')
        assert transcript_target(warm, ref) == warm.physical_target(ref)

    cold = await initialize_session_authority(_Runner(tmp_path), profile_id='fixture', instance_id='cold')
    # A browser Webapp viewer: never a native legacy transport, so legacy adoption is not its way in.
    webapp = Principal('webapp', 'fixture', frozenset({'session:read', 'session:control'}), 'browser')
    assert resolve_titled_session(cold, webapp, 'Bot Chat') == ref
    cold.authorize(webapp, ref, 'session:control')  # pin, rename, archive

    live = cold.sessions[ref.session_id]
    assert (live.source.platform, live.route) == (Platform.TELEGRAM, route)
    with cold.db._read_ctx() as conn:
        assert conn.execute('SELECT 1 FROM state_meta WHERE key IN (?,?)', keys).fetchone() is None
    tip = cold.db.get_session(cold.physical_target(ref))
    assert json.loads(tip['origin_json'])['platform'] == 'telegram'


async def _handed_off_bot_chat(tmp_path):
    """A Bot Chat the Desktop minted and a ``/handoff`` moved to Telegram, still open in that Desktop."""
    from gateway.session_authority import initialize_session_authority
    from gateway.session_contract import Principal
    from gateway.session_local import create_local_session
    runner = _Runner(tmp_path)
    warm = await initialize_session_authority(runner, profile_id='fixture', instance_id='warm')
    owner = Principal('owner', 'fixture', frozenset({'session:create', 'session:read', 'session:control',
                                                     'session:submit'}), 'socket')
    ref = create_local_session(warm, owner, {'request_id': 'bot-chat', 'source': 'gui', 'cwd': str(tmp_path),
                                             'model': 'frozen', 'toolsets': []})
    warm.db.set_session_title(ref.session_id, 'Bot Chat')
    telegram = _source()
    route = await _handoff(runner, warm, ref.session_id, telegram)
    warm.authorize(owner, ref, 'session:control')
    return runner, warm, owner, ref, telegram, route


@pytest.mark.asyncio
async def test_a_handed_off_bot_chat_stays_the_bots_chat_while_its_lane_holds_it(tmp_path):
    """Bot Mode's one canonical chat per profile: a Bot or cron delivery after the restart goes to
    the handed-off Bot Chat on its lane; it is never renamed aside for a freshly minted one."""
    from gateway.session_authority import initialize_session_authority
    from gateway.session_bot import _target, displaceable_bot_chat
    from gateway.session_local_title import resolve_titled_session
    _runner, _warm, owner, ref, _telegram, route = await _handed_off_bot_chat(tmp_path)
    cold = await initialize_session_authority(_Runner(tmp_path), profile_id='fixture', instance_id='cold')
    assert displaceable_bot_chat(cold, cold.db.get_session_by_title('Bot Chat')) is None
    target, live, entry = await _target(cold, owner)
    assert target == ref and live.route == route == entry.session_key
    assert resolve_titled_session(cold, owner, 'Bot Chat') == ref
    assert cold.db.get_session(ref.session_id)['title'] == 'Bot Chat'


@pytest.mark.asyncio
@pytest.mark.parametrize('door', ['Desktop create', 'Bot delivery'])
async def test_a_handed_off_bot_chat_whose_lane_moved_on_is_moved_aside(tmp_path, door):
    """After the platform's own ``/new`` the handed-off chat is that lane's history: the Desktop
    still viewing it is read-only (its route names the new conversation), and after a restart the
    bot row no longer adopts it: it is renamed aside and a fresh Bot Chat is minted."""
    from gateway.session_authority import initialize_session_authority
    from gateway.session_bot import _target
    from gateway.session_controls import AuthorityConnection
    from gateway.session_local_title import resolve_titled_session
    from hermes_state_runtime import RuntimeStoreError
    runner, warm, owner, ref, telegram, route = await _handed_off_bot_chat(tmp_path)
    runner.session_store.reset_session(route)  # Telegram's /new
    newer = warm.register(telegram)
    assert newer != ref and warm.sessions[newer.session_id].route == route
    assert warm.sessions[ref.session_id].superseded and warm.agent(ref) is None
    with pytest.raises(RuntimeStoreError, match='session_superseded'):
        warm.authorize(owner, ref, 'session:control')

    cold_runner = _Runner(tmp_path)
    cold = await initialize_session_authority(cold_runner, profile_id='fixture', instance_id='cold')
    cold_runner._adapter_for_source = lambda source: cold_runner.adapters.get(source.platform)
    cold_runner._cached_agent_for = lambda route: None
    assert resolve_titled_session(cold, owner, 'Bot Chat', missing_ok=True) is None
    if door == 'Desktop create':
        connection = AuthorityConnection(cold, object(), {'user_id': owner.subject})
        opened = await connection.create(None, {'title': 'Bot Chat', 'request_id': 'bot-chat-2', 'source': 'gui',
                                                'cwd': str(tmp_path), 'model': 'frozen', 'toolsets': [], 'hidden': True})
        fresh = opened['session_id']
    else:
        fresh = (await _target(cold, owner))[0].session_id
    assert fresh not in (ref.session_id, newer.session_id)
    assert cold.db.get_session_by_title('Bot Chat')['id'] == fresh
    moved = cold.db.get_session(ref.session_id)
    assert moved['title'] == f'Bot Chat (telegram {ref.session_id})' and not moved['hidden']


def _serve(tmp_path, monkeypatch):
    """A multiplexed gateway's homes: the default profile and a named ``bot`` profile, each with its
    own ``state.db``; ``bot`` delivers to a Telegram home channel."""
    from pathlib import Path
    import hermes_state
    root = tmp_path / '.hermes'
    bot = root / 'profiles' / 'bot'
    bot.mkdir(parents=True, mode=0o700)
    (bot / 'config.yaml').write_text(
        'platforms:\n  telegram:\n    enabled: true\n    token: fixture\n'
        '    home_channel: {platform: telegram, chat_id: "-1001", name: Bot Home}\n')
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)  # the profiles root is HOME-anchored
    monkeypatch.setenv('HERMES_HOME', str(root))
    monkeypatch.setattr(hermes_state, 'DEFAULT_DB_PATH', hermes_state._IMPORT_DEFAULT_DB_PATH)
    return root, bot


async def _boot(root, bot):
    """One gateway process: a real runner whose runtime builds one authority per served home."""
    from gateway.config import GatewayConfig
    from gateway.run import GatewayRunner
    from gateway.run_runtime import initialize_gateway_runtime
    config = GatewayConfig()
    config.multiplex_profiles = True
    config._runtime_profile_homes = (('default', root), ('bot', bot))
    runner = GatewayRunner(config)
    await initialize_gateway_runtime(runner)
    return runner


def _shutdown(runner):
    for authority in list(runner.session_authorities):
        authority.db.close()
    runner.session_store.close_all_db_handles()
    runner.close_all_session_db_handles()


@pytest.mark.asyncio
@pytest.mark.parametrize('handed_off_by', ['this build', 'an older build'])
async def test_a_named_profiles_handed_off_bot_chat_opens_on_its_lane_after_a_restart(
        tmp_path, monkeypatch, handed_off_by):
    """The reporter's topology: a named bot profile under a multiplexed gateway. The chat lives in
    that profile's own store, the handoff watcher's ``_process_handoff`` runs in its scope and keys
    the lane ``agent:bot:...``; after a restart the Webapp still finds and opens it there."""
    import sqlite3
    from unittest.mock import AsyncMock
    from gateway.config import Platform
    from gateway.run import _async_profile_runtime_scope
    from gateway.runtime_ownership import process_ownership
    from gateway.session_authorities import active_authority
    from gateway.session_contract import Principal
    from gateway.session_local import create_local_session
    from gateway.session_local_recovery import transcript_target
    from gateway.session_local_title import resolve_titled_session
    from hermes_state_local import POLICY_PREFIX
    from hermes_state_local_migration import LEGACY_PREFIX

    root, bot = _serve(tmp_path, monkeypatch)
    process_ownership.reserve([root, bot])
    try:
        runner = await _boot(root, bot)
        runner._profile_adapters = {'bot': {Platform.TELEGRAM: SimpleNamespace(
            create_handoff_thread=AsyncMock(return_value=None))}}
        runner._handle_message = AsyncMock(return_value='')  # the synthetic handoff turn itself
        async with _async_profile_runtime_scope(bot):
            warm = active_authority(runner)
            caps = frozenset({'session:create', 'session:read', 'session:control'})
            owner = Principal('owner', warm.profile_id, caps, 'socket')
            ref = create_local_session(warm, owner, {'request_id': 'bot-chat', 'source': 'gui', 'cwd': str(tmp_path),
                                                     'model': 'frozen', 'toolsets': []})
            warm.db.set_session_title(ref.session_id, 'Bot Chat')
            keys = (POLICY_PREFIX + ref.session_id, LEGACY_PREFIX + ref.session_id)
            with warm.db._read_ctx() as conn:
                receipt = conn.execute('SELECT key, value FROM state_meta WHERE key IN (?,?)', keys).fetchall()
            assert receipt
            await runner._process_handoff({'id': ref.session_id, 'title': 'Bot Chat', 'handoff_platform': 'telegram'},
                                          'bot')
            if handed_off_by == 'an older build':
                warm.db._execute_write(lambda conn: conn.executemany(
                    'INSERT OR REPLACE INTO state_meta(key, value) VALUES(?,?)', [tuple(row) for row in receipt]))
            else:
                warm.authorize(owner, ref, 'session:control')  # the Desktop that ran /handoff
                assert transcript_target(warm, ref) == warm.physical_target(ref)
        _shutdown(runner)

        runner = await _boot(root, bot)
        async with _async_profile_runtime_scope(bot):
            cold = active_authority(runner)
            webapp = Principal('webapp', cold.profile_id, frozenset({'session:read', 'session:control'}), 'browser')
            assert resolve_titled_session(cold, webapp, 'Bot Chat') == ref
            cold.authorize(webapp, ref, 'session:control')  # pin, rename, archive
            live = cold.sessions[ref.session_id]
            assert live.source.platform == Platform.TELEGRAM and live.route.startswith('agent:bot:telegram:')
            with cold.db._read_ctx() as conn:
                assert conn.execute('SELECT 1 FROM state_meta WHERE key IN (?,?)', keys).fetchone() is None
        _shutdown(runner)
        with sqlite3.connect(root / 'state.db') as conn:  # never the default profile's store
            assert conn.execute('SELECT 1 FROM sessions WHERE id=?', (ref.session_id,)).fetchone() is None
    finally:
        for home in (root, bot):
            process_ownership.release(home)
