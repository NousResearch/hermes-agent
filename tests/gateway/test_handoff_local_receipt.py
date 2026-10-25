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
    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'platform_toolsets': {'cli': []}})


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
@pytest.mark.parametrize('compressed', [False, True], ids=['plain', 'compressed'])
@pytest.mark.parametrize('handed_off_by', ['this build', 'an older build'])
async def test_a_handed_off_bot_chat_opens_on_its_lane_after_a_restart(tmp_path, handed_off_by, compressed):
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
    if compressed:
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
