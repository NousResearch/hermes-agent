"""A local chat handed off to a messaging lane keeps opening there (#143286).

``/handoff`` repoints the platform's route at the local session (``SessionStore.switch_session``),
which rewrites the row's routing identity. Its local receipt must go with that write, or the row
matches neither lane: every open, pin or rename refused ``storage_unavailable`` (``stored identity
mismatch``) and its title (a Bot Chat's registry lookup) answered ``not_found``. A store an older
build left that way heals when the gateway starts.
"""
import json

import pytest

from tests.gateway.test_authority_cold_route_recovery import _Runner, _source


@pytest.fixture(autouse=True)
def _home(tmp_path, monkeypatch):
    from gateway import run
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'platform_toolsets': {'cli': []}})


@pytest.mark.asyncio
@pytest.mark.parametrize('handed_off_by', ['this build', 'an older build'])
async def test_a_handed_off_bot_chat_opens_on_its_lane_after_a_restart(tmp_path, handed_off_by):
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
    keys = (POLICY_PREFIX + ref.session_id, LEGACY_PREFIX + ref.session_id)
    with warm.db._read_ctx() as conn:
        receipt = conn.execute('SELECT key, value FROM state_meta WHERE key IN (?,?)', keys).fetchall()
    assert receipt

    # The handoff watcher's repoint (gateway/run_startup.py ``_process_handoff``), then its synthetic
    # turn's admission publishing the lane while the Desktop still holds the chat open.
    telegram = _source()
    runner.session_store.get_or_create_session(telegram)
    route = runner.session_store._generate_session_key(telegram)
    assert runner.session_store.switch_session(route, ref.session_id, preserve_prompt_pin=False)
    if handed_off_by == 'an older build':
        warm.db._execute_write(lambda conn: conn.executemany(
            'INSERT OR REPLACE INTO state_meta(key, value) VALUES(?,?)', [tuple(row) for row in receipt]))
    else:
        assert warm.register(telegram) == ref
        warm.authorize(owner, ref, 'session:control')
        assert transcript_target(warm, ref) == ref.session_id

    cold = await initialize_session_authority(_Runner(tmp_path), profile_id='fixture', instance_id='cold')
    # A browser Webapp viewer: never a native legacy transport, so legacy adoption is not its way in.
    webapp = Principal('webapp', 'fixture', frozenset({'session:read', 'session:control'}), 'browser')
    assert resolve_titled_session(cold, webapp, 'Bot Chat') == ref
    cold.authorize(webapp, ref, 'session:control')  # pin, rename, archive

    live = cold.sessions[ref.session_id]
    assert (live.source.platform, live.route) == (Platform.TELEGRAM, route)
    with cold.db._read_ctx() as conn:
        assert conn.execute('SELECT 1 FROM state_meta WHERE key IN (?,?)', keys).fetchone() is None
    assert json.loads(cold.db.get_session(ref.session_id)['origin_json'])['platform'] == 'telegram'
