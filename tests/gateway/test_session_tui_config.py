"""The TUI's profile preferences (/statusbar /mouse /density /details /focus /battery /theme
/indicator) are written by the shared owner into the OWNING profile's config.yaml, like main's
per-TUI sidecar ``config.set`` — never refused, never inference."""
import asyncio
import json

import hermes_yaml as yaml
import pytest

from tests.gateway.test_session_busy_controls import owner, admissions, settled
from tests.gateway.fixtures.local_recovery_probe import rpc, websocket


def _display(home):
    return yaml.safe_load((home / 'config.yaml').read_text()).get('display', {})


@pytest.mark.platforms("linux")
def test_display_preferences_theme_and_indicator_persist_to_the_profile(tmp_path):
    with owner(tmp_path) as (home, peer, desc):
        async def probe():
            async with websocket(home, desc) as ws:
                sid = (await rpc(ws, 'session.create', request_id='prefs', source='tui',
                                 toolsets=[]))['result']['session_id']
                # Main's TUI sends display preferences without a session (profile-wide).
                for key, value, reported in [
                    ('statusbar', 'off', 'off'), ('mouse', 'wheel', 'wheel'), ('density', 'on', 'on'),
                    ('battery', 'on', 'on'), ('details_mode', 'expanded', 'expanded'),
                    ('details_mode.tools', 'hidden', 'hidden'), ('theme', 'dark', 'dark'),
                    ('indicator', 'ascii', 'ascii'),
                ]:
                    result = await rpc(ws, 'config.set', key=key, value=value)
                    assert result.get('result') == {'key': key, 'value': reported}, (key, result)
                focus = await rpc(ws, 'config.set', key='focus', value='on', session_id=sid)
                assert focus.get('result') == {'key': 'focus', 'value': 'on', 'tool_progress': 'off'}, focus
                display = _display(home)
                assert display['tui_statusbar'] == 'off' and display['mouse_tracking'] == 'wheel'
                assert display['tui_compact'] is True and display['battery'] is True
                assert display['details_mode'] == 'expanded'
                assert display['sections'] == {'thinking': 'expanded', 'tools': 'hidden',
                                               'subagents': 'expanded', 'activity': 'expanded'}
                assert display['tui_theme'] == 'dark' and display['tui_status_indicator'] == 'ascii'
                assert display['focus_view'] is True and display['tool_progress'] == 'off'
                # /focus also pins this session's tool-progress mode, as main's session did.
                assert (await rpc(ws, 'config.get', key='verbose', session_id=sid))['result']['value'] == 'off'
                assert (await rpc(ws, 'config.get', key='theme'))['result']['value'] == 'dark'
                assert (await rpc(ws, 'config.get', key='indicator'))['result']['value'] == 'ascii'
                for params, reason in [
                    ({'key': 'theme', 'value': 'neon'}, 'invalid_params'),
                    ({'key': 'indicator', 'value': 'sparkles'}, 'invalid_params'),
                    ({'key': 'details_mode.nope', 'value': 'hidden'}, 'invalid_params'),
                    ({'key': 'theme', 'value': 'dark', 'profile': 'foreign'}, 'profile_mismatch'),
                    ({'key': 'model', 'value': 'x', 'session_id': sid}, 'use_session_mutation_model'),
                ]:
                    assert (await rpc(ws, 'config.set', **params)).get('error', {}).get('message') == reason, params
                assert _display(home)['tui_theme'] == 'dark'
                assert not admissions(home) and not peer.requests
        asyncio.run(probe())


async def _rpc_and_events(ws, method, **params):
    """``rpc`` that keeps the event frames delivered before the reply (``rpc`` drops them)."""
    await ws.send(json.dumps({'jsonrpc': '2.0', 'id': method, 'method': method, 'params': params}))
    events = []
    async with asyncio.timeout(20):
        while True:
            frame = json.loads(await ws.recv())
            if frame.get('id') == method:
                return frame, events
            if frame.get('method') == 'event':
                events.append(frame['params'])


async def _skin_event(ws, sid, seen=()):
    """The session's own frame: the owner's repaint of THIS profile (the process-wide change
    watcher is session-less and only watches the launch profile)."""
    for params in seen:
        if params['type'] == 'skin.changed' and params['session_id'] == sid:
            return params['payload']
    async with asyncio.timeout(20):
        while True:
            frame = json.loads(await ws.recv())
            if (frame.get('method') == 'event' and frame['params']['type'] == 'skin.changed'
                    and frame['params']['session_id'] == sid):
                return frame['params']['payload']


@pytest.mark.platforms("linux")
def test_skin_persists_and_repaints_every_attached_tui_of_the_profile(tmp_path):
    with owner(tmp_path) as (home, peer, desc):
        async def probe():
            async with websocket(home, desc) as ws, websocket(home, desc) as other:
                sid = (await rpc(ws, 'session.create', request_id='skin', source='tui',
                                 toolsets=[]))['result']['session_id']
                peer_sid = (await rpc(other, 'session.create', request_id='skin-peer', source='tui',
                                      toolsets=[]))['result']['session_id']
                result, seen = await _rpc_and_events(ws, 'config.set', key='skin', value='mono')
                assert result.get('result') == {'key': 'skin', 'value': 'mono'}, result
                # Main broadcast the resolved skin to every surface of the profile.
                for socket, session, frames in ((ws, sid, seen), (other, peer_sid, ())):
                    payload = await _skin_event(socket, session, frames)
                    assert payload['name'] == 'mono' and payload['colors'], payload
                assert _display(home)['skin'] == 'mono'
                assert (await rpc(ws, 'config.get', key='skin'))['result']['value'] == 'mono'
                assert (await rpc(ws, 'config.set', key='skin', value='')).get('error', {}).get('message') == 'invalid_params'
                assert not admissions(home) and not peer.requests
        asyncio.run(probe())


@pytest.mark.platforms("linux")
def test_fast_is_session_scoped_validated_and_read_back(tmp_path):
    with owner(tmp_path) as (home, peer, desc):
        async def probe():
            async with websocket(home, desc) as ws:
                # A route that bills Priority Processing (never called: no turn runs).
                sid = (await rpc(ws, 'session.create', request_id='fast', source='tui', toolsets=[],
                                 model='gpt-5.4', provider='openai',
                                 base_url='https://api.openai.com/v1'))['result']['session_id']
                other = (await rpc(ws, 'session.create', request_id='slow', source='tui',
                                   toolsets=[]))['result']['session_id']
                before = (home / 'config.yaml').read_bytes()
                result = await rpc(ws, 'config.set', key='fast', value='fast', session_id=sid)
                assert result.get('result') == {'key': 'fast', 'value': 'fast'}, result
                assert (await rpc(ws, 'config.get', key='fast', session_id=sid))['result']['value'] == 'fast'
                assert (await rpc(ws, 'config.get', key='fast', session_id=other))['result']['value'] == 'normal'
                assert (await rpc(ws, 'config.set', key='fast', value='toggle', session_id=sid)
                        )['result']['value'] == 'normal'
                assert (await rpc(ws, 'config.get', key='fast', session_id=sid))['result']['value'] == 'normal'
                for params, reason in [
                    ({'key': 'fast', 'value': 'fast', 'session_id': other}, 'fast_unavailable'),
                    ({'key': 'fast', 'value': 'warp', 'session_id': sid}, 'invalid_params'),
                    ({'key': 'fast', 'value': 'fast'}, 'invalid_params'),
                ]:
                    assert (await rpc(ws, 'config.set', **params)).get('error', {}).get('message') == reason, params
                # Session-scoped like main: config.yaml is untouched.
                assert (home / 'config.yaml').read_bytes() == before
                assert not admissions(home) and not peer.requests
        asyncio.run(probe())


@pytest.mark.platforms("linux")
def test_personality_persists_to_the_profile_and_drives_the_next_turn(tmp_path):
    with owner(tmp_path) as (home, peer, desc):
        async def probe():
            async with websocket(home, desc) as ws:
                sid = (await rpc(ws, 'session.create', request_id='persona', source='tui',
                                 toolsets=[]))['result']['session_id']
                await rpc(ws, 'prompt.submit', session_id=sid, input_id='before', text='BEFORE')
                await settled(home)
                result = await rpc(ws, 'config.set', key='personality', value='pirate', session_id=sid)
                assert result.get('result') == {'key': 'personality', 'value': 'pirate',
                                                'history_reset': False}, result
                assert _display(home)['personality'] == 'pirate'
                await rpc(ws, 'prompt.submit', session_id=sid, input_id='after', text='AFTER')
                await settled(home)
                systems = [' '.join(str(m.get('content')) for m in body['messages'] if m['role'] == 'system')
                           for body in peer.requests]
                assert 'Captain Hermes' not in systems[0] and 'Captain Hermes' in systems[-1], systems
                assert (await rpc(ws, 'config.set', key='personality', value='nobody', session_id=sid)
                        ).get('error', {}).get('message') == 'unknown_personality'
                result = await rpc(ws, 'config.set', key='personality', value='none', session_id=sid)
                assert result.get('result', {}).get('value') == 'none', result
                assert _display(home)['personality'] == ''
        asyncio.run(probe())
