"""Actual native-ticket WS creation/resume stays bound to each served profile."""
import asyncio
import difflib
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import sqlite3
import threading

from tests.gateway.fixtures.local_recovery_probe import Model, child_env, daemon, rpc, websocket


def test_native_gui_plumbing_creation_and_idle_refresh_in_two_served_homes(tmp_path):
    root = Path(__file__).resolve().parents[2]
    home, user = tmp_path / 'state', tmp_path / 'user'
    beta = home / 'profiles' / 'beta'
    beta.mkdir(parents=True)
    user.mkdir()
    (beta / 'profile.yaml').write_text('name: beta\n')
    peer = ThreadingHTTPServer(('127.0.0.1', 0), Model)
    peer.requests = []
    peer.blocked, peer.release = threading.Event(), threading.Event()
    threading.Thread(target=peer.serve_forever, daemon=True).start()
    url = f'http://127.0.0.1:{peer.server_port}/v1'
    def config(path, model):
        value = {'gateway': {'multiplex_profiles': True}, 'model': {'provider': 'custom', 'default': model, 'base_url': url},
                 'platform_toolsets': {'cli': []}, 'auxiliary': {'title_generation': {'enabled': False}}}
        temporary = path / 'fixture-config.yaml'
        temporary.write_text(json.dumps(value)); temporary.replace(path / 'config.yaml')
    config(home, 'launch-before'); config(beta, 'beta-before')
    env = child_env() | dict(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(root),
        OPENAI_API_KEY='fixture-only', OPENAI_BASE_URL=url, HERMES_DISABLE_LAZY_INSTALLS='1')
    async def exercise(desc):
        async with websocket(home, desc) as ws:
            described = await rpc(ws, 'runtime.describe')
            assert 'room_plumbing' in described['result']['session_create']['parameters']
            sessions = {}
            for profile in ('default', 'beta'):
                for invalid in ({'room_plumbing': 'true'}, {'source': 'cli'}, {'model': 'override'}, {'hidden': False}):
                    refused = await rpc(ws, 'session.create', **({'profile': profile, 'source': 'gui',
                        'hidden': True, 'title': 'Group: rejected', 'room_plumbing': True} | invalid))
                    assert refused.get('error', {}).get('data', {}).get('reason') == 'invalid_params', refused
                created = await rpc(ws, 'session.create', profile=profile, source='gui', title='Group: owned · thread',
                    hidden=True, room_plumbing=True, follow_profile_config=True, request_id='same-request')
                assert 'result' in created, created
                sessions[profile] = created['result']['session_id']
                expected = 'launch-before' if profile == 'default' else 'beta-before'
                assert created['result']['info']['model'] == expected
                resumed = await rpc(ws, 'session.resume', profile=profile, title='Group: owned · thread')
                assert resumed['result']['session_id'] == sessions[profile]
            assert len(set(sessions.values())) == 2
            sid = sessions['beta']
            submitted = await rpc(ws, 'prompt.submit', profile='beta', session_id=sid,
                submission_id='first', text='BLOCK_STARTED')
            assert 'result' in submitted, submitted
            assert await asyncio.to_thread(peer.blocked.wait, 20)
            config(beta, 'beta-after')
            busy = await rpc(ws, 'session.resume', profile='beta', session_id=sid)
            assert busy['result']['running'] and busy['result']['info']['model'] == 'beta-before'
            with sqlite3.connect(beta / 'state.db') as db:
                before = db.execute('SELECT COALESCE(p.prompt,s.system_prompt) FROM sessions s LEFT JOIN system_prompts p ON p.hash=s.system_prompt_hash WHERE s.id=?', (sid,)).fetchone()[0]
            peer.release.set()
            async with asyncio.timeout(25):
                while True:
                    resumed = await rpc(ws, 'session.resume', profile='beta', title='Group: owned · thread')
                    if not resumed['result']['running'] and not resumed['result']['pending']:
                        break
                    await asyncio.sleep(.05)
            assert resumed['result']['info']['model'] == 'beta-after', json.dumps(resumed)
            with sqlite3.connect(beta / 'state.db') as db:
                assert db.execute('SELECT COALESCE(p.prompt,s.system_prompt) FROM sessions s LEFT JOIN system_prompts p ON p.hash=s.system_prompt_hash WHERE s.id=?', (sid,)).fetchone()[0] == before
            async def turn(profile, text, expected_model):
                reply = await rpc(ws, 'prompt.submit', profile=profile, session_id=sessions[profile],
                                  submission_id=text, text=text)
                assert 'result' in reply, reply
                async with asyncio.timeout(25):
                    while True:
                        resumed = await rpc(ws, 'session.resume', profile=profile, session_id=sessions[profile])
                        if not resumed['result']['running'] and not resumed['result']['pending']:
                            break
                        await asyncio.sleep(.05)
                assert any(m.get('content') == 'RECOVERY_ACK_' + text for m in resumed['result']['messages'])
                request = next(r for r in peer.requests if any(m.get('content') == text for m in r['messages']))
                assert request['model'] == expected_model
                return request
            refreshed = await turn('beta', 'BETA_AFTER', 'beta-after')
            await turn('default', 'LAUNCH_UNCHANGED', 'launch-before')
            again = await turn('beta', 'BETA_AGAIN', 'beta-after')
            first = next(r for r in peer.requests if any(m.get('content') == 'BLOCK_STARTED' for m in r['messages']))
            first_system = '\n'.join(m['content'] for m in first['messages'] if m['role'] == 'system')
            refreshed_system = '\n'.join(m['content'] for m in refreshed['messages'] if m['role'] == 'system')
            assert first_system.count('\nModel: beta-before\n') == 1
            expected_system = first_system.replace('\nModel: beta-before\n', '\nModel: beta-after\n', 1)
            assert expected_system == refreshed_system, '\n'.join(difflib.unified_diff(expected_system.splitlines(), refreshed_system.splitlines()))
            assert [m for m in refreshed['messages'] if m['role'] == 'system'] == [m for m in again['messages'] if m['role'] == 'system']
            launch = await rpc(ws, 'session.resume', profile='default', session_id=sessions['default'])
            assert launch['result']['info']['model'] == 'launch-before'
            wrong = await rpc(ws, 'session.resume', profile='default', session_id=sid)
            assert 'error' in wrong
            for profile, path in [('default', home), ('beta', beta)]:
                with sqlite3.connect(path / 'state.db') as db:
                    own = db.execute("SELECT id,hidden FROM sessions WHERE title='Group: owned · thread'").fetchone()
                    assert own == (sessions[profile], 1)
    try:
        with daemon(root, home, env, barrier=False) as (_, desc):
            asyncio.run(exercise(desc))
    finally:
        peer.release.set(); peer.shutdown(); peer.server_close()
