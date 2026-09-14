"""Local slash resolution is not another execution owner."""
import asyncio
from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path
import signal
import sqlite3
import threading
from types import SimpleNamespace

import pytest
from websockets.asyncio.client import connect
from websockets.exceptions import InvalidStatus

from tests.gateway.fixtures.local_recovery_probe import Model, daemon, rpc, websocket


@pytest.mark.linux_only
def test_ordinary_daemon_slash_skill_uses_durable_fifo(tmp_path):
    root = Path(__file__).resolve().parents[2]
    home, user = tmp_path / 'state', tmp_path / 'user'
    home.mkdir(mode=0o700)
    user.mkdir()
    peer = ThreadingHTTPServer(('127.0.0.1', 0), Model)
    peer.requests = []
    peer.blocked, peer.release = threading.Event(), threading.Event()
    thread = threading.Thread(target=peer.serve_forever, daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{peer.server_port}/v1'
    (home / 'config.yaml').write_text(json.dumps({
        'gateway': {'multiplex_profiles': False},
        'model': {'provider': 'custom', 'default': 'slash-fixture', 'base_url': url},
        'auxiliary': {'title_generation': {'enabled': False}},
        'quick_commands': {'probe-quick': {'type': 'alias', 'target': '/probe-skill'}},
    }))
    skill = home / 'skills' / 'probe-skill'
    skill.mkdir(parents=True)
    (skill / 'SKILL.md').write_text(
        '---\nname: probe-skill\ndescription: Disposable slash fixture\n---\n'
        '# Probe\nSLASH_SKILL_BODY: follow this owned fixture instruction.\n')
    env = {k: os.environ[k] for k in ('PATH', 'LANG', 'TZ', 'TIRITH_ENABLED') if k in os.environ}
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home),
               PYTHONPATH=str(root), PYTHONUNBUFFERED='1',
               OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=url,
               HERMES_DASHBOARD_SESSION_TOKEN='owned-slash-negative')
    receipts = []

    def rows():
        with sqlite3.connect(f'file:{home / "state.db"}?mode=ro', uri=True) as db:
            return list(db.execute('SELECT request_id,status,seq FROM session_admissions ORDER BY seq'))

    async def probe(desc):
        with pytest.raises(InvalidStatus):
            async with connect(desc['api_origin'].replace('http:', 'ws:') + '/api/ws'):
                pass
        async with websocket(home, desc) as ws:
            created = await rpc(ws, 'session.create', request_id='slash', source='gui', toolsets=[])
            sid = created['result']['session_id']

            async def command(method='slash.exec', **params):
                result = await rpc(ws, method, session_id=sid, **params)
                receipts.append(result)
                return result

            help_reply = await command(command='help')
            assert help_reply.get('result', {}).get('output'), help_reply
            for read_command in ('commands', 'context', 'version', 'whoami'):
                read_reply = await command(command=read_command)
                assert read_reply.get('result', {}).get('output'), read_reply
            status = await command(command='status')
            assert sid in status['result']['output'], status
            for method, params in [('slash.exec', {'command': 'probe-quick TASK_ONE'}),
                                   ('command.dispatch', {'name': 'probe-skill', 'arg': 'TASK_TWO'})]:
                result = await command(method, **params)
                assert result.get('result', {}).get('type') == 'skill', result
                assert 'SLASH_SKILL_BODY' in result['result']['message'], result
                assert 'SLASH_SKILL_BODY' not in result['result']['display'], result
            assert not peer.requests and not rows(), 'resolution executed a turn'
            await rpc(ws, 'prompt.submit', session_id=sid, input_id='warm', text='WARM_SLASH')
            async with asyncio.timeout(30):
                while not rows() or rows()[0][1] != 'terminal':
                    await asyncio.sleep(.05)
            block = await rpc(ws, 'prompt.submit', session_id=sid, input_id='block', text='BLOCK_STARTED')
            assert block.get('result', {}).get('status') == 'queued', block
            assert await asyncio.to_thread(peer.blocked.wait, 25), ([m for r in peer.requests for m in r['messages'] if m['role'] == 'user'], rows())
            for index, marker in enumerate(('TASK_ONE', 'TASK_TWO')):
                directive = await command(command='probe-quick ' + marker)
                admitted = await rpc(ws, 'prompt.submit', session_id=sid,
                                     input_id=marker, text=directive['result']['message'])
                receipts.append(admitted)
                assert admitted['result']['status'] == 'queued', admitted
                retry = await rpc(ws, 'prompt.submit', session_id=sid,
                                  input_id=marker, text=directive['result']['message'])
                assert retry['result']['admission_id'] == admitted['result']['admission_id']
            assert [r[:2] for r in rows()] == [('warm', 'terminal'), ('block', 'started'), ('TASK_ONE', 'queued'), ('TASK_TWO', 'queued')]
            busy_title = await command(command='title forbidden-while-busy')
            assert 'mid-turn' in busy_title['result']['output'], busy_title
            for params, reason in [({'command': 'help', 'profile': 'foreign'}, 'profile_mismatch'),
                                   ({'command': 'help', 'source': 'telegram'}, 'invalid_params'),
                                   ({'command': 'restart'}, 'unsupported_command'),
                                   ({'command': 'approve'}, 'unsupported_command')]:
                denied = await command(**params)
                assert denied['error']['message'] == reason, denied
            # A second dashboard operator shares access but still obeys the busy fence.
            other_url = desc['api_origin'].replace('http:', 'ws:') + '/api/ws?token=owned-slash-negative'
            async with connect(other_url) as other:
                for method, payload in [('slash.exec', {'command': 'title stolen'}),
                                        ('command.dispatch', {'name': 'title', 'arg': 'stolen'})]:
                    result = await rpc(other, method, session_id=sid, **payload)
                    receipts.append(result)
                    assert 'mid-turn' in result.get('result', {}).get('output', ''), result
            peer.release.set()
            async with asyncio.timeout(45):
                while any(row[1] != 'terminal' for row in rows()):
                    await asyncio.sleep(.05)
            title = await command(command='title Owned slash title')
            assert 'Owned slash title' in title['result']['output'], title
            listing = await rpc(ws, 'session.list')
            assert next(r for r in listing['result']['sessions'] if r['id'] == sid)['title'] == 'Owned slash title'
            snapshot = await rpc(ws, 'session.resume', session_id=sid)
            assert 'TASK_TWO' in json.dumps(snapshot['result']['messages']), snapshot
            receipts.append({'session_id': sid, 'ledger': rows(), 'history': snapshot['result']['messages']})
    try:
        with daemon(root, home, env, barrier=False) as (proc, desc):
            asyncio.run(probe(desc))
            texts = [next(m['content'] for m in reversed(r['messages']) if m['role'] == 'user') for r in peer.requests]
            assert len(texts) == 4 and texts[1] == 'BLOCK_STARTED', texts
            assert 'TASK_ONE' in texts[2] and 'TASK_TWO' in texts[3], texts
            systems = [[m for m in r['messages'] if m['role'] == 'system'] for r in peer.requests]
            assert systems[1] == systems[2] == systems[3], 'slash invalidated cached system prefix'
            proc.send_signal(signal.SIGINT)
            assert proc.wait(timeout=20) == 0
            print(json.dumps({'pid': proc.pid, 'exit': proc.returncode, 'descriptor': desc,
                              'wire_receipts': receipts, 'model_texts': texts,
                              'system_prefix_unchanged': True, 'unauthenticated_ws_rejected': True}))
    finally:
        peer.release.set()
        peer.shutdown()
        peer.server_close()
        thread.join(timeout=5)


@pytest.mark.asyncio
async def test_command_authority_precedes_resolution_and_mutation(tmp_path, monkeypatch):
    from gateway.config import Platform
    from gateway.session import SessionSource
    from gateway.session_authority import SessionAuthority, LiveSession
    from gateway.session_controls import AuthorityConnection
    from agent import skill_commands

    runner = SimpleNamespace(_draining=False)
    authority = SessionAuthority(runner, profile_id=str(tmp_path), instance_id='owned', db=None, epoch=1)
    source = SessionSource(platform=Platform.LOCAL, chat_id='local-test', user_id='owner')
    authority.sessions['local-test'] = LiveSession(source, 'route')
    def forbidden(*args, **kwargs):
        pytest.fail('unauthorized command reached skill resolution')
    monkeypatch.setattr(skill_commands, 'get_skill_commands', forbidden)
    for identity, command, reason in [
        ({'user_id': 'owner', 'capabilities': ['session:read']}, 'title mutated', 'permission_denied'),
        ({'user_id': 'owner', 'capabilities': ['session:read']}, 'private-skill', 'permission_denied'),
        ({'user_id': 'other'}, 'help', 'permission_denied'),
        ({'user_id': 'owner', 'profile_id': 'foreign'}, 'help', 'profile_mismatch'),
        ({'user_id': 'owner', 'instance_id': 'stale'}, 'help', 'permission_denied'),
    ]:
        connection = AuthorityConnection(authority, object(), identity)
        for method, payload in [('slash.exec', {'command': command}),
                                ('command.dispatch', {'name': command.split()[0], 'arg': 'mutated'})]:
            result = await connection.dispatch({'id': 1, 'method': method,
                'params': {'session_id': 'local-test', **payload}})
            assert result.get('error', {}).get('message') == reason, result
        await connection.close()
