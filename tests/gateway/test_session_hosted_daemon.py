"""Real native socket and canonical room member inference in a disposable daemon."""
import asyncio
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading

from tests.gateway.fixtures.local_recovery_probe import child_env, daemon, rpc, websocket


class RoomModel(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        if body.get('messages'):
            self.server.requests.append(body)
        text = next((m.get('content', '') for m in reversed(body.get('messages', [])) if m['role'] == 'user'), '')
        if 'BLOCK_HOSTED' in str(text):
            self.server.blocked.set()
            self.server.release.wait(60)
        message = {'role': 'assistant', 'content': 'PASS'}
        data = {'id': 'fixture', 'choices': [{'index': 0, 'message': message, 'finish_reason': 'stop'}],
                'usage': {'prompt_tokens': 10, 'completion_tokens': 1, 'total_tokens': 11}}
        payload, kind = json.dumps(data).encode(), 'application/json'
        if body.get('stream'):
            payload = ('data: ' + json.dumps({'id': 'fixture', 'choices': [{'index': 0, 'delta': message, 'finish_reason': 'stop'}]}) + '\n\ndata: [DONE]\n\n').encode()
            kind = 'text/event-stream'
        self.send_response(200)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        try:
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass


def test_native_room_service_local_member_and_restart(tmp_path):
    root = Path(__file__).resolve().parents[2]
    home, user = tmp_path / 'state', tmp_path / 'user'
    home.mkdir(mode=0o700); user.mkdir()
    target = home / 'profiles' / 'two'
    target.mkdir(parents=True)
    model = ThreadingHTTPServer(('127.0.0.1', 0), RoomModel)
    model.requests = []
    threading.Thread(target=model.serve_forever, daemon=True).start()
    base = f'http://127.0.0.1:{model.server_port}/v1'
    (home / 'config.yaml').write_text(json.dumps({
        'gateway': {'multiplex_profiles': False}, 'hosted_rooms': {'profiles': {'two': str(target)}},
        'model': {'provider': 'custom', 'default': 'loopback-room', 'base_url': base},
        'platform_toolsets': {'gui': [], 'bot_room': []},
        'auxiliary': {'title_generation': {'enabled': False}},
        'terminal': {'cwd': str(home)},
    }))
    (target / 'config.yaml').write_text((home / 'config.yaml').read_text())
    env = child_env() | dict(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home),
        PYTHONPATH=str(root), OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=base, PYTHONUNBUFFERED='1')
    members = [{'member_id': 'one', 'profile': 'default', 'handle': 'one'},
               {'member_id': 'two', 'profile': 'two', 'handle': 'two'}]
    async def probe(desc, restart):
        async with websocket(home, desc) as ws:
            capabilities = await rpc(ws, 'groups.capabilities')
            assert capabilities['result']['driver'], capabilities
            if not restart:
                # Directory existence must not become cross-profile execution permission.
                invalid = [members[0], {'member_id': 'other', 'profile': 'other', 'handle': 'other'}]
                rejected = await rpc(ws, 'groups.create', room_id='invalid', name='Invalid', members=invalid)
                assert 'error' in rejected, rejected
                created = await rpc(ws, 'groups.create', room_id='owned', name='Owned', members=members)
                assert 'result' in created, created
            sent = await rpc(ws, 'groups.send', room_id='owned', event_id='input-one',
                             payload={'text': 'LOCAL_HOSTED_PROOF', 'thread_id': 'thread'})
            assert sent['result']['accepted'], sent
            async with asyncio.timeout(25):
                while True:
                    log = await rpc(ws, 'groups.log', room_id='owned')
                    if sum(e['kind'] == 'turn.settled' for e in log['result']['events']) >= 2:
                        break
                    await asyncio.sleep(.1)
            state = await rpc(ws, 'groups.state', room_id='owned')
            assert state['result']['driver_status']['counts'].get('settled') == 2, state
            assert len(model.requests) == 2, model.requests
    try:
        for restart in (False, True):
            with daemon(root, target, env | {'HERMES_HOME': str(target)}, barrier=False), daemon(root, home, env, barrier=False) as (_, desc):
                asyncio.run(probe(desc, restart))
    finally:
        model.shutdown(); model.server_close()


def test_room_unknown_discard_releases_only_its_followers(tmp_path):
    root = Path(__file__).resolve().parents[2]
    home, user = tmp_path / 'state', tmp_path / 'user'
    home.mkdir(mode=0o700)
    user.mkdir()
    target = home / 'profiles' / 'two'
    target.mkdir(parents=True)
    model = ThreadingHTTPServer(('127.0.0.1', 0), RoomModel)
    model.requests = []
    model.blocked, model.release = threading.Event(), threading.Event()
    threading.Thread(target=model.serve_forever, daemon=True).start()
    base = f'http://127.0.0.1:{model.server_port}/v1'
    (home / 'config.yaml').write_text(json.dumps({
        'gateway': {'multiplex_profiles': False}, 'hosted_rooms': {'profiles': {'two': str(target)}},
        'model': {'provider': 'custom', 'default': 'loopback-room', 'base_url': base},
        'platform_toolsets': {'gui': [], 'bot_room': []},
        'auxiliary': {'title_generation': {'enabled': False}},
        'terminal': {'cwd': str(home)},
    }))
    (target / 'config.yaml').write_text((home / 'config.yaml').read_text())
    env = child_env() | dict(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home),
        PYTHONPATH=str(root), OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=base, PYTHONUNBUFFERED='1')
    members = [{'member_id': 'one', 'profile': 'default', 'handle': 'one'},
               {'member_id': 'two', 'profile': 'two', 'handle': 'two'}]

    async def state(ws):
        reply = await rpc(ws, 'groups.state', room_id='unknown')
        assert 'result' in reply, reply
        return reply['result']['driver_status']

    previous = {}

    async def first(proc, desc):
        async with websocket(home, desc) as ws:
            for room in ('unknown', 'unrelated'):
                reply = await rpc(ws, 'groups.create', room_id=room, name=room, members=members)
                assert 'result' in reply, reply
            reply = await rpc(ws, 'groups.send', room_id='unknown', event_id='started',
                              payload={'text': 'BLOCK_HOSTED', 'thread_id': 'thread'})
            assert reply['result']['accepted'], reply
            assert await asyncio.to_thread(model.blocked.wait, 25)
            reply = await rpc(ws, 'groups.send', room_id='unknown', event_id='follower',
                              payload={'text': 'FOLLOWER_HOSTED', 'thread_id': 'thread'})
            assert reply['result']['accepted'], reply
            import sqlite3
            with sqlite3.connect(f'file:{home / "state.db"}?mode=ro', uri=True) as db:
                previous["lease"] = db.execute("SELECT process_generation FROM hosted_room_driver_leases WHERE room_id=?", ("unknown",)).fetchone()[0]
            proc.kill()
            await asyncio.to_thread(proc.wait, 10)

    async def recovered(desc):
        async with websocket(home, desc) as ws:
            async with asyncio.timeout(30):
                while True:
                    snapshot = await state(ws)
                    actions = [a for a in snapshot['pending_actions'] if a['kind'] == 'discard']
                    if actions:
                        import sqlite3
                        with sqlite3.connect(f'file:{home / "state.db"}?mode=ro', uri=True) as db:
                            lease = db.execute('SELECT process_generation FROM hosted_room_driver_leases WHERE room_id=?', ('unknown',)).fetchone()
                        if lease is not None and lease[0] != previous['lease']:
                            break
                    await asyncio.sleep(.1)
            assert len(model.requests) == 1, model.requests
            control = {k: v for k, v in actions[0].items() if k != 'kind'}
            for method, params, reason in [
                ('groups.retry', control, 'unknown_execution'),
                ('groups.discard', {**control, 'execution_generation': control['execution_generation'] + 1}, 'stale_generation'),
                ('groups.discard', control | {'room_id': 'unrelated'}, 'stale_generation'),
            ]:
                reply = await rpc(ws, method, **({'room_id': 'unknown'} | params))
                assert reply.get('error', {}).get('message') == reason, reply
            reply = await rpc(ws, 'groups.discard', room_id='unknown', **control)
            assert reply.get('result', {}).get('discarded'), reply
            same = await rpc(ws, 'groups.discard', room_id='unknown', **control)
            assert same == reply, (same, reply)
            async with asyncio.timeout(30):
                while True:
                    snapshot = await state(ws)
                    if snapshot['counts'].get('settled', 0) >= 2:
                        break
                    await asyncio.sleep(.1)
            assert snapshot['counts'].get('cancelled') == 1, snapshot
            assert len(model.requests) == 3, model.requests
    try:
        with daemon(root, target, env | {'HERMES_HOME': str(target)}, barrier=False), daemon(root, home, env, barrier=False) as (proc, desc):
            asyncio.run(first(proc, desc))
        model.release.set()
        with daemon(root, target, env | {'HERMES_HOME': str(target)}, barrier=False), daemon(root, home, env, barrier=False) as (_, desc):
            asyncio.run(recovered(desc))
    finally:
        model.release.set()
        model.shutdown()
        model.server_close()
