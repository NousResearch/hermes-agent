"""Two real canonical gateways: a Group Chat on one, with a member hosted by the other."""
import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import socket
import sqlite3
import threading
import urllib.error
import urllib.request

from tests.gateway.fixtures.local_recovery_probe import child_env, daemon, rpc, websocket


_PIXEL = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aP1sAAAAASUVORK5CYII='


def _last_user_text(body):
    return next((str(m.get('content', '')) for m in reversed(body.get('messages', []))
                 if m.get('role') == 'user'), '')


class Model(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        if body.get('messages'):
            self.server.requests.append(body)
        for marker, (held, release) in self.server.gates.items():
            if marker in _last_user_text(body):
                held.set()
                release.wait(60)
        message = {'role': 'assistant', 'content': self.server.reply}
        data = {'id': 'fixture', 'choices': [{'index': 0, 'message': message, 'finish_reason': 'stop'}],
                'usage': {'prompt_tokens': 10, 'completion_tokens': 1, 'total_tokens': 11}}
        payload, kind = json.dumps(data).encode(), 'application/json'
        if body.get('stream'):
            chunk = {'id': 'fixture', 'choices': [{'index': 0, 'delta': message, 'finish_reason': 'stop'}]}
            payload, kind = ('data: ' + json.dumps(chunk) + '\n\ndata: [DONE]\n\n').encode(), 'text/event-stream'
        self.send_response(200)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        try:
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass


def _model(reply, *markers):
    server = ThreadingHTTPServer(('127.0.0.1', 0), Model)
    server.requests, server.reply = [], reply
    server.gates = {marker: (threading.Event(), threading.Event()) for marker in markers}
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def _gateway(tmp_path, name, model, root, *, api_port=None):
    home, user = tmp_path / name, tmp_path / f'{name}-user'
    home.mkdir(mode=0o700)
    user.mkdir()
    base = f'http://127.0.0.1:{model.server_port}/v1'
    gateway = {'multiplex_profiles': False}
    env = child_env() | dict(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home),
                             PYTHONPATH=str(root), OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=base,
                             PYTHONUNBUFFERED='1')
    if api_port is not None:
        gateway |= {'room_link_url': f'http://127.0.0.1:{api_port}',
                    'platforms': {'api_server': {'enabled': True, 'port': api_port, 'host': '127.0.0.1'}}}
        env |= dict(API_SERVER_KEY='target-gateway-owned-secret', API_SERVER_ENABLED='true',
                    API_SERVER_PORT=str(api_port))
    (home / 'config.yaml').write_text(json.dumps({
        'model': {'provider': 'custom', 'default': 'room-model', 'base_url': base},
        'gateway': gateway, 'platform_toolsets': {'gui': [], 'bot_room': [], 'api_server': []},
        'auxiliary': {'title_generation': {'enabled': False}}, 'terminal': {'cwd': str(home)}}))
    return home, env


def _grant_status(url, grant):
    request = urllib.request.Request(url + '/v1/room-members/capabilities',
                                     headers={'Authorization': 'HermesRoom ' + grant})
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


async def _events(ws, kind, count=1, timeout=60):
    async with asyncio.timeout(timeout):
        while len(found := [e for e in (await rpc(ws, 'groups.log', room_id='linked'))['result']['events']
                            if e['kind'] == kind]) < count:
            await asyncio.sleep(.1)
    return found


async def _send(ws, event_id, text):
    sent = await rpc(ws, 'groups.send', room_id='linked', event_id=event_id,
                     payload={'text': text, 'thread_id': 'thread'})
    assert sent['result']['accepted'], sent


def test_peer_member_on_another_gateway_joins_replies_stops_recovers_and_is_revoked(tmp_path):
    root = Path(__file__).resolve().parents[2]
    home_model = _model('HOME_REPLY')
    target_model = _model('PEER_REPLY', 'HOLD_STOP', 'HOLD_RESTART')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        api_port = sock.getsockname()[1]
    home, home_env = _gateway(tmp_path, 'home', home_model, root)
    target, target_env = _gateway(tmp_path, 'target', target_model, root, api_port=api_port)
    grants, routes = [], []

    async def invite(target_ws, room, **lifetimes):
        invited = await rpc(target_ws, 'groups.peer.invite', room_id='linked', member_id='reviewer',
                            home_install_id=room['authority_gateway_id'],
                            authority_gateway_id=room['authority_gateway_id'],
                            authority_epoch=room['authority_epoch'], **lifetimes)
        grants.append(invited['result']['grant'])
        return invited['result']

    def stored_route():
        with sqlite3.connect(f'file:{home / "state.db"}?mode=ro', uri=True) as db:
            return db.execute('SELECT grant, trace_id, cancellation_scope_id FROM hosted_room_links').fetchall()

    async def before_restart(home_desc, target_desc, home_proc):
        async with websocket(home, home_desc) as home_ws, websocket(target, target_desc) as target_ws:
            async with asyncio.timeout(30):
                while not (link := (await rpc(target_ws, 'groups.capabilities'))['result']['room_link'])['enabled']:
                    await asyncio.sleep(.1)
            catalog = link['catalog']
            routes.append(link['endpoint']['url'])
            assert link['endpoint'] == catalog['endpoint'] == {
                'available': True, 'url': f'http://127.0.0.1:{api_port}', 'transport_security': 'loopback'}
            assert catalog['text'] and not catalog['attachments'], catalog
            capabilities = (await rpc(home_ws, 'groups.capabilities'))['result']
            assert capabilities['driver'], capabilities
            assert capabilities['room_link'] == {'enabled': False, 'reason': 'api_server_required'}
            assert {'groups.peer.register', 'groups.peer.invite', 'groups.peer.revoke'} <= set(capabilities['methods'])

            pinned = {'kind': 'peer', 'peer_id': 'target-gateway', 'installation_id': catalog['installation_id'],
                      'profile': 'default', 'capability_digest': catalog['catalog_digest']}
            members = [{'member_id': 'host', 'profile': 'default', 'handle': 'host'},
                       {'member_id': 'reviewer', 'profile': 'default', 'handle': 'reviewer', 'target': pinned}]
            created = await rpc(home_ws, 'groups.create', room_id='linked', name='Linked', members=members)
            room = created['result']['room']

            # Before its gateway joins, the member's turn fails visibly instead of hanging the room.
            await _send(home_ws, 'early', '@reviewer EARLY')
            failed, = await _events(home_ws, 'turn.failed')
            assert 'has not joined this Group Chat' in failed['payload']['error'], failed
            assert not (await rpc(home_ws, 'groups.state', room_id='linked'))['result']['driver_status']['working']

            invitation = await invite(target_ws, room)
            assert invitation['catalog'] == catalog and invitation['target_profile'] == 'default', invitation
            registered = await rpc(home_ws, 'groups.peer.register', room_id='linked', member_id='reviewer',
                                   target_url=routes[0], target_profile='default',
                                   grant=invitation['grant'], catalog=catalog)
            assert registered['result'] == {
                'registered': True, 'mode': 'direct', 'transport_security': 'loopback',
                'target_install_id': catalog['installation_id'], 'target_profile': 'default'}, registered

            await _send(home_ws, 'ask', '@reviewer PEER_PROOF')
            reply, = await _events(home_ws, 'message.member')
            assert reply['payload']['member_id'] == 'reviewer', reply
            assert reply['payload']['text'] == 'PEER_REPLY', reply
            assert reply['actor']['connection_id'] == 'target-gateway', reply
            assert home_model.requests == []
            assert len(target_model.requests) == 1 and 'PEER_PROOF' in _last_user_text(target_model.requests[0])
            # The member's gateway ran it as a room run: POST /v1/runs under the room's idempotency
            # key, kept in its Runs store and admitted as an ordinary API turn there.
            with sqlite3.connect(f'file:{target / "runs_idempotency.db"}?mode=ro', uri=True) as db:
                (key, run_id), = db.execute('SELECT idempotency_key, run_id FROM run_idempotency').fetchall()
            assert key.startswith('room:dtask:') and key.endswith(':1'), key
            with sqlite3.connect(f'file:{target / "state.db"}?mode=ro', uri=True) as db:
                assert db.execute("SELECT status FROM session_admissions WHERE principal_id='api' "
                                  'AND request_id=?', (run_id,)).fetchall() == [('terminal',)]
            assert len(await _events(home_ws, 'turn.failed')) == 1
            state = (await rpc(home_ws, 'groups.state', room_id='linked'))['result']
            assert state['driver_status']['peer_routes'] == [
                {'room_id': 'linked', 'member_id': 'reviewer', 'status': 'ready'}], state

            # The member receives text only: a file addressed to it fails visibly, nothing is sent.
            uploaded = (await rpc(home_ws, 'groups.attachment.upload', room_id='linked', upload_id='pixel',
                                  kind='image', name='pixel.png', mime='image/png', data_base64=_PIXEL))['result']
            manifest = [{k: uploaded[k] for k in ('attachment_id', 'kind', 'name', 'size', 'mime')}]
            sent = await rpc(home_ws, 'groups.send', room_id='linked', event_id='file', payload={
                'text': '@reviewer FILE_PROOF', 'thread_id': 'thread', 'attachments': manifest})
            assert sent['result']['accepted'], sent
            failed = (await _events(home_ws, 'turn.failed', count=2))[-1]
            assert 'can receive text only' in failed['payload']['error'], failed
            assert len(target_model.requests) == 1

            # Stop reaches the work running on the other gateway.
            held, release = target_model.gates['HOLD_STOP']
            await _send(home_ws, 'stop', '@reviewer HOLD_STOP')
            assert await asyncio.to_thread(held.wait, 30)
            assert (await rpc(home_ws, 'groups.stop', room_id='linked'))['result'] == {'cancelled': 1}
            await _events(home_ws, 'turn.cancelled', timeout=30)
            release.set()

            # A fresh grant re-registers the same route: accepted work keeps its dispatch identity.
            (_, trace, cancel_scope), = stored_route()
            invitation = await invite(target_ws, room)
            registered = await rpc(home_ws, 'groups.peer.register', room_id='linked', member_id='reviewer',
                                   target_url=routes[0], target_profile='default',
                                   grant=invitation['grant'], catalog=catalog)
            assert registered['result']['registered'], registered
            assert stored_route() == [(invitation['grant'], trace, cancel_scope)]
            # The grant it replaced is retired at once, not when it expires.
            status, body = await asyncio.to_thread(_grant_status, routes[0], grants[-2])
            assert (status, body['error']['code']) == (403, 'room_reauthorization_required'), body
            assert (await asyncio.to_thread(_grant_status, routes[0], grants[-1]))[0] == 200

            # A one-minute grant under a one-hour horizon: the home renews it on its own before it
            # expires, retires it, and keeps the member working on the renewal (checked below).
            short = await invite(target_ws, room, ttl_seconds=60, status_ttl_seconds=3600)
            registered = await rpc(home_ws, 'groups.peer.register', room_id='linked', member_id='reviewer',
                                   target_url=routes[0], target_profile='default',
                                   grant=short['grant'], catalog=catalog)
            assert registered['result']['registered'], registered
            async with asyncio.timeout(60):
                while stored_route()[0][0] == short['grant']:
                    await asyncio.sleep(.2)
            assert stored_route()[0][1:] == (trace, cancel_scope)
            status, body = await asyncio.to_thread(_grant_status, routes[0], short['grant'])
            assert (status, body['error']['code']) == (403, 'room_reauthorization_required'), body
            assert (await asyncio.to_thread(_grant_status, routes[0], stored_route()[0][0]))[0] == 200

            held, _ = target_model.gates['HOLD_RESTART']
            await _send(home_ws, 'restart', '@reviewer HOLD_RESTART')
            assert await asyncio.to_thread(held.wait, 30)
            home_proc.kill()
            await asyncio.to_thread(home_proc.wait, 10)

    async def after_restart(home_desc):
        async with websocket(home, home_desc) as home_ws:
            # The restarted home recovers the accepted turn from its stored route: one run, one reply.
            replies = await _events(home_ws, 'message.member', count=2)
            assert [r['payload']['text'] for r in replies] == ['PEER_REPLY', 'PEER_REPLY'], replies
            assert sum('HOLD_RESTART' in _last_user_text(r) for r in target_model.requests) == 1
            # Only the current (renewed) grant works; every grant it replaced stayed retired.
            assert [(await asyncio.to_thread(_grant_status, routes[0], grant))[0] for grant in grants] == [403] * 3
            assert (await asyncio.to_thread(_grant_status, routes[0], stored_route()[0][0]))[0] == 200
            grants.append(stored_route()[0][0])

            disbanded = await rpc(home_ws, 'groups.disband', room_id='linked')
            assert 'tombstone' in disbanded['result'], disbanded
            for grant in grants:
                status, body = await asyncio.to_thread(_grant_status, routes[0], grant)
                assert (status, body['error']['code']) == (403, 'room_reauthorization_required'), body

    try:
        with daemon(root, target, target_env, barrier=False) as (_, target_desc):
            with daemon(root, home, home_env, barrier=False) as (home_proc, home_desc):
                asyncio.run(before_restart(home_desc, target_desc, home_proc))
            target_model.gates['HOLD_RESTART'][1].set()
            with daemon(root, home, home_env, barrier=False) as (_, home_desc):
                asyncio.run(after_restart(home_desc))
    finally:
        for model in (home_model, target_model):
            for _, release in model.gates.values():
                release.set()
            model.shutdown()
            model.server_close()
