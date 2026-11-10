"""Two real canonical gateways: a Group Chat on one, with a member hosted by the other."""
import asyncio
from contextlib import ExitStack
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import socket
import sqlite3
import struct
from tests.waiting import wait_until, async_wait_until
import threading
from types import SimpleNamespace
import urllib.error
import urllib.request

import pytest

from tests.gateway.fixtures.local_recovery_probe import child_env, daemon, rpc, websocket


# These tests kill only the child handles returned by their isolated daemon fixture.
pytestmark = pytest.mark.live_system_guard_bypass

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


def _model(reply, *markers, handler=Model):
    server = ThreadingHTTPServer(('127.0.0.1', 0), handler)
    server.requests, server.reply = [], reply
    server.gates = {marker: (threading.Event(), threading.Event()) for marker in markers}
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def _gateway(tmp_path, name, model, root, *, api_port=None, link_port=None):
    home, user = tmp_path / name, tmp_path / f'{name}-user'
    home.mkdir(mode=0o700)
    user.mkdir()
    base = f'http://127.0.0.1:{model.server_port}/v1'
    gateway = {'multiplex_profiles': False}
    env = child_env() | dict(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home),
                             PYTHONPATH=str(root), OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=base,
                             PYTHONUNBUFFERED='1')
    if api_port is not None:
        gateway |= {'room_link_url': f'http://127.0.0.1:{link_port or api_port}',
                    'platforms': {'api_server': {'enabled': True, 'port': api_port, 'host': '127.0.0.1'}}}
        env |= dict(API_SERVER_KEY='target-gateway-owned-secret', API_SERVER_ENABLED='true',
                    API_SERVER_PORT=str(api_port))
    (home / 'config.yaml').write_text(json.dumps({
        'model': {'provider': 'custom', 'default': 'room-model', 'base_url': base},
        'gateway': gateway, 'platform_toolsets': {'gui': [], 'bot_room': [], 'api_server': []},
        'auxiliary': {'title_generation': {'enabled': False}}, 'terminal': {'cwd': str(home)}}))
    return home, env


@pytest.fixture
def gateways(tmp_path):
    """Isolated models and gateway environments; tests own every daemon/restart boundary."""
    models, relays = [], []

    def create(*markers, lossy=False, handler=Model):
        root = Path(__file__).resolve().parents[2]
        home_model, target_model = _model('HOME_REPLY'), _model('PEER_REPLY', *markers, handler=handler)
        models.extend((home_model, target_model))
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            api_port = sock.getsockname()[1]
        relay = _LossyRelay(api_port) if lossy else None
        if relay:
            relays.append(relay)
        home, home_env = _gateway(tmp_path, 'home', home_model, root)
        target, target_env = _gateway(tmp_path, 'target', target_model, root,
                                      api_port=api_port, link_port=relay.port if relay else None)
        return SimpleNamespace(root=root, home=home, home_env=home_env, target=target, target_env=target_env,
                               home_model=home_model, target_model=target_model, api_port=api_port, relay=relay)
    yield create
    for relay in relays:
        relay.close()
    for model in models:
        for _, release in model.gates.values():
            release.set()
        model.shutdown()
        model.server_close()


async def _create(home_ws, catalog, *, peer_id='target-gateway'):
    pinned = {'kind': 'peer', 'peer_id': peer_id, 'installation_id': catalog['installation_id'],
              'profile': 'default', 'capability_digest': catalog['catalog_digest']}
    return (await rpc(home_ws, 'groups.create', room_id='linked', name='Linked', members=[
        {'member_id': 'host', 'profile': 'default', 'handle': 'host'},
        {'member_id': 'reviewer', 'profile': 'default', 'handle': 'reviewer', 'target': pinned}]))['result']['room']


async def _invite(target_ws, room, **lifetimes):
    invited = await rpc(target_ws, 'groups.peer.invite', room_id=room['room_id'], member_id='reviewer',
                        home_install_id=room['authority_gateway_id'],
                        authority_gateway_id=room['authority_gateway_id'],
                        authority_epoch=room['authority_epoch'], **lifetimes)
    assert invited['result']['grant'], invited
    return invited['result']


async def _register(home_ws, catalog, grant):
    return await rpc(home_ws, 'groups.peer.register', room_id='linked', member_id='reviewer',
                     target_url=catalog['endpoint']['url'], target_profile='default', grant=grant, catalog=catalog)


def _grant_status(url, grant):
    request = urllib.request.Request(url + '/v1/room-members/capabilities',
                                     headers={'Authorization': 'HermesRoom ' + grant})
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


async def _revoked(url, grant, timeout=60):
    """Wait for the home's worker to pay the revoke Disband left it owing."""
    def revoked():
        status, body = _grant_status(url, grant)
        if status == 200:
            return False
        assert (status, body['error']['code']) == (403, 'room_reauthorization_required'), body
        return True
    await asyncio.to_thread(wait_until, revoked, timeout=timeout, interval=.25, what="grant revocation")


def _owed(home):
    with sqlite3.connect(f'file:{home / "state.db"}?mode=ro', uri=True) as db:
        return db.execute('SELECT room_id, member_id FROM hosted_room_owed_revokes').fetchall()


async def _wait_rpc(ws, method, ready, *, timeout, **params):
    loop = asyncio.get_running_loop()

    def probe():
        reply = asyncio.run_coroutine_threadsafe(rpc(ws, method, **params), loop).result(timeout=timeout)
        return ready(reply['result'])
    return await asyncio.to_thread(wait_until, probe, timeout=timeout, interval=.1, what=method)


async def _events(ws, kind, count=1, timeout=60):
    def ready(result):
        found = [event for event in result['events'] if event['kind'] == kind]
        return (found,) if len(found) >= count else None
    found, = await _wait_rpc(ws, 'groups.log', ready, timeout=timeout, room_id='linked')
    return found


async def _room_link(ws, timeout=30):
    """The member gateway's room_link once its API server is up."""
    return await _wait_rpc(ws, 'groups.capabilities',
                          lambda result: result['room_link'] if result['room_link']['enabled'] else None,
                          timeout=timeout)


async def _send(ws, event_id, text):
    sent = await rpc(ws, 'groups.send', room_id='linked', event_id=event_id,
                     payload={'text': text, 'thread_id': 'thread'})
    assert sent['result']['accepted'], sent


def test_peer_member_on_another_gateway_joins_replies_stops_recovers_and_is_revoked(gateways):
    g = gateways('HOLD_STOP', 'HOLD_RESTART')
    grants, routes = [], []

    async def invite(target_ws, room):
        invited = await _invite(target_ws, room)
        grants.append(invited['grant'])
        return invited

    def stored_route():
        with sqlite3.connect(f'file:{g.home / "state.db"}?mode=ro', uri=True) as db:
            return db.execute('SELECT grant, trace_id, cancellation_scope_id FROM hosted_room_links').fetchall()

    async def before_restart(home_desc, target_desc, home_proc):
        async with websocket(g.home, home_desc) as home_ws, websocket(g.target, target_desc) as target_ws:
            link = await _room_link(target_ws)
            catalog = link['catalog']
            routes.append(link['endpoint']['url'])
            assert link['endpoint'] == catalog['endpoint'] == {
                'available': True, 'url': f'http://127.0.0.1:{g.api_port}', 'transport_security': 'loopback'}
            assert catalog['text'] and not catalog['attachments'], catalog
            capabilities = (await rpc(home_ws, 'groups.capabilities'))['result']
            assert capabilities['driver'], capabilities
            assert capabilities['room_link'] == {'enabled': False, 'reason': 'api_server_required'}
            assert {'groups.peer.register', 'groups.peer.invite', 'groups.peer.revoke'} <= set(capabilities['methods'])

            room = await _create(home_ws, catalog)

            # Before its gateway joins, the member's turn fails visibly instead of hanging the room.
            await _send(home_ws, 'early', '@reviewer EARLY')
            failed, = await _events(home_ws, 'turn.failed')
            assert failed['payload']['error'] == 'peer room route is unavailable', failed
            assert not (await rpc(home_ws, 'groups.state', room_id='linked'))['result']['driver_status']['working']

            invitation = await invite(target_ws, room)
            assert invitation['catalog'] == catalog and invitation['target_profile'] == 'default', invitation
            registered = await _register(home_ws, catalog, invitation['grant'])
            assert registered['result'] == {
                'registered': True, 'mode': 'direct', 'transport_security': 'loopback',
                'target_install_id': catalog['installation_id'], 'target_profile': 'default'}, registered

            await _send(home_ws, 'ask', '@reviewer PEER_PROOF')
            reply, = await _events(home_ws, 'message.member')
            assert reply['payload']['member_id'] == 'reviewer', reply
            assert reply['payload']['text'] == 'PEER_REPLY', reply
            assert reply['actor']['connection_id'] == 'target-gateway', reply
            assert g.home_model.requests == []
            assert len(g.target_model.requests) == 1 and 'PEER_PROOF' in _last_user_text(g.target_model.requests[0])
            # The member's gateway ran it as a room run: POST /v1/runs under the room's idempotency
            # key, kept in its Runs store and admitted as an ordinary API turn there.
            with sqlite3.connect(f'file:{g.target / "runs_idempotency.db"}?mode=ro', uri=True) as db:
                (key, run_id), = db.execute('SELECT idempotency_key, run_id FROM run_idempotency').fetchall()
            assert key.startswith('room:dtask:') and key.endswith(':1'), key
            with sqlite3.connect(f'file:{g.target / "state.db"}?mode=ro', uri=True) as db:
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
            assert failed['payload']['error'].startswith('attachments_unsupported_on_route'), failed
            assert len(g.target_model.requests) == 1
            # Finish the file-refusal scenario before starting the independent Stop scenario.
            await _wait_rpc(home_ws, 'groups.log', lambda result: any(
                event['kind'] == 'room.activity' and event['payload'].get('discussion_event_id') ==
                failed['payload']['discussion_event_id'] for event in result['events']),
                timeout=30, room_id='linked')

            # Stop reaches the work running on the other gateway.
            held, release = g.target_model.gates['HOLD_STOP']
            await _send(home_ws, 'stop', '@reviewer HOLD_STOP')
            assert await asyncio.to_thread(held.wait, 90)
            assert (await rpc(home_ws, 'groups.stop', room_id='linked'))['result'] == {'cancelled': 1}
            await _events(home_ws, 'turn.cancelled', timeout=30)
            release.set()

            # A fresh grant re-registers the same route: accepted work keeps its dispatch identity.
            (_, trace, cancel_scope), = stored_route()
            invitation = await invite(target_ws, room)
            registered = await _register(home_ws, catalog, invitation['grant'])
            assert registered['result']['registered'], registered
            assert stored_route() == [(invitation['grant'], trace, cancel_scope)]

            held, _ = g.target_model.gates['HOLD_RESTART']
            await _send(home_ws, 'restart', '@reviewer HOLD_RESTART')
            assert await asyncio.to_thread(held.wait, 90)
            home_proc.kill()
            await asyncio.to_thread(home_proc.wait, 10)

    async def after_restart(home_desc):
        async with websocket(g.home, home_desc) as home_ws:
            # The restarted home recovers the accepted turn from its stored route: one run, one reply.
            replies = await _events(home_ws, 'message.member', count=2)
            assert [r['payload']['text'] for r in replies] == ['PEER_REPLY', 'PEER_REPLY'], replies
            assert sum('HOLD_RESTART' in _last_user_text(r) for r in g.target_model.requests) == 1
            for grant in grants:
                assert (await asyncio.to_thread(_grant_status, routes[0], grant))[0] == 200

    async def disband_while_member_is_down(home_desc):
        async with websocket(g.home, home_desc) as home_ws:
            # Disband revokes at the member gateway first: while it is down the room stays.
            refused = await rpc(home_ws, 'groups.disband', room_id='linked')
            assert 'error' in refused, refused
            assert not any(grant in json.dumps(refused) for grant in grants)
            assert stored_route() and _owed(g.home) == []

    async def revoked_once_member_returns(home_desc):
        async with websocket(g.home, home_desc) as home_ws:
            disbanded = await rpc(home_ws, 'groups.disband', room_id='linked')
            assert 'tombstone' in disbanded['result'], disbanded
            assert not any(grant in json.dumps(disbanded) for grant in grants)
        for grant in grants:
            await _revoked(routes[0], grant, timeout=120)
        assert not _owed(g.home) and not stored_route()

    with ExitStack() as running:
        member = running.enter_context(ExitStack())
        _, target_desc = member.enter_context(daemon(g.root, g.target, g.target_env, barrier=False))
        with daemon(g.root, g.home, g.home_env, barrier=False) as (home_proc, home_desc):
            asyncio.run(before_restart(home_desc, target_desc, home_proc))
        g.target_model.gates['HOLD_RESTART'][1].set()
        _, home_desc = running.enter_context(daemon(g.root, g.home, g.home_env, barrier=False))
        asyncio.run(after_restart(home_desc))
        # The member's gateway stops while the room is live: Disband reports it unreachable and keeps
        # the room; once that gateway runs again, Disband revokes there and ends the room.
        member.close()
        asyncio.run(disband_while_member_is_down(home_desc))
        running.enter_context(daemon(g.root, g.target, g.target_env, barrier=False))
        asyncio.run(revoked_once_member_returns(home_desc))


class _LossyRelay:
    """Lose an admitted request's reply, then definitively refuse its immediate replay."""

    def __init__(self, target_port):
        self.target_port, self.admissions = target_port, 0
        self.closed, self.replay_refused = threading.Event(), threading.Event()
        self.listener = socket.create_server(('127.0.0.1', 0))
        self.port = self.listener.getsockname()[1]
        threading.Thread(target=self._serve, daemon=True).start()

    def close(self):
        self.closed.set()
        self.listener.close()

    def _serve(self):
        while not self.closed.is_set():
            try:
                client, _ = self.listener.accept()
            except OSError:
                return
            threading.Thread(target=self._relay, args=(client,), daemon=True).start()

    @staticmethod
    def _request(client):
        data = b''
        while b'\r\n\r\n' not in data:
            data += client.recv(65536) or b''
        head, _, body = data.partition(b'\r\n\r\n')
        length = next((int(line.split(b':', 1)[1]) for line in head.split(b'\r\n')
                       if line.lower().startswith(b'content-length:')), 0)
        while len(body) < length:
            body += client.recv(65536)
        return head, head + b'\r\n\r\n' + body

    def _relay(self, client):
        with client:
            head, request = self._request(client)
            admission = head.startswith(b'POST /v1/runs ')
            if admission and self.admissions == 1 and not self.replay_refused.is_set():
                # A definite refusal cannot disprove the admission whose reply was lost.
                self.replay_refused.set()
                client.sendall(b'HTTP/1.1 429 Too Many Requests\r\nContent-Length: 2\r\n'
                               b'Connection: close\r\n\r\n{}')
                return
            with socket.create_connection(('127.0.0.1', self.target_port)) as upstream:
                upstream.sendall(request)
                reply = b''
                while chunk := upstream.recv(65536):
                    reply += chunk
            if admission and self.admissions == 0:
                assert reply.split(b' ', 2)[1] == b'202', reply
                self.admissions += 1
                client.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack('ii', 1, 0))
                return
            client.sendall(reply)


def test_a_lost_admission_reply_never_runs_the_peer_turn_twice(gateways):
    """The member's gateway admits the turn, its reply is lost, and the home's identical replay
    is refused before admission. That refusal proves nothing about the first request, so the home recovers the
    same attempt: the member's gateway runs the turn once and the room gets one reply."""
    g = gateways(lossy=True)
    async def drive(home_desc, target_desc):
        async with websocket(g.home, home_desc) as home_ws, websocket(g.target, target_desc) as target_ws:
            catalog = (await _room_link(target_ws))['catalog']
            room = await _create(home_ws, catalog)
            invited = await _invite(target_ws, room)
            registered = await _register(home_ws, catalog, invited['grant'])
            assert registered['result']['registered'], registered

            await _send(home_ws, 'ask', '@reviewer LOST_REPLY_PROOF')
            reply, = await _events(home_ws, 'message.member', timeout=90)
            assert (reply['payload']['member_id'], reply['payload']['text']) == ('reviewer', 'PEER_REPLY'), reply
            assert g.relay.admissions == 1 and g.relay.replay_refused.is_set()
            assert not await _events(home_ws, 'turn.failed', count=0)
            assert sum('LOST_REPLY_PROOF' in _last_user_text(r) for r in g.target_model.requests) == 1
            with sqlite3.connect(f'file:{g.target / "runs_idempotency.db"}?mode=ro', uri=True) as db:
                (key,), = db.execute('SELECT idempotency_key FROM run_idempotency').fetchall()
            assert key.startswith('room:dtask:') and key.endswith(':1'), key

    with daemon(g.root, g.target, g.target_env, barrier=False) as (_, target_desc):
        with daemon(g.root, g.home, g.home_env, barrier=False) as (_, home_desc):
            asyncio.run(drive(home_desc, target_desc))
