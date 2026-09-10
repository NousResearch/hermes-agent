"""Loopback OIDC/model boundaries; daemon, auth, tickets and room controls are real."""
import asyncio
import base64
from contextlib import ExitStack
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import threading
import time
from urllib.parse import parse_qs, urlencode, urlsplit

import httpx
import jwt
from cryptography.hazmat.primitives.asymmetric import rsa
from websockets.asyncio.client import connect

from tests.gateway.fixtures.local_recovery_probe import child_env, daemon, rpc


class OIDC(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, body, status=200):
        raw = json.dumps(body).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        url = urlsplit(self.path)
        issuer = self.server.issuer
        if url.path == '/jwks':
            self.reply({'keys': [self.server.jwk]})
        elif url.path == '/.well-known/openid-configuration':
            self.reply({'issuer': issuer, 'jwks_uri': issuer + '/jwks',
                        'authorization_endpoint': issuer + '/authorize',
                        'token_endpoint': issuer + '/token'})
        elif url.path == '/authorize':
            params = {k: v[0] for k, v in parse_qs(url.query).items()}
            assert params['client_id'] == 'room-fixture'
            assert params['code_challenge_method'] == 'S256'
            assert params['subject'] in ('alice', 'bob')
            code = secrets.token_urlsafe(24)
            self.server.codes[code] = params
            self.send_response(302)
            self.send_header('Location', params['redirect_uri'] + '?' + urlencode(
                {'code': code, 'state': params['state']}))
            self.send_header('Content-Length', '0')
            self.end_headers()
        else:
            self.reply({}, 404)

    def do_POST(self):
        params = {k: v[0] for k, v in parse_qs(
            self.rfile.read(int(self.headers['Content-Length'])).decode()).items()}
        grant = self.server.codes.pop(params.get('code'), None)
        challenge = base64.urlsafe_b64encode(hashlib.sha256(
            params.get('code_verifier', '').encode()).digest()).decode().rstrip('=')
        if (self.path != '/token' or not grant or challenge != grant['code_challenge']
                or params.get('redirect_uri') != grant['redirect_uri']
                or params.get('client_id') != 'room-fixture'):
            self.reply({'error': 'invalid_grant'}, 400)
            return
        subject = grant['subject']
        token = jwt.encode({'iss': self.server.issuer, 'sub': subject,
            'aud': 'room-fixture', 'iat': int(time.time()), 'exp': int(time.time()) + 600},
            self.server.key, algorithm='RS256', headers={'kid': 'room-key'})
        self.server.logins.append(subject)
        self.reply({'id_token': token, 'access_token': token, 'token_type': 'Bearer',
                    'expires_in': 600})


class Model(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        if body.get('messages'):
            self.server.requests.append(body)
        text = str(next((m.get('content', '') for m in reversed(body.get('messages', []))
                         if m['role'] == 'user'), ''))
        for actor in ('alice', 'bob'):
            if 'BLOCK_' + actor in text:
                self.server.blocked[actor].set()
                self.server.release.wait(60)
        message = {'role': 'assistant', 'content': 'PASS'}
        if 'APPROVE_HOSTED' in text and body.get('messages', [{}])[-1].get('role') != 'tool':
            message = {'role': 'assistant', 'content': None, 'tool_calls': [{'index': 0, 'id': 'owned-remove',
                'type': 'function', 'function': {'name': 'terminal',
                'arguments': json.dumps({'command': self.server.command})}}]}
        data = {'id': 'loopback', 'choices': [{'index': 0, 'message': message, 'finish_reason': 'stop'}],
                'usage': {'prompt_tokens': 10, 'completion_tokens': 1, 'total_tokens': 11}}
        payload, kind = json.dumps(data).encode(), 'application/json'
        if body.get('stream'):
            payload = ('data: ' + json.dumps({'id': 'loopback', 'choices': [
                {'index': 0, 'delta': message, 'finish_reason': 'stop'}]}) + '\n\ndata: [DONE]\n\n').encode()
            kind = 'text/event-stream'
        try:
            self.send_response(200)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass


def login(client, issuer, actor):
    response = client.get('/auth/login', params={'provider': 'self-hosted'})
    assert response.status_code == 302, response.text
    authorize = response.headers['location']
    assert authorize.startswith(issuer + '/authorize?'), authorize
    response = client.get(authorize + '&' + urlencode({'subject': actor}))
    assert response.status_code == 302, response.text
    # public_url enables the gate; route its callback to the owned loopback listener.
    callback = urlsplit(response.headers['location'])
    response = client.get(callback.path + '?' + callback.query)
    assert response.status_code == 302, response.text
    identity = client.get('/api/auth/me')
    assert identity.status_code == 200, identity.text
    assert identity.json()['user_id'] == actor, identity.text
    ticket = client.post('/api/auth/ws-ticket')
    assert ticket.status_code == 200, ticket.text
    return ticket.json()['ticket']


def probe(tmp_path):
    root = Path(__file__).resolve().parents[3]
    home, user = tmp_path / 'state', tmp_path / 'user'
    home.mkdir(mode=0o700)
    user.mkdir()
    target = home / 'profiles' / 'two'
    target.mkdir(parents=True)
    oidc = ThreadingHTTPServer(('127.0.0.1', 0), OIDC)
    oidc.issuer = f'http://127.0.0.1:{oidc.server_port}'
    oidc.key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    oidc.jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(oidc.key.public_key()))
    oidc.jwk.update(kid='room-key', use='sig', alg='RS256')
    oidc.codes, oidc.logins = {}, []
    model = ThreadingHTTPServer(('127.0.0.1', 0), Model)
    model.requests = []
    removal = home / 'owned-removal'
    removal.mkdir()
    (removal / 'sentinel').write_text('owned fixture')
    import shlex
    model.command = 'rm -r -- ' + shlex.quote(str(removal))
    model.blocked = {actor: threading.Event() for actor in ('alice', 'bob')}
    model.release = threading.Event()
    threads = [threading.Thread(target=s.serve_forever, daemon=True) for s in (oidc, model)]
    for thread in threads:
        thread.start()
    base = f'http://127.0.0.1:{model.server_port}/v1'
    cfg = {'gateway': {'multiplex_profiles': False},
        'hosted_rooms': {'profiles': {'two': str(target)}},
        'dashboard': {'public_url': 'http://rooms.invalid', 'oauth': {'self_hosted': {
            'issuer': oidc.issuer, 'client_id': 'room-fixture'}}},
        'model': {'provider': 'custom', 'default': 'loopback-room', 'base_url': base},
        'platform_toolsets': {'gui': ['terminal'], 'bot_room': ['terminal']},
        'approvals': {'mode': 'manual', 'timeout': 60},
        'auxiliary': {'title_generation': {'enabled': False}}, 'terminal': {'cwd': str(home)}}
    for path in (home, target):
        (path / 'config.yaml').write_text(json.dumps(cfg))
    env = child_env() | dict(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home),
        PYTHONPATH=str(root), OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=base, PYTHONUNBUFFERED='1')
    members = [{'member_id': 'one', 'profile': 'default', 'handle': 'one'},
               {'member_id': 'two', 'profile': 'two', 'handle': 'two'}]

    async def exercise(desc, tickets):
        origin = desc['api_origin'].replace('http:', 'ws:') + '/api/ws'
        async with connect(origin + '?ticket=' + tickets[0]) as alice, connect(
                origin + '?ticket=' + tickets[1]) as bob:
            actors = [('alice', alice, bob), ('bob', bob, alice)]
            for actor, ws, _ in actors:
                result = await rpc(ws, 'groups.create', room_id=actor + '-room', name=actor, members=members)
                assert 'result' in result, result
            denials = 0
            for actor, ws, other in actors:
                room = actor + '-room'
                listing = await rpc(ws, 'groups.list')
                assert [r['room_id'] for r in listing['result']['rooms']] == [room], listing
                state = await rpc(ws, 'groups.state', room_id=room)
                assert state['result']['room']['room_id'] == room, state
                sent = await rpc(ws, 'groups.send', room_id=room, event_id='own-input',
                                 payload={'text': 'BLOCK_' + actor, 'thread_id': 'thread'})
                assert sent['result']['accepted'], sent
                assert await asyncio.to_thread(model.blocked[actor].wait, 30), state
                before = await rpc(ws, 'groups.log', room_id=room)
                for method, params in [
                    ('groups.create', {'name': 'hijack', 'members': members}),
                    ('groups.state', {}), ('groups.log', {}),
                    ('groups.send', {'event_id': 'forbidden', 'payload': {'text': 'FORBIDDEN', 'thread_id': 'thread'}}),
                    ('groups.stop', {'cancel_id': 'foreign-stop'}),
                ]:
                    rejected = await rpc(other, method, room_id=room, **params)
                    assert rejected.get('error', {}).get('message') == 'permission_denied', rejected
                    denials += 1
                after = await rpc(ws, 'groups.log', room_id=room)
                assert after == before, (before, after)
                state = await rpc(ws, 'groups.state', room_id=room)
                assert state['result']['driver_status']['counts'].get('running') == 1, state
            for actor, ws, _ in actors:
                stopped = await rpc(ws, 'groups.stop', room_id=actor + '-room', cancel_id='owner-stop')
                assert stopped['result']['cancelled'] >= 1, stopped
                if actor == 'alice':
                    sibling = await rpc(bob, 'groups.state', room_id='bob-room')
                    assert sibling['result']['driver_status']['counts'].get('running') == 1, sibling
            model.release.set()
            for actor, ws, _ in actors:
                async with asyncio.timeout(30):
                    while True:
                        state = await rpc(ws, 'groups.state', room_id=actor + '-room')
                        counts = state['result']['driver_status']['counts']
                        if not counts.get('running') and not counts.get('stopping'):
                            assert counts.get('cancelled', 0) >= 1, state
                            break
                        await asyncio.sleep(.1)
            assert len(model.requests) == 2, model.requests
            sent = await rpc(alice, 'groups.send', room_id='alice-room', event_id='approval',
                             payload={'text': 'APPROVE_HOSTED', 'thread_id': 'approval-thread'})
            assert sent['result']['accepted'], sent
            async with asyncio.timeout(30):
                while True:
                    state = await rpc(alice, 'groups.state', room_id='alice-room')
                    actions = state['result']['driver_status']['pending_actions']
                    approvals = [a for a in actions if a['kind'] == 'approval']
                    if approvals:
                        break
                    await asyncio.sleep(.1)
            assert removal.exists()
            action = approvals[0]
            params = {k: action[k] for k in ('member_id', 'task_id', 'execution_generation', 'request_id')}
            denied = await rpc(bob, 'groups.approve', room_id='alice-room', choice='once', **params)
            assert denied.get('error', {}).get('message') == 'permission_denied', denied
            accepted = await rpc(alice, 'groups.approve', room_id='alice-room', choice='once', **params)
            assert accepted.get('result', {}).get('approved'), accepted
            async with asyncio.timeout(30):
                while removal.exists():
                    await asyncio.sleep(.1)

            assert 'FORBIDDEN' not in json.dumps(model.requests)
            return {'actors': oidc.logins, 'rooms': ['alice-room', 'bob-room'],
                    'cross_actor_denials': denials, 'owner_stops': 2}

    try:
        with daemon(root, target, env | {'HERMES_HOME': str(target)}, barrier=False), daemon(
                root, home, env, barrier=False) as (_, desc), ExitStack() as stack:
            clients = [stack.enter_context(httpx.Client(base_url=desc['api_origin'],
                follow_redirects=False, trust_env=False, timeout=20)) for _ in range(2)]
            for credential in ('', 'Bearer invalid'):
                rejected = clients[0].post('/api/auth/ws-ticket', headers={'Authorization': credential})
                assert rejected.status_code == 401, rejected.text
            tickets = [login(client, oidc.issuer, actor)
                       for client, actor in zip(clients, ('alice', 'bob'))]
            return asyncio.run(exercise(desc, tickets))
    finally:
        model.release.set()
        for server in (oidc, model):
            server.shutdown()
            server.server_close()
        for thread in threads:
            thread.join(timeout=5)
