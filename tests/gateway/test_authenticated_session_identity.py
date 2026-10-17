"""Verified dashboard identities must own separate durable authority receipts."""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from starlette.requests import Request
from starlette.websockets import WebSocket


@pytest.fixture
def oidc():
    from plugins.dashboard_auth.self_hosted import SelfHostedOIDCProvider
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(key.public_key()))
    jwk.update(kid='test-key', use='sig', alg='RS256')

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            realm = self.path.removesuffix('/.well-known/openid-configuration')
            body = {'keys': [jwk]} if self.path.endswith('/jwks') else {
                'issuer': issuer + realm, 'jwks_uri': issuer + '/jwks',
                'authorization_endpoint': issuer + '/authorize', 'token_endpoint': issuer + '/token'}
            raw = json.dumps(body).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    issuer = f'http://127.0.0.1:{server.server_port}'
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    def login(subject, realm=''):
        provider = SelfHostedOIDCProvider(issuer=issuer + realm, client_id='fixture-client')
        token = jwt.encode({'iss': issuer + realm, 'sub': subject, 'aud': 'fixture-client',
                            'iat': int(time.time()), 'exp': int(time.time()) + 600}, key, algorithm='RS256',
                           headers={'kid': 'test-key'})
        return provider.verify_session(access_token=token)

    try:
        yield login
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)


def connection(authority, session, monkeypatch):
    from hermes_cli import web_server as web
    from hermes_cli.dashboard_auth.routes import api_auth_ws_ticket
    from hermes_cli.web_server_chat import _ws_auth_reason
    from gateway.session_controls import AuthorityConnection
    # Exercise the production ticket endpoint and upgrade stamp, not a made-up Principal.
    monkeypatch.setattr(web.app.state, 'auth_required', True, raising=False)
    request = Request({'type': 'http', 'app': web.app, 'state': {'session': session},
                       'headers': [], 'method': 'POST', 'path': '/api/auth/ws-ticket'})
    return request, api_auth_ws_ticket, _ws_auth_reason, AuthorityConnection, WebSocket


@pytest.mark.asyncio
async def test_bundled_provider_issuer_subject_owns_durable_receipts(tmp_path, monkeypatch, oidc):
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_local import create_local_session
    from gateway.session_contract import Principal
    from hermes_cli.web_server_sessions import _session_mutation_context
    from hermes_state_runtime import RuntimeStoreError
    from plugins.dashboard_auth.basic import BasicAuthProvider, hash_password

    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'platform_toolsets': {'cli': []}})
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False)
    authority = await initialize_session_authority(runner, profile_id=str(tmp_path), instance_id='first')

    async def actor(session):
        request, endpoint, auth, factory, ws_type = connection(authority, session, monkeypatch)
        monkeypatch.setattr(request.app.state, 'session_authority', authority, raising=False)
        ticket = (await endpoint(request))['ticket']
        ws = ws_type({'type': 'websocket', 'path': '/api/ws', 'headers': [],
                      'query_string': ('ticket=' + ticket).encode(), 'client': ('127.0.0.1', 1)},
                     None, None)
        assert auth(ws)[0] is None
        result = factory(authority, object(), ws._hermes_auth_identity).actor
        # HTTP mutation must use precisely the WS ownership/retry key.
        monkeypatch.setattr('hermes_state._default_db_path', lambda: authority.db.db_path)
        assert _session_mutation_context(request, None)[1].subject == result.subject
        return result

    oauth = oidc('user-a')
    basic = BasicAuthProvider(username='user-a', password_hash=hash_password('fixture-password'), secret=b'fixture-secret-32-characters-long')
    password = basic.complete_password_login(username='user-a', password='fixture-password')
    owner, other, same_sub = await actor(oauth), await actor(oidc('user-b')), await actor(password)
    params = {'request_id': 'same-request', 'source': 'gui', 'cwd': str(tmp_path), 'model': 'fixture', 'toolsets': []}
    ref = create_local_session(authority, owner, params)
    from gateway.session_mutations import mutate_session
    edit = dict(session_id=ref.session_id, request_id='oauth-rename', expected_revision=0,
                operation='rename', payload={'title': 'OAuth owned'})
    receipt = await mutate_session(authority, owner, ref, edit)
    for foreign in (other, same_sub):
        with pytest.raises(RuntimeStoreError, match='permission_denied'):
            await authority.resolve(foreign, ref)
        assert create_local_session(authority, foreign, params) != ref
    # Same provider name under a different verified issuer is a different authority.
    foreign_issuer = await actor(oidc('user-a', '/other'))
    with pytest.raises(RuntimeStoreError, match='permission_denied'):
        await authority.resolve(foreign_issuer, ref)
    from tui_gateway.methods_browser_control import _principal_digest
    assert _principal_digest({'provider': oauth.provider, 'issuer': oauth.issuer, 'user_id': oauth.user_id}) != _principal_digest(
        {'provider': oauth.provider, 'issuer': oauth.issuer + '/other', 'user_id': oauth.user_id})
    runner.adapters = {}
    cold = await initialize_session_authority(runner, profile_id=str(tmp_path), instance_id='second')
    reconnect = await actor(oidc('user-a'))
    assert create_local_session(cold, reconnect, params) == ref
    assert await mutate_session(cold, reconnect, ref, edit) == receipt
    assert (await cold.resolve(reconnect, ref)).ref == ref
    # Pre-cutover remote records have no issuer provenance: never adopt on first request.
    old = create_local_session(cold, Principal('user-a', owner.profile_id, owner.capabilities, 'old'),
                               {**params, 'request_id': 'old-ambiguous'})
    for current in (reconnect, same_sub):
        with pytest.raises(RuntimeStoreError, match='permission_denied'):
            await cold.resolve(current, old)


@pytest.mark.asyncio
async def test_private_native_subject_preserves_existing_creation_and_mutation_retries(tmp_path, monkeypatch):
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_controls import AuthorityConnection
    from gateway.session_contract import Principal
    from gateway.session_local import create_local_session
    from gateway.session_mutations import mutate_session
    from hermes_state_runtime import RuntimeStoreError

    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'platform_toolsets': {'cli': []}})
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False)
    authority = await initialize_session_authority(runner, profile_id='fixture', instance_id='first')
    caps = frozenset({'session:create', 'session:read', 'session:control'})
    old = Principal('uid:1000', 'fixture', caps, 'old')
    params = {'request_id': 'original', 'source': 'cli', 'cwd': str(tmp_path), 'model': 'fixture', 'toolsets': []}
    ref = create_local_session(authority, old, params)
    native = AuthorityConnection(authority, object(), {'provider': 'local', 'user_id': old.subject,
        'profile_id': 'fixture', 'instance_id': 'first', 'capabilities': caps}).actor
    assert native.subject == old.subject
    assert create_local_session(authority, native, params) == ref
    edit = dict(session_id=ref.session_id, request_id='rename', expected_revision=0, operation='rename', payload={'title': 'Native preserved'})
    result = await mutate_session(authority, old, ref, edit)
    assert await mutate_session(authority, native, ref, edit) == result
    remote = AuthorityConnection(authority, object(), {'provider': 'local', 'user_id': old.subject}).actor
    with pytest.raises(RuntimeStoreError, match='permission_denied'):
        await authority.resolve(remote, ref)
