"""Connect-only access uses scoped credentials and never exposes secret fields."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from agent.secret_scope import set_secret_scope, reset_secret_scope, set_multiplex_active
from agent.vault_backends.onepassword import OnePasswordLoginBackend

V = 'a' * 26
I = 'b' * 26
SYNTHETIC_VALUE = 'synthetic-only-passphrase'


def test_connect_native_metadata_resolution_and_fail_closed(monkeypatch):
    requests = []
    state = {'status': 200}
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_): pass
        def do_GET(self):
            requests.append((self.path, self.headers.get('Authorization')))
            item = {'id': I, 'vault': {'id': V}, 'category': 'LOGIN', 'title': 'Example',
                    'urls': [{'href': 'https://example.com/login'}]}
            routes = {'/v1/vaults': [{'id': V}], f'/v1/vaults/{V}/items': [item],
                      f'/v1/vaults/{V}/items/{I}': dict(item, fields=[
                          {'purpose': 'USERNAME', 'value': 'synthetic@example.com'},
                          {'purpose': 'PASSWORD', 'value': SYNTHETIC_VALUE}])}
            self.send_response(state['status']); self.end_headers()
            self.wfile.write(json.dumps(routes.get(self.path, {})).encode())
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    monkeypatch.delenv('OP_SERVICE_ACCOUNT_TOKEN', raising=False)
    monkeypatch.setenv('OP_CONNECT_TOKEN', 'wrong-parent-token')
    set_multiplex_active(True)
    token = set_secret_scope({'OP_CONNECT_HOST': f'http://127.0.0.1:{server.server_port}', 'OP_CONNECT_TOKEN': 'scoped-token'})
    try:
        backend = OnePasswordLoginBackend()
        assert backend.is_unlocked()
        items = backend.list_items()
        assert len(items) == 1 and items[0].origin == 'https://example.com'
        assert SYNTHETIC_VALUE not in repr(items)
        meta = backend.get_meta(items[0].id)
        assert meta.identifier == 'synthetic@example.com'
        assert backend.resolve_password(items[0].id) == SYNTHETIC_VALUE
        assert all(auth == 'Bearer scoped-token' for _, auth in requests)
        for status in (401, 403, 302, 500):
            state['status'] = status
            with pytest.raises(RuntimeError) as exc:
                backend.list_items()
            assert SYNTHETIC_VALUE not in str(exc.value) and 'scoped-token' not in str(exc.value)
        prior = len(requests)
        with pytest.raises(ValueError): backend.get_meta('op:connect:../bad:item')
        assert len(requests) == prior
        empty = set_secret_scope({})
        try:
            assert not OnePasswordLoginBackend().is_unlocked()
        finally: reset_secret_scope(empty)
    finally:
        reset_secret_scope(token); set_multiplex_active(False)
        server.shutdown(); server.server_close(); thread.join()


OTP_URI = ('otpauth://totp/Example:me?secret=GEZDGNBVGY3TQOJQGEZDGNBVGY3TQOJQ'
           '&digits=8&period=3600&algorithm=SHA1')


def _connect_server(item_fields):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_): pass
        def do_GET(self):
            item = {'id': I, 'vault': {'id': V}, 'category': 'LOGIN', 'title': 'Example',
                    'urls': [{'href': 'https://example.com/login'}]}
            routes = {'/v1/vaults': [{'id': V}], f'/v1/vaults/{V}/items': [item],
                      f'/v1/vaults/{V}/items/{I}': dict(item, fields=item_fields)}
            self.send_response(200); self.end_headers()
            self.wfile.write(json.dumps(routes.get(self.path, {})).encode())
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    return server, thread


def _scoped_backend(server):
    scope = set_secret_scope({'OP_CONNECT_HOST': f'http://127.0.0.1:{server.server_port}',
                              'OP_CONNECT_TOKEN': 'scoped-token'})
    return OnePasswordLoginBackend(), scope


def test_connect_one_time_password_field_is_announced_and_minted(monkeypatch):
    """A Connect item storing an otpauth:// URI is announced as automatic AND actually minted.

    Regression: the Connect branch of ``resolve_otp`` returned None unconditionally, so even an
    item with a usable authenticator seed could never produce a code — ``browser_vault_enter_code``
    fell through to asking the user (unavailable headless) instead of minting it.
    """
    from agent.vault_store import normalize_otp_secret, totp_now
    fields = [{'purpose': 'USERNAME', 'value': 'synthetic@example.com'},
              {'purpose': 'PASSWORD', 'value': SYNTHETIC_VALUE},
              {'id': 'TOTP_synthetic', 'label': 'one-time password', 'type': 'OTP', 'value': OTP_URI}]
    server, thread = _connect_server(fields)
    monkeypatch.delenv('OP_SERVICE_ACCOUNT_TOKEN', raising=False)
    set_multiplex_active(True)
    backend, scope = _scoped_backend(server)
    try:
        handle = backend.list_items()[0].id
        # The Connect LIST route carries no fields, so a field-less list item must NOT claim
        # automatic 2FA; the authoritative answer is the item detail route.
        assert backend.list_items()[0].has_otp is False
        seed = normalize_otp_secret(OTP_URI)
        # Sample the clock on BOTH sides of the production call: the expectation can no longer
        # straddle a step boundary with a slower/faster mint.
        before = totp_now(seed)
        code = backend.resolve_otp(handle)
        assert code in {before, totp_now(seed)}
        assert code is not None and code.isdigit() and len(code) == 8
        meta = backend.get_meta(handle)
        assert meta.has_otp is True, 'a stored authenticator seed must be announced as automatic'
        assert OTP_URI not in repr(meta)
        assert seed not in repr(meta), 'the bare seed must not surface in metadata either'
    finally:
        reset_secret_scope(scope); set_multiplex_active(False)
        server.shutdown(); server.server_close(); thread.join()


def test_connect_without_usable_seed_stays_single_factor(monkeypatch):
    """No OTP field, or an unusable one, must not be announced as automatic nor mint a code."""
    cases = [
        [{'purpose': 'USERNAME', 'value': 'synthetic@example.com'},
         {'purpose': 'PASSWORD', 'value': SYNTHETIC_VALUE}],
        [{'purpose': 'PASSWORD', 'value': SYNTHETIC_VALUE},
         {'type': 'OTP', 'value': 'not-a-seed!!'}],
    ]
    for fields in cases:
        server, thread = _connect_server(fields)
        monkeypatch.delenv('OP_SERVICE_ACCOUNT_TOKEN', raising=False)
        set_multiplex_active(True)
        backend, scope = _scoped_backend(server)
        try:
            handle = backend.list_items()[0].id
            assert backend.get_meta(handle).has_otp is False
            assert backend.resolve_otp(handle) is None
        finally:
            reset_secret_scope(scope); set_multiplex_active(False)
            server.shutdown(); server.server_close(); thread.join()
