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
