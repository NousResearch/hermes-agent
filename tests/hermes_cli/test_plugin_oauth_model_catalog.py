"""OAuth plugin catalogs use the owning profile's pooled credential on every surface."""

import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


@pytest.fixture
def catalog_server():
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            bearer = self.headers.get('Authorization')
            requests.append(bearer)
            self.send_response(503 if bearer == 'Bearer unavailable' else 200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'data': [{'id': f'{bearer.removeprefix("Bearer ")}-model'}]}).encode())

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}/v1', requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@contextmanager
def profile_home(home):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    token = set_hermes_home_override(home)
    try:
        yield
    finally:
        reset_hermes_home_override(token)


def install_plugin(home, auth_type, endpoint, bearer=None):
    from agent.credential_pool import AUTH_TYPE_OAUTH, PooledCredential, load_pool
    plugin = home / 'plugins' / 'model-providers' / 'catalog-probe'
    plugin.mkdir(parents=True)
    (home / 'config.yaml').write_text('plugins:\n  enabled: [catalog-probe]\n')
    (plugin / 'plugin.yaml').write_text('name: catalog-probe\nkind: model-provider\nversion: 0.0.1\n')
    (plugin / '__init__.py').write_text(
        'from providers import register_provider\nfrom providers.base import ProviderProfile\n'
        f'register_provider(ProviderProfile(name="catalog-probe", auth_type={auth_type!r}, '
        f'base_url={endpoint!r}, fallback_models=()))\n')
    if bearer:
        with profile_home(home):
            load_pool('catalog-probe').add_entry(PooledCredential(
                provider='catalog-probe', id='test', label='test', auth_type=AUTH_TYPE_OAUTH,
                priority=0, source='manual:test', access_token=bearer))


@pytest.mark.parametrize('auth_type', ['oauth_external', 'oauth_device_code'])
def test_live_catalog_follows_the_profile_pool_across_a_b_a(tmp_path, catalog_server, auth_type):
    from hermes_cli.models import provider_model_ids
    endpoint, requests = catalog_server
    a, b = tmp_path / 'a', tmp_path / 'b'
    install_plugin(a, auth_type, endpoint, 'token-a')
    install_plugin(b, auth_type, endpoint, 'token-b')
    for home, bearer in [(a, 'token-a'), (b, 'token-b'), (a, 'token-a')]:
        with profile_home(home):
            assert provider_model_ids('catalog-probe', force_refresh=True) == [f'{bearer}-model']
    assert requests == ['Bearer token-a', 'Bearer token-b', 'Bearer token-a']


@pytest.mark.parametrize('bearer', [None, 'unavailable'])
def test_missing_credentials_or_failed_catalog_never_invents_a_model(tmp_path, catalog_server, bearer):
    from hermes_cli.models import provider_model_ids
    endpoint, requests = catalog_server
    home = tmp_path / 'home'
    install_plugin(home, 'oauth_external', endpoint, bearer)
    with profile_home(home):
        assert provider_model_ids('catalog-probe', force_refresh=True) == []
    assert requests == ([] if bearer is None else ['Bearer unavailable'])
