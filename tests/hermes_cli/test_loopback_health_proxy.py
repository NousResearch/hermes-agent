"""Internal readiness probes must reach a local service, not the outbound proxy."""
import json
import threading
import urllib.request
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import pytest


@contextmanager
def http_fixture(*, reject=False, detailed_missing=False):
    paths = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            paths.append(self.path)
            status = 503 if reject else (404 if detailed_missing and self.path in {'/health/detailed', '/json/version'} else 200)
            body = json.dumps({'status': 'ok'}).encode()
            self.send_response(status)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}', paths
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=3)


@pytest.mark.parametrize('probe', ['health', 'cdp'])
@pytest.mark.parametrize('fallback', [False, True])
def test_loopback_probe_ignores_outbound_proxy(monkeypatch, probe, fallback):
    from hermes_cli import web_server as ws, web_server_gateway
    from tui_gateway import server

    with http_fixture(reject=True) as (proxy, proxy_paths), http_fixture(detailed_missing=fallback) as (origin, paths):
        # Inject proxy discovery only; requests go over real sockets. Avoid the host's
        # registry/system proxy and any ambient NO_PROXY exception on every OS.
        monkeypatch.setattr(urllib.request, 'getproxies', lambda: {'http': proxy})
        monkeypatch.setattr(urllib.request, 'proxy_bypass', lambda host: False)
        monkeypatch.setattr(urllib.request, '_opener', None)
        if probe == 'health':
            monkeypatch.setattr(ws, '_GATEWAY_HEALTH_URL', origin)
            monkeypatch.setattr(ws, '_GATEWAY_HEALTH_TIMEOUT', 0.75)
            assert web_server_gateway._probe_gateway_health() == (True, {'status': 'ok'})
            assert paths == (['/health/detailed', '/health'] if fallback else ['/health/detailed'])
        else:
            assert server._cdp_http_reachable(urlparse(origin), timeout=0.75)
            assert paths == (['/json/version', '/json'] if fallback else ['/json/version'])
        assert proxy_paths == []


@pytest.mark.parametrize('probe', ['health', 'cdp'])
def test_remote_probe_keeps_proxy_and_failure_policy(monkeypatch, probe):
    from hermes_cli import web_server as ws, web_server_gateway
    from tui_gateway import server

    for reject in (False, True):
        with http_fixture(reject=reject) as (proxy, paths):
            monkeypatch.setattr(urllib.request, 'getproxies', lambda: {'http': proxy})
            monkeypatch.setattr(urllib.request, 'proxy_bypass', lambda host: False)
            monkeypatch.setattr(urllib.request, '_opener', None)
            origin = 'http://gateway.invalid:8765'
            if probe == 'health':
                monkeypatch.setattr(ws, '_GATEWAY_HEALTH_URL', origin)
                monkeypatch.setattr(ws, '_GATEWAY_HEALTH_TIMEOUT', 0.75)
                result = web_server_gateway._probe_gateway_health()
                assert result == ((False, None) if reject else (True, {'status': 'ok'}))
            else:
                assert server._cdp_http_reachable(urlparse(origin), timeout=0.75) is (not reject)
            assert len(paths) == (2 if reject else 1)
            assert all(path.startswith(origin) for path in paths)


def test_probe_failure_and_absent_config_remain_bounded(monkeypatch):
    from hermes_cli import web_server as ws, web_server_gateway
    from tui_gateway import server
    from agent.proxy_bypass import urlopen_bypass_proxy_for_loopback
    from unittest.mock import Mock

    monkeypatch.setattr(ws, '_GATEWAY_HEALTH_URL', '')
    assert web_server_gateway._probe_gateway_health() == (False, None)
    open_call = Mock(side_effect=TimeoutError('fixture timeout'))
    monkeypatch.setattr(urllib.request, 'urlopen', open_call)
    monkeypatch.setattr(ws, '_GATEWAY_HEALTH_URL', 'http://gateway.invalid')
    monkeypatch.setattr(ws, '_GATEWAY_HEALTH_TIMEOUT', 0.125)
    assert web_server_gateway._probe_gateway_health() == (False, None)
    assert server._cdp_http_reachable(urlparse('http://gateway.invalid'), timeout=0.125) is False
    assert len(open_call.call_args_list) == 4
    assert all(call.kwargs['timeout'] == 0.125 for call in open_call.call_args_list)
    with pytest.raises(TimeoutError):
        urlopen_bypass_proxy_for_loopback('http://gateway.invalid', timeout=0.125)

