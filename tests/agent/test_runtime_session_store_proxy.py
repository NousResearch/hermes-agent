"""A managed worker's owner RPC is a loopback dial: a user's HTTPS_PROXY must never carry it.

(websockets reads ws/socks/https/http proxies only; a bare ALL_PROXY never reached this dial.)"""
import json
import socket
import threading
from types import SimpleNamespace


def _refusing_proxy():
    """A real HTTP proxy that records every CONNECT and refuses it (a proxy that cannot reach loopback)."""
    server = socket.create_server(('127.0.0.1', 0))
    seen = []

    def serve():
        while True:
            try:
                conn, _ = server.accept()
            except OSError:
                return
            with conn:
                seen.append(conn.recv(4096).split(b'\r\n', 1)[0].decode(errors='replace'))
                conn.sendall(b'HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n')
    threading.Thread(target=serve, daemon=True).start()
    return server, seen


def test_worker_rpc_reaches_loopback_owner_with_proxy_env(monkeypatch, tmp_path):
    from websockets.sync.server import serve
    from agent import runtime_session_store as store
    from hermes_cli import gateway_client
    from hermes_cli import gateway_runtime
    import hermes_cli.gateway_runtime_discovery as discovery

    def owner(ws):
        request = json.loads(ws.recv())
        ws.send(json.dumps({'jsonrpc': '2.0', 'id': request['id'], 'result': {'method': request['method']}}))

    proxy, seen = _refusing_proxy()
    with serve(owner, '127.0.0.1', 0, subprotocols=['hermes-gateway-v1']) as server:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        origin = f'http://127.0.0.1:{server.socket.getsockname()[1]}'
        endpoint = SimpleNamespace(api_origin=origin, control_home=None)
        monkeypatch.setattr(gateway_runtime, 'discover_gateway_endpoint',
                            lambda home, timeout: SimpleNamespace(state='ready', endpoint=endpoint))
        monkeypatch.setattr(discovery, 'query_identify', lambda home, timeout: {'pid': -1})
        monkeypatch.setattr(gateway_client, '_session_ticket', lambda home, endpoint, purpose: 'ticket')
        for name in ('NO_PROXY', 'no_proxy', 'HTTP_PROXY', 'http_proxy', 'HTTPS_PROXY', 'https_proxy',
                     'ALL_PROXY', 'all_proxy'):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv('HTTPS_PROXY', f'http://127.0.0.1:{proxy.getsockname()[1]}')
        try:
            assert store.WorkerRPC(tmp_path)('worker.persist') == {'method': 'worker.persist'}
        finally:
            server.shutdown()
            proxy.close()
    assert seen == []
