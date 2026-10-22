"""The ``worker.adopt`` handshake budget must survive a busy owner, and say so when it doesn't.

2026-10-22: the fan gateway was mid-turn (``time=1018.3s api_calls=46``) when a kanban worker's
``worker.adopt`` dial hit ``open_timeout=5`` and raised
``TimeoutError: timed out while waiting for handshake response``. The managed worker died, the
card kept saying ``running`` for 49 minutes, and nothing retried. A 5 s budget against a
single-process gateway that legitimately serves 1000 s turns is a coin flip; the budget is now
60 s by default (``HERMES_WORKER_RPC_OPEN_TIMEOUT`` overrides it) and a TIMED-OUT handshake is
retried with backoff.
"""
from __future__ import annotations

import json
import threading
from types import SimpleNamespace

import pytest


def _serve_owner(monkeypatch):
    """A real loopback owner whose only answer is the method name; discovery is stubbed."""
    from websockets.sync.server import serve
    from hermes_cli import gateway_client, gateway_runtime
    import hermes_cli.gateway_runtime_discovery as discovery

    def owner(ws):
        request = json.loads(ws.recv())
        ws.send(json.dumps({'jsonrpc': '2.0', 'id': request['id'],
                            'result': {'method': request['method']}}))

    server = serve(owner, '127.0.0.1', 0, subprotocols=['hermes-gateway-v1'])
    threading.Thread(target=server.serve_forever, daemon=True).start()
    endpoint = SimpleNamespace(api_origin=f'http://127.0.0.1:{server.socket.getsockname()[1]}',
                               control_home=None)
    monkeypatch.setattr(gateway_runtime, 'discover_gateway_endpoint',
                        lambda home, timeout: SimpleNamespace(state='ready', endpoint=endpoint))
    monkeypatch.setattr(discovery, 'query_identify', lambda home, timeout: {'pid': -1})
    monkeypatch.setattr(gateway_client, '_session_ticket', lambda home, endpoint, purpose: 'ticket')
    return server


def test_the_handshake_budget_defaults_to_60s_and_is_configurable(monkeypatch, tmp_path):
    import websockets.sync.client as ws_client
    from agent import runtime_session_store as store

    real_connect, seen = ws_client.connect, {}

    def spy(url, **kwargs):
        seen.update(kwargs)
        return real_connect(url, **kwargs)

    monkeypatch.setattr(ws_client, 'connect', spy)
    server = _serve_owner(monkeypatch)
    try:
        assert store.WorkerRPC(tmp_path)('worker.persist') == {'method': 'worker.persist'}
        assert seen['open_timeout'] == 60.0            # NOT the old 5 s coin flip
        monkeypatch.setenv('HERMES_WORKER_RPC_OPEN_TIMEOUT', '90')
        assert store.worker_rpc_open_timeout() == 90.0
        seen.clear()
        assert store.WorkerRPC(tmp_path)('worker.persist') == {'method': 'worker.persist'}
        assert seen['open_timeout'] == 90.0
    finally:
        server.shutdown()


def test_a_timed_out_handshake_is_retried_and_the_call_still_lands(monkeypatch, tmp_path):
    import websockets.sync.client as ws_client
    from agent import runtime_session_store as store

    real_connect = ws_client.connect
    calls = []

    def flaky(url, **kwargs):
        calls.append(kwargs['open_timeout'])
        if len(calls) < 3:
            raise TimeoutError('timed out while waiting for handshake response')
        return real_connect(url, **kwargs)

    monkeypatch.setattr(ws_client, 'connect', flaky)
    monkeypatch.setattr(store, 'WORKER_RPC_RETRY_SLEEP_SECONDS', 0.0)
    server = _serve_owner(monkeypatch)
    try:
        assert store.WorkerRPC(tmp_path)('worker.adopt') == {'method': 'worker.adopt'}
    finally:
        server.shutdown()
    assert len(calls) == 3


def test_connect_worker_link_backs_off_only_on_timeouts(monkeypatch):
    from agent import runtime_session_store as store
    delays, attempts = [], []

    def sleeper(seconds):
        delays.append(seconds)

    def timeout_twice_then_ok(url, **kwargs):
        attempts.append(kwargs['open_timeout'])
        if len(attempts) < 3:
            raise TimeoutError('busy owner')
        return 'link'

    assert store.connect_worker_link(timeout_twice_then_ok, 'ws://x', ['p'],
                                     open_timeout=60, sleep=sleeper) == 'link'
    assert delays == [1.0, 2.0]
    assert attempts == [60.0, 60.0, 60.0]

    attempts.clear()
    delays.clear()

    def auth_failure(url, **kwargs):
        attempts.append(kwargs['open_timeout'])
        raise OSError('handshake rejected')

    with pytest.raises(OSError):
        store.connect_worker_link(auth_failure, 'ws://x', ['p'], open_timeout=60, sleep=sleeper)
    assert len(attempts) == 1 and delays == []   # a rejected handshake never heals by retrying
