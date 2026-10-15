"""A managed worker keeps one authenticated owner connection instead of dialing per persistence call."""
import json
import threading
import time
from types import SimpleNamespace

import pytest


@pytest.fixture
def owner(monkeypatch, tmp_path):
    from websockets.sync.server import serve
    from hermes_cli import gateway_client, gateway_runtime
    import hermes_cli.gateway_runtime_discovery as discovery

    state = SimpleNamespace(connections=0, tickets=0, requests=[], drop=None, closed=threading.Event())

    def handler(ws):
        state.connections += 1
        for raw in ws:
            request = json.loads(raw)
            state.requests.append(request['method'])
            if state.drop == 'before_reply':
                break
            # Owners broadcast events to every attached transport; they must not satisfy a call.
            ws.send(json.dumps({'jsonrpc': '2.0', 'method': 'event', 'params': {'type': 'sessions.changed'}}))
            ws.send(json.dumps({'jsonrpc': '2.0', 'id': request['id'], 'result': {'n': len(state.requests)}}))
            if state.drop == 'after_reply':
                break
        state.drop = None
        ws.close()
        state.closed.set()

    def ticket(home, endpoint, purpose):
        assert purpose == 'worker-adoption'
        state.tickets += 1
        return f'ticket-{state.tickets}'

    with serve(handler, '127.0.0.1', 0, subprotocols=['hermes-gateway-v1']) as server:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        endpoint = SimpleNamespace(api_origin=f'http://127.0.0.1:{server.socket.getsockname()[1]}', control_home=None)
        monkeypatch.setattr(gateway_runtime, 'discover_gateway_endpoint',
                            lambda home, timeout: SimpleNamespace(state='ready', endpoint=endpoint))
        monkeypatch.setattr(discovery, 'query_identify', lambda home, timeout: {'pid': -1})
        monkeypatch.setattr(gateway_client, '_session_ticket', ticket)
        try:
            yield state
        finally:
            server.shutdown()


def test_one_connection_serves_every_call(owner, tmp_path):
    from agent.runtime_session_store import WorkerRPC
    rpc = WorkerRPC(tmp_path)
    assert [rpc('worker.persist')['n'] for _ in range(5)] == [1, 2, 3, 4, 5]
    assert (owner.connections, owner.tickets) == (1, 1)


def test_reconnects_with_a_fresh_ticket_after_the_owner_drops_it(owner, tmp_path):
    from agent.runtime_session_store import WorkerRPC
    rpc = WorkerRPC(tmp_path)
    owner.drop = 'after_reply'
    assert rpc('worker.persist') == {'n': 1}
    assert owner.closed.wait(5)
    assert rpc('worker.persist') == {'n': 2}
    assert (owner.connections, owner.tickets) == (2, 2)


def test_in_flight_call_on_a_dropped_connection_fails_and_is_not_resent(owner, tmp_path):
    from websockets.exceptions import ConnectionClosed
    from agent.runtime_session_store import WorkerRPC
    rpc = WorkerRPC(tmp_path)
    assert rpc('worker.persist') == {'n': 1}
    owner.drop = 'before_reply'
    started = time.monotonic()
    # The write may have landed: the caller sees the transport error (outcome unknown), never a silent retry.
    with pytest.raises(ConnectionClosed):
        rpc('worker.persist')
    assert time.monotonic() - started < 5
    assert owner.requests == ['worker.persist', 'worker.persist']
    assert rpc('worker.persist') == {'n': 3}
    assert owner.connections == 2
