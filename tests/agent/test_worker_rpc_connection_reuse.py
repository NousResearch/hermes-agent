"""A managed worker keeps one authenticated owner connection instead of dialing per persistence call."""
import contextlib
import json
import socket
import threading
import time
from types import SimpleNamespace

import pytest

from tests.waiting import wait_until


class _Relay:
    """Loopback TCP relay whose owner->worker direction can be held: the bytes an owner with a
    stalled loop would not have written yet (its replies and pongs) arrive only on release."""
    def __init__(self, port):
        self.flowing = threading.Event()
        self.flowing.set()
        self.listener = socket.create_server(('127.0.0.1', 0))
        self.port = self.listener.getsockname()[1]
        threading.Thread(target=self._accept, args=(port,), daemon=True).start()

    def _accept(self, port):
        while True:
            try:
                worker, _ = self.listener.accept()
            except OSError:
                return
            owner = socket.create_connection(('127.0.0.1', port))
            threading.Thread(target=self._pump, args=(worker, owner, None), daemon=True).start()
            threading.Thread(target=self._pump, args=(owner, worker, self.flowing), daemon=True).start()

    @staticmethod
    def _pump(src, dst, gate):
        try:
            while data := src.recv(65536):
                if gate is not None:
                    gate.wait()
                dst.sendall(data)
        except OSError:
            pass
        finally:
            for sock in (src, dst):
                with contextlib.suppress(OSError):
                    sock.shutdown(socket.SHUT_RDWR)

    def stall(self, seconds):
        self.flowing.clear()
        threading.Timer(seconds, self.flowing.set).start()


@pytest.fixture
def owner(monkeypatch, tmp_path):
    from websockets.sync.server import serve
    from hermes_cli import gateway_client, gateway_runtime
    import hermes_cli.gateway_runtime_discovery as discovery

    state = SimpleNamespace(connections=0, tickets=0, requests=[], drop=None, closed=threading.Event(),
                            pongs=[], flooded=threading.Event(), relay=None, live=None)

    def handler(ws):
        state.connections += 1
        state.live = ws
        ws.close_timeout = 1
        for raw in ws:
            request = json.loads(raw)
            state.requests.append(request['method'])
            if state.drop == 'before_reply':
                break
            if request['method'] == 'stall':
                state.relay.stall(request['params']['seconds'])
            # Owners broadcast events to every attached transport; they must not satisfy a call.
            ws.send(json.dumps({'jsonrpc': '2.0', 'method': 'event', 'params': {'type': 'sessions.changed'}}))
            if request['method'] == 'fail':
                ws.send(json.dumps({'jsonrpc': '2.0', 'id': request['id'],
                                    'error': {'code': -32000, 'message': 'stale_generation'}}))
                continue
            ws.send(json.dumps({'jsonrpc': '2.0', 'id': request['id'],
                                'result': {'n': len(state.requests), **request.get('params', {})}}))
            if request['method'] == 'flood':
                # sessions.changed keeps flowing while the worker is busy in a model call; the owner's
                # keepalive then needs a pong, which a worker that stopped reading never sends.
                for index in range(request['params']['events']):
                    time.sleep(0.005)  # health: allow HX013 -- separate broadcasts, not one coalesced socket read
                    ws.send(json.dumps({'jsonrpc': '2.0', 'method': 'event',
                                        'params': {'type': 'sessions.changed', 'index': index}}))
                state.pongs.append(ws.ping().wait(2))
                state.flooded.set()
                if not state.pongs[-1]:
                    break
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
        state.relay = _Relay(server.socket.getsockname()[1])
        endpoint = SimpleNamespace(api_origin=f'http://127.0.0.1:{state.relay.port}', control_home=None)
        monkeypatch.setattr(gateway_runtime, 'discover_gateway_endpoint',
                            lambda home, timeout: SimpleNamespace(state='ready', endpoint=endpoint))
        monkeypatch.setattr(discovery, 'query_identify', lambda home, timeout: {'pid': -1})
        monkeypatch.setattr(gateway_client, '_session_ticket', ticket)
        try:
            yield state
        finally:
            state.relay.flowing.set()
            state.relay.listener.close()
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
    # The worker observes the idle close before its next call.
    wait_until(lambda: not rpc.link.reader.is_alive(), timeout=5)
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


def _owner_closes_before_the_frame_is_written(owner, monkeypatch):
    """Boundary hook on the library's send: while ``owner.sabotage`` is set, the idle check has already
    passed when the owner's close lands (and the worker observes it), just before websockets writes."""
    from websockets.protocol import State
    from websockets.sync import client
    real, dialed = client.connect, []

    def connect(*args, **kwargs):
        ws = real(*args, **kwargs)
        dialed.append(ws)
        real_send = ws.send

        def send(message, *a, **k):
            if owner.sabotage:
                owner.sabotage -= 1
                wait_until(lambda: owner.connections == len(dialed), timeout=5)
                owner.live.close()
                wait_until(lambda: ws.protocol.state is State.CLOSED, timeout=5)
            return real_send(message, *a, **k)
        ws.send = send
        return ws
    monkeypatch.setattr(client, 'connect', connect)


def test_owner_close_between_the_idle_check_and_the_send_redials_and_sends_once(owner, tmp_path, monkeypatch):
    """websockets raises ConnectionClosed from send only before the frame is written: nothing reached
    the owner, so the call redials once with a fresh ticket instead of failing as outcome-unknown."""
    from agent.runtime_session_store import WorkerRPC
    owner.sabotage = 0
    _owner_closes_before_the_frame_is_written(owner, monkeypatch)
    rpc = WorkerRPC(tmp_path)
    assert rpc('worker.persist') == {'n': 1}
    owner.sabotage = 1
    assert rpc('worker.persist') == {'n': 2}
    assert owner.requests == ['worker.persist', 'worker.persist']
    assert (owner.connections, owner.tickets) == (2, 2)
    assert rpc('worker.persist') == {'n': 3}
    assert owner.connections == 2


def test_a_redial_whose_send_also_fails_is_not_retried_again(owner, tmp_path, monkeypatch):
    from websockets.exceptions import ConnectionClosed
    from agent.runtime_session_store import WorkerRPC
    owner.sabotage = 0
    _owner_closes_before_the_frame_is_written(owner, monkeypatch)
    rpc = WorkerRPC(tmp_path)
    assert rpc('worker.persist') == {'n': 1}
    owner.sabotage = 2
    with pytest.raises(ConnectionClosed):
        rpc('worker.persist')
    assert owner.requests == ['worker.persist']
    assert (owner.connections, owner.tickets) == (2, 2)
    assert rpc.link is None


def test_idle_broadcasts_past_the_receive_high_water_mark_keep_the_connection(owner, tmp_path):
    """Events pushed between calls (far over websockets' 16-frame receive mark) are consumed while the
    worker is idle, so the owner's keepalive is answered and the next call reuses the connection."""
    from agent.runtime_session_store import WorkerRPC
    rpc = WorkerRPC(tmp_path)
    assert rpc('flood', events=64)['n'] == 1
    assert owner.flooded.wait(10)
    assert rpc('worker.persist')['n'] == 2
    assert owner.pongs == [True]
    assert (owner.connections, owner.tickets) == (1, 1)


def test_owner_stall_shorter_than_the_receipt_budget_does_not_fail_on_a_ping_deadline(owner, tmp_path, monkeypatch):
    """The receipt budget (20 s) is the only per-call deadline: a library keepalive whose pong is held
    by the same stall must not close the connection first. The library defaults (20 s / 20 s) are
    scaled down so an inherited keepalive would fire inside the stall."""
    from websockets.sync import client
    from agent.runtime_session_store import WorkerRPC
    real = client.connect

    def connect(*args, **kwargs):
        kwargs.setdefault('ping_interval', 0.2)
        kwargs.setdefault('ping_timeout', 0.2)
        return real(*args, **kwargs)
    monkeypatch.setattr(client, 'connect', connect)
    rpc = WorkerRPC(tmp_path)
    assert rpc('worker.persist')['n'] == 1
    started = time.monotonic()
    assert rpc('stall', seconds=1.5) == {'n': 2, 'seconds': 1.5}
    assert time.monotonic() - started >= 1.4
    assert rpc('worker.persist')['n'] == 3
    assert owner.connections == 1


def test_concurrent_callers_are_serialized_and_each_gets_its_own_reply(owner, tmp_path):
    from agent.runtime_session_store import WorkerRPC
    rpc = WorkerRPC(tmp_path)
    replies, errors = {'a': [], 'b': []}, []

    def caller(tag):
        try:
            for index in range(25):
                replies[tag].append(rpc('worker.persist', tag=tag, index=index))
        except Exception as exc:  # surfaced below
            errors.append(exc)
    threads = [threading.Thread(target=caller, args=(tag,)) for tag in replies]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)
    assert errors == []
    for tag, got in replies.items():
        assert [(reply['tag'], reply['index']) for reply in got] == [(tag, index) for index in range(25)]
    assert sorted(reply['n'] for got in replies.values() for reply in got) == list(range(1, 51))
    assert (owner.connections, owner.tickets) == (1, 1)


def test_an_owner_error_reply_keeps_the_connection_for_the_next_call(owner, tmp_path):
    from agent.runtime_session_store import WorkerPersistenceError, WorkerRPC
    rpc = WorkerRPC(tmp_path)
    with pytest.raises(WorkerPersistenceError, match='stale_generation'):
        rpc('fail')
    assert rpc('worker.persist') == {'n': 2}
    assert owner.requests == ['fail', 'worker.persist']
    assert (owner.connections, owner.tickets) == (1, 1)
