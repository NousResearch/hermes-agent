"""The adopt handshake's budget: a busy owner is waited out, a broken owner still fails fast.

Run 730 died in the adopt handshake: ``TimeoutError`` against an owner serving a 1,018 s turn,
because the link asked for exactly 5 s and never asked again. The budget is
``WORKER_RPC_OPEN_TIMEOUT_SECONDS`` (default 60) per attempt with a bounded retry, and only a
handshake TIMEOUT is retried — a refusal (auth, subprotocol, socket) is the broken-link answer and
must reach the caller on the first attempt.
"""
import json
import os
import socket
import threading
import time
from types import SimpleNamespace

import pytest
from websockets.datastructures import Headers
from websockets.exceptions import InvalidHandshake, InvalidStatus
from websockets.http11 import Response
from websockets.sync.server import serve

from agent import runtime_session_store as store

SUBPROTOCOL = 'hermes-gateway-v1'
BUDGET_ENVS = ('WORKER_RPC_OPEN_TIMEOUT_SECONDS', 'HERMES_WORKER_RPC_OPEN_TIMEOUT')


def _clear_budget(monkeypatch):
    for name in BUDGET_ENVS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def owner_surface(monkeypatch, tmp_path):
    """Discovery, identify and tickets stubbed; the real loopback socket comes from a fake owner."""
    import hermes_cli.gateway_runtime_discovery as discovery
    from hermes_cli import gateway_client, gateway_runtime

    tickets = []

    def ticket(home, endpoint, purpose):
        tickets.append(purpose)
        return f'ticket-{len(tickets)}'

    def point_at(port, state='ready'):
        endpoint = SimpleNamespace(api_origin=f'http://127.0.0.1:{port}', control_home=None)
        monkeypatch.setattr(gateway_runtime, 'discover_gateway_endpoint',
                            lambda home, timeout: SimpleNamespace(state=state, endpoint=endpoint))
        return endpoint

    monkeypatch.setattr(discovery, 'query_identify', lambda home, timeout: {'pid': -1})
    monkeypatch.setattr(gateway_client, '_session_ticket', ticket)
    return SimpleNamespace(point_at=point_at, tickets=tickets, home=tmp_path)


def _serve(handler, process_request=None):
    """A websockets owner on loopback; its bound port is known before serve_forever starts."""
    server = serve(handler, '127.0.0.1', 0, subprotocols=[SUBPROTOCOL], process_request=process_request)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, server.socket.getsockname()[1]


def _answering_owner(requests):
    def handler(ws):
        for raw in ws:
            request = json.loads(raw)
            requests.append(request['method'])
            ws.send(json.dumps({'jsonrpc': '2.0', 'id': request['id'], 'result': {'ok': True}}))
    return handler


def _hold_until(deadline, seen):
    """A busy owner: it accepts the socket and answers the upgrade only once its loop is free."""
    def process_request(connection, request):
        seen.append(time.monotonic())
        while time.monotonic() < deadline:
            time.sleep(0.01)
        return None
    return process_request


# --- the helper's own invariants, no sockets -------------------------------------------------


def test_each_attempt_mints_its_own_ticket_and_the_backoff_doubles():
    """A single-use ticket would be refused as expired after 30 s: every attempt mints its own."""
    tickets, budgets, pauses = [], [], []

    def target():
        tickets.append(f'ws://owner/ticket-{len(tickets) + 1}')
        return tickets[-1], [SUBPROTOCOL]

    def busy_twice_then_connected(url, protocols, budget):
        budgets.append(budget)
        if len(budgets) < 3:
            raise TimeoutError('timed out while waiting for handshake response')
        return 'link'

    assert store.connect_worker_link(target, open_timeout=1.5, sleep=pauses.append,
                                     connect=busy_twice_then_connected) == 'link'
    assert tickets == ['ws://owner/ticket-1', 'ws://owner/ticket-2', 'ws://owner/ticket-3']
    assert budgets == [1.5, 1.5, 1.5], 'every attempt gets the whole budget'
    assert pauses == [store.WORKER_RPC_RETRY_SLEEP_SECONDS, 2 * store.WORKER_RPC_RETRY_SLEEP_SECONDS]


def test_a_refusal_is_never_retried_and_propagates_on_the_first_attempt():
    tickets, pauses = [], []

    def target():
        tickets.append(1)
        return 'ws://owner/ticket', [SUBPROTOCOL]

    def refused(url, protocols, budget):
        raise PermissionError('invalid bootstrap ticket')

    with pytest.raises(PermissionError, match='invalid bootstrap ticket'):
        store.connect_worker_link(target, sleep=pauses.append, connect=refused)
    assert tickets == [1], 'an auth/protocol refusal is not a busy owner'
    assert pauses == []


def test_a_dead_owner_raises_the_last_timeout_after_the_bounded_attempts():
    tickets, pauses = [], []

    def target():
        tickets.append(1)
        return 'ws://owner/ticket', [SUBPROTOCOL]

    def never(url, protocols, budget):
        raise TimeoutError('timed out while waiting for handshake response')

    with pytest.raises(TimeoutError, match='timed out while waiting'):
        store.connect_worker_link(target, open_timeout=0.2, sleep=pauses.append, connect=never)
    assert len(tickets) == store.WORKER_RPC_CONNECT_ATTEMPTS == 3, 'bounded, never unbounded'
    assert pauses == [store.WORKER_RPC_RETRY_SLEEP_SECONDS, 2 * store.WORKER_RPC_RETRY_SLEEP_SECONDS]


# --- end to end against a real loopback owner ------------------------------------------------


def test_a_busy_owner_that_answers_late_still_yields_a_live_worker(monkeypatch, owner_surface):
    _clear_budget(monkeypatch)
    monkeypatch.setenv('WORKER_RPC_OPEN_TIMEOUT_SECONDS', '0.5')
    monkeypatch.setattr(store, 'WORKER_RPC_RETRY_SLEEP_SECONDS', 0.05)
    budget = store.worker_rpc_open_timeout()
    assert budget == 0.5
    # 1.8 budgets of busy loop: the first handshake times out, the redial lands after it frees up.
    handshakes, requests = [], []
    server, port = _serve(_answering_owner(requests), _hold_until(time.monotonic() + 1.8 * budget, handshakes))
    try:
        owner_surface.point_at(port)
        began = time.monotonic()
        assert store.WorkerRPC(owner_surface.home)('worker.adopt') == {'ok': True}
        elapsed = time.monotonic() - began
    finally:
        server.shutdown()
    assert elapsed >= 1.7 * budget, 'the redial waited out the busy window instead of giving up'
    assert len(handshakes) == 2, 'one timed-out handshake and the redial that landed'
    assert requests == ['worker.adopt']
    assert owner_surface.tickets == ['worker-adoption', 'worker-adoption'], 'a ticket is single-use'


def test_an_owner_that_never_answers_the_handshake_fails_inside_the_configured_budget(monkeypatch, owner_surface):
    _clear_budget(monkeypatch)
    monkeypatch.setenv('WORKER_RPC_OPEN_TIMEOUT_SECONDS', '0.4')
    monkeypatch.setattr(store, 'WORKER_RPC_RETRY_SLEEP_SECONDS', 0.02)
    budget, tries = store.worker_rpc_open_timeout(), store.WORKER_RPC_CONNECT_ATTEMPTS
    release, handshakes = threading.Event(), []

    def never_answers(connection, request):
        handshakes.append(1)
        release.wait(30)
        return None

    server, port = _serve(_answering_owner([]), never_answers)
    try:
        owner_surface.point_at(port)
        began = time.monotonic()
        with pytest.raises(TimeoutError):
            store.WorkerRPC(owner_surface.home)('worker.persist')
        elapsed = time.monotonic() - began
    finally:
        release.set()
        server.shutdown()
    assert len(handshakes) == tries == 3, 'every attempt was made and then it gave up'
    assert elapsed >= tries * budget * 0.9, f'the retries did not wait their budgets: {elapsed:.2f}s'
    assert elapsed <= tries * budget + 0.5, f'gave up outside attempts x budget: {elapsed:.2f}s'


def test_an_auth_refusal_fails_fast_without_a_retry(monkeypatch, owner_surface):
    _clear_budget(monkeypatch)
    handshakes = []

    def refuse(connection, request):
        handshakes.append(1)
        return Response(401, 'Unauthorized', Headers())

    server, port = _serve(_answering_owner([]), refuse)
    try:
        owner_surface.point_at(port)
        began = time.monotonic()
        with pytest.raises(InvalidStatus):
            store.WorkerRPC(owner_surface.home)('worker.adopt')
        elapsed = time.monotonic() - began
    finally:
        server.shutdown()
    assert handshakes == [1], 'a rejected ticket is not retried'
    assert elapsed < 5, f'a refusal must not wait out the 60 s budget: {elapsed:.2f}s'


def test_a_protocol_violation_fails_fast_without_a_retry(monkeypatch, owner_surface):
    """A server that answers 101 without the upgrade headers: a protocol error, not a busy owner."""
    _clear_budget(monkeypatch)
    listener = socket.create_server(('127.0.0.1', 0))
    seen = []

    def reply_bogus_upgrade():
        while True:
            try:
                conn, _ = listener.accept()
            except OSError:
                return
            with conn:
                seen.append(conn.recv(65536))
                conn.sendall(b'HTTP/1.1 101 Switching Protocols\r\n\r\n')
                time.sleep(0.5)  # let the client read the response before the FIN

    threading.Thread(target=reply_bogus_upgrade, daemon=True).start()
    try:
        owner_surface.point_at(listener.getsockname()[1])
        began = time.monotonic()
        with pytest.raises(InvalidHandshake) as refusal:
            store.WorkerRPC(owner_surface.home)('worker.adopt')
        elapsed = time.monotonic() - began
    finally:
        listener.close()
    assert not isinstance(refusal.value, TimeoutError)
    assert len(seen) == 1, 'a protocol violation is not retried'
    assert elapsed < 5, f'a protocol error must not wait out the 60 s budget: {elapsed:.2f}s'


def test_the_adopt_budget_reads_the_environment(monkeypatch):
    _clear_budget(monkeypatch)
    assert store.worker_rpc_open_timeout() == store.WORKER_RPC_OPEN_TIMEOUT_SECONDS == 60.0
    monkeypatch.setenv('WORKER_RPC_OPEN_TIMEOUT_SECONDS', '2.5')
    assert store.worker_rpc_open_timeout() == 2.5
    monkeypatch.setenv('HERMES_WORKER_RPC_OPEN_TIMEOUT', '90')
    assert store.worker_rpc_open_timeout() == 2.5, 'the documented name wins over the earlier alias'
    monkeypatch.delenv('WORKER_RPC_OPEN_TIMEOUT_SECONDS')
    assert store.worker_rpc_open_timeout() == 90.0, "the earlier attempt's name still works"
    monkeypatch.delenv('HERMES_WORKER_RPC_OPEN_TIMEOUT')
    for bad in ('', 'soon', '0', '-3', 'nan', 'inf'):
        monkeypatch.setenv('WORKER_RPC_OPEN_TIMEOUT_SECONDS', bad)
        assert store.worker_rpc_open_timeout() == 60.0, f'{bad!r} must keep the default'


@pytest.mark.skipif(not os.environ.get('HERMES_ADOPT_SLOW_PROOF'),
                    reason='real-scale proof (~100 s at the default budget): set HERMES_ADOPT_SLOW_PROOF=1')
def test_a_100s_busy_owner_is_waited_out_at_the_default_budget(monkeypatch, owner_surface):
    """The run-730 shape at real scale: the owner's loop is busy past one whole default budget."""
    _clear_budget(monkeypatch)
    assert store.worker_rpc_open_timeout() == 60.0
    handshakes, requests = [], []
    server, port = _serve(_answering_owner(requests), _hold_until(time.monotonic() + 100, handshakes))
    try:
        owner_surface.point_at(port)
        began = time.monotonic()
        assert store.WorkerRPC(owner_surface.home)('worker.adopt') == {'ok': True}
        elapsed = time.monotonic() - began
    finally:
        server.shutdown()
    assert elapsed >= 95, f'one 60 s budget must have expired first: {elapsed:.2f}s'
    assert len(handshakes) == 2, 'the 100 s handshake needs the second attempt to land'
    assert requests == ['worker.adopt']
