"""The ``worker.adopt`` handshake must survive an owner that is busy inside a long turn.

Run 730 (fan/managed-worker.log, 2026-10-22) died on ``TimeoutError: timed out while waiting
for handshake response``: the dial budget was a hard-coded 5 s while the owner's event loop was
serving a 1018 s turn, and nothing retried it.

A flat larger budget does not fix that. The bootstrap ticket is minted BEFORE the dial and
redeemed at admission (gateway/run_api.py), and ``TicketStore.TTL_SECONDS`` refuses a ticket
older than 30 s -- so an attempt that outlives the TTL is answered with a *refusal*, not a
timeout, and a retry predicate that (correctly) retries only timeouts would never retry it.
The shape that works, pinned here:

  T1  a gateway that accepts the socket but answers the handshake only after ~100 s still
      yields a live worker;
  T2  a gateway that never answers still fails within the configured budget (bounded, not
      infinite patience);
  T3  a refused handshake (auth/protocol) is not retried -- a genuinely broken link fails fast;
  T4  every attempt mints its OWN ticket, and the budget stays inside the bootstrap TTL.
"""

from __future__ import annotations

import socket
import threading
import time

import pytest

from agent.runtime_session_store import (
    WORKER_RPC_OPEN_TIMEOUT_SECONDS,
    connect_worker_link,
)
from gateway.runtime_bootstrap import TicketStore
from hermes_cli.gateway_client import GATEWAY_WS_PROTOCOL, GATEWAY_WS_TICKET_PREFIX

PROFILE = 'worker'


def _ticket_protocol(ticket: str) -> str:
    return GATEWAY_WS_TICKET_PREFIX + ticket


def _ticket_from(request) -> str:
    """The ticket the client offered as a second WebSocket subprotocol."""
    offered = request.headers.get('Sec-WebSocket-Protocol', '')
    for name in (part.strip() for part in offered.split(',')):
        if name.startswith(GATEWAY_WS_TICKET_PREFIX):
            return name[len(GATEWAY_WS_TICKET_PREFIX):]
    return ''


def _quiet_thread_exceptions():
    """An abandoned attempt's handshake write hits a closed socket -- the same broken pipe the
    real owner logs when a client gives up. It must not turn into test noise."""
    previous = threading.excepthook

    def swallow(args):
        if not issubclass(args.exc_type, (ConnectionError, OSError)):
            previous(args)

    threading.excepthook = swallow
    return previous


class _Owner:
    """A fake gateway: real ticket store, real WebSocket upgrade, delayed by ``stall`` seconds."""

    def __init__(self, stall: float = 0.0, *, refuse: bool = False):
        from websockets.sync.server import serve

        self.store = TicketStore('instance-test', frozenset({PROFILE}))
        self.refused = 0
        self.accepted = 0
        self._refuse = refuse
        self._gate = threading.Event()
        self._deadline = time.monotonic() + stall
        if stall:
            timer = threading.Timer(stall, self._gate.set)
            timer.daemon = True
            timer.start()
        else:
            self._gate.set()
        self._server = serve(
            self._serve,
            '127.0.0.1',
            0,
            subprotocols=[GATEWAY_WS_PROTOCOL],
            select_subprotocol=lambda conn, offered: GATEWAY_WS_PROTOCOL,
            process_request=self._process_request,
        )
        self.port = self._server.socket.getsockname()[1]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def _process_request(self, connection, request):
        from websockets.http11 import Response

        # The owner's loop is busy until the deadline; the upgrade waits here.
        self._gate.wait()
        if self._refuse:
            self.refused += 1
            return Response(403, 'Forbidden', b'')
        try:
            self.store.redeem(_ticket_from(request), profile_id=PROFILE, purpose='worker-adoption')
        except PermissionError:
            self.refused += 1
            return Response(403, 'Forbidden', b'')
        self.accepted += 1
        return None

    def _serve(self, connection):
        connection.close()

    def dial(self, budget):
        """Exactly what ``WorkerRPC._connection`` does: mint a FRESH ticket, then dial."""
        ticket = self.store.mint(profile_id=PROFILE, subject='sid:test', purpose='worker-adoption')
        from websockets.sync.client import connect
        return connect(f'ws://127.0.0.1:{self.port}/api/ws',
                       subprotocols=[GATEWAY_WS_PROTOCOL, _ticket_protocol(ticket)],
                       open_timeout=budget)

    def close(self):
        self._gate.set()
        self._server.shutdown()


def test_a_gateway_that_answers_after_100s_yields_a_live_worker():
    """T1: the run-730 scenario, one scale down from the 1018 s turn."""
    previous = _quiet_thread_exceptions()
    owner = _Owner(stall=100.0)
    minted = []
    original_dial = owner.dial

    def recording_dial(budget):
        minted.append(budget)
        return original_dial(budget)

    try:
        started = time.monotonic()
        link = connect_worker_link(recording_dial, sleep=time.sleep)
        elapsed = time.monotonic() - started
    finally:
        owner.close()
        threading.excepthook = previous

    assert link is not None
    assert link.subprotocol == GATEWAY_WS_PROTOCOL
    assert elapsed >= 100.0, f'a live worker after {elapsed:.1f}s means the gate was not held'
    assert owner.accepted == 1
    assert len(minted) >= 4, 'a 100 s stall outlasts one 25 s attempt'
    assert all(budget == WORKER_RPC_OPEN_TIMEOUT_SECONDS for budget in minted)


def test_the_hard_coded_5s_dial_still_dies_where_the_new_one_lives():
    """Before/after on ONE gateway: run 730's exact dial fails, the retried one attaches.

    This is the control for the test above -- without it, a passing 100 s test proves only that
    some wait eventually succeeds, not that the 5 s budget was the defect.
    """
    previous = _quiet_thread_exceptions()
    owner = _Owner(stall=20.0)
    try:
        # The pre-fix dial, verbatim: one ticket, 5 s, no second attempt.
        from websockets.sync.client import connect
        ticket = owner.store.mint(profile_id=PROFILE, subject='sid:test', purpose='worker-adoption')
        started = time.monotonic()
        with pytest.raises(TimeoutError):
            connect(f'ws://127.0.0.1:{owner.port}/api/ws',
                    subprotocols=[GATEWAY_WS_PROTOCOL, _ticket_protocol(ticket)],
                    open_timeout=5)
        assert time.monotonic() - started < 10

        link = connect_worker_link(owner.dial, sleep=time.sleep)
    finally:
        owner.close()
        threading.excepthook = previous

    assert link is not None
    assert link.subprotocol == GATEWAY_WS_PROTOCOL
    # Two admissions: the abandoned 5 s dial's ticket is still inside the TTL when the gate opens
    # (the real owner admits it too, then writes into a dead socket), plus the surviving attempt.
    assert owner.accepted >= 1


def test_a_gateway_that_never_answers_fails_within_the_configured_budget():
    """T2: patience is bounded -- a dead owner fails, it does not hang the dispatch."""
    listener = socket.socket()
    listener.bind(('127.0.0.1', 0))
    listener.listen(8)
    port = listener.getsockname()[1]
    held = []

    def accept_and_hold():
        while True:
            try:
                connection, _ = listener.accept()
            except OSError:
                return
            held.append(connection)  # accepted, never answered

    threading.Thread(target=accept_and_hold, daemon=True).start()
    attempts = []

    def dial(budget):
        attempts.append(budget)
        from websockets.sync.client import connect
        return connect(f'ws://127.0.0.1:{port}/api/ws', subprotocols=[GATEWAY_WS_PROTOCOL],
                       open_timeout=budget)

    try:
        started = time.monotonic()
        with pytest.raises(TimeoutError):
            connect_worker_link(dial, open_timeout=2.0, attempts=3, sleep=lambda _s: None)
        elapsed = time.monotonic() - started
    finally:
        listener.close()
        for connection in held:
            connection.close()

    assert len(attempts) == 3
    assert elapsed < 2.0 * 3 + 5, f'waited {elapsed:.1f}s for a 3 x 2 s budget'
    assert elapsed >= 2.0 * 3, 'each attempt must get the full budget'


def test_a_refused_handshake_is_not_retried():
    """T3: an auth refusal will not heal with another attempt (and must not resend)."""
    owner = _Owner(refuse=True)
    attempts = []

    def dial(budget):
        attempts.append(budget)
        return owner.dial(budget)

    try:
        with pytest.raises(Exception) as caught:
            connect_worker_link(dial, open_timeout=2.0, attempts=3, sleep=lambda _s: None)
    finally:
        owner.close()

    assert not isinstance(caught.value, TimeoutError)
    assert len(attempts) == 1, 'a refusal must fail fast, not retry'
    assert owner.refused == 1


def test_every_attempt_mints_its_own_ticket():
    """T4a: a retry must never resend a spent, single-use ticket."""
    tokens = []

    def dial(budget):
        ticket = TicketStore('instance-test', frozenset({PROFILE})).mint(
            profile_id=PROFILE, subject='sid:test', purpose='worker-adoption')
        tokens.append(ticket)
        if len(tokens) < 3:
            raise TimeoutError('timed out while waiting for handshake response')
        return 'linked'

    assert connect_worker_link(dial, open_timeout=25.0, attempts=3,
                               sleep=lambda _s: None) == 'linked'
    assert len(set(tokens)) == 3


def test_the_adopt_budget_stays_inside_the_bootstrap_ticket_ttl():
    """T4b: the knob and the ticket lifetime are one contract, not two."""
    from agent.runtime_session_store import WORKER_RPC_RETRY_SLEEP_SECONDS

    assert WORKER_RPC_OPEN_TIMEOUT_SECONDS < TicketStore.TTL_SECONDS
    # The margin must also absorb the handshake's own round trip on a healthy owner.
    assert TicketStore.TTL_SECONDS - WORKER_RPC_OPEN_TIMEOUT_SECONDS >= 5
    assert WORKER_RPC_RETRY_SLEEP_SECONDS < WORKER_RPC_OPEN_TIMEOUT_SECONDS


def test_a_ticket_past_the_ttl_is_refused_not_timed_out(monkeypatch):
    """T4c: why a flat 60 s budget is not a fix.

    Past the TTL the owner answers ``PermissionError`` -- a refusal, which the retry predicate
    deliberately does not retry. So a 60 s first attempt against a 45 s stall would have killed
    the worker exactly as the 5 s budget did, only later.
    """
    store = TicketStore('instance-test', frozenset({PROFILE}))
    now = time.monotonic()
    monkeypatch.setattr('gateway.runtime_bootstrap.time.monotonic', lambda: now)
    ticket = store.mint(profile_id=PROFILE, subject='sid:test', purpose='worker-adoption')

    monkeypatch.setattr('gateway.runtime_bootstrap.time.monotonic', lambda: now + 31)
    with pytest.raises(PermissionError) as caught:
        store.redeem(ticket, profile_id=PROFILE, purpose='worker-adoption')

    assert not isinstance(caught.value, TimeoutError)
    assert 'expired' in str(caught.value)
