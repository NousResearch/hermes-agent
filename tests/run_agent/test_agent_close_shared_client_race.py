"""Regressions for issue #107475 — ``close()`` hard-closed the shared OpenAI
client from the teardown thread while another thread could still be unwinding a
request on it (the reporter's case: an auxiliary context-compression stream that
produced no further event and no error until its inactivity budget expired).

The codebase already holds the invariant "never hard-close the shared client from
a thread that may not own its FDs" and enforces it at the two sibling sites —
``release_clients()`` (cache_evict) and ``_replace_primary_openai_client``
(replace) both route the shared client through ``_retire_shared_openai_client``
(shutdown(SHUT_RDWR) only; FD release deferred to GC). ``close()`` was the third
shared-client teardown site and the only one still calling ``_close_openai_client``
(FDs released inline). These tests pin the fixed contract:

1. ``close()`` retires the shared client — ``client.close()`` must NEVER run on
   it from the teardown thread.
2. An in-flight (checked-out-of-pool) stream socket is untouched by teardown —
   the stream can still settle on its own.
3. Teardown still drops the reference (idempotence for a second ``close()``) and
   the per-request slot keeps its own in-flight-aware close.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

from run_agent import AIAgent


def _make_close_agent(client):
    """Real AIAgent instance (no __init__/network) wired for close().

    close() runs many guarded phases; only the seams it actually reads need to
    exist. The shared client is the object under observation.
    """
    agent = AIAgent.__new__(AIAgent)
    agent.client = client
    agent._session_messages = None
    return agent


class _RetireSpy:
    """Records retire/close routing for the shared client without touching FDs."""

    def __init__(self, client):
        self.client = client
        self.retired = []
        self.closed = []
        self.force_closed = []
        agent = _make_close_agent(client)
        self.agent = agent

        # Spy on the two seams production routes through; retire does the real
        # (safe) sweep via the patched _force_close_tcp_sockets below.
        def _retire(client, *, reason):
            self.retired.append((client, reason))

        def _close(client, *, reason, shared):
            self.closed.append((client, reason, shared))

        agent._retire_shared_openai_client = _retire
        agent._close_openai_client = _close
        agent._force_close_tcp_sockets = lambda client: (
            self.force_closed.append(client),
            len(self.force_closed),
        )[1]

    def run_close(self):
        self.agent.close()


def test_close_retires_shared_client_instead_of_hard_close():
    """The core fix: close() must route the shared client through retire."""
    client = _StubSharedClient()
    spy = _RetireSpy(client)
    spy.run_close()

    assert spy.retired == [(client, "agent_close")]
    assert spy.closed == [], (
        "close() must not hard-close the shared client (#107475): another thread "
        "may still be unwinding a request on it"
    )
    assert client.close_calls == 0


def test_close_shared_client_survives_in_flight_stream():
    """Object-granularity replay of the reporter's timeline.

    The stream's socket is checked OUT of the idle connection pool while the
    request is in flight, so the pool walk in the shutdown sweep finds nothing
    (tcp_force_closed=0) — the reporter saw exactly that — and the danger is the
    inline ``client.close()`` that follows. After the fix, teardown neither shuts
    down nor closes that socket: the stream keeps its connection.
    """
    stream_sock = _FakeSocket()
    client = _build_shared_client_with_checked_out_stream(stream_sock)

    agent = AIAgent.__new__(AIAgent)
    agent.client = client
    agent._session_messages = None
    agent._retire_shared_openai_client = _real_retire_spy()
    agent._close_openai_client = MagicMock(
        side_effect=AssertionError("must not be called for the shared client")
    )

    agent.close()

    assert stream_sock.shutdown_calls == 0, (
        "a checked-out stream socket must not be shut down by teardown — the aux "
        "stream must be able to complete on its own (#107475)"
    )
    assert stream_sock.close_calls == 0
    assert agent.client is None, "teardown must still drop the shared-client reference"


def test_close_idempotent_second_call_is_noop():
    """close() is documented idempotent: a second call must not re-retire."""
    client = _StubSharedClient()
    spy = _RetireSpy(client)
    spy.run_close()
    spy.run_close()

    assert spy.retired == [(client, "agent_close")]
    assert agent_client_of(spy.agent) is None


def agent_client_of(agent):
    return getattr(agent, "client", "missing")


class _StubSharedClient:
    def __init__(self):
        self.close_calls = 0

    def close(self):
        # Real OpenAI SDK close() would raise once the underlying httpx client
        # is gone; counting the call is enough for the routing assertion.
        self.close_calls += 1


class _FakeSocket:
    def __init__(self):
        self.shutdown_calls = 0
        self.close_calls = 0

    def shutdown(self, _how):
        self.shutdown_calls += 1

    def close(self):
        self.close_calls += 1  # tripwire: teardown must never release the FD (#29507)


def _build_shared_client_with_checked_out_stream(stream_sock):
    """Mimic an OpenAI client whose pool has NO idle connections (the stream
    holds the only one, checked out) — the layout the reporter's
    ``tcp_force_closed=0`` log line proves."""
    pool = SimpleNamespace(_connections=[])
    transport = SimpleNamespace(_pool=pool)
    http_client = SimpleNamespace(_transport=transport)
    return SimpleNamespace(_client=http_client, close=lambda: None)


def _real_retire_spy():
    """A retire that performs the real (safe) sweep: force_close_tcp_sockets
    walks only idle pool sockets, so a checked-out stream socket is untouched."""
    from agent.agent_runtime_helpers import force_close_tcp_sockets

    def _retire(client, *, reason):
        force_close_tcp_sockets(client)

    return _retire
