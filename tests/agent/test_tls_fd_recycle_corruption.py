"""Regressions for issue #29507 — cross-thread close of the per-request OpenAI
client could release a TLS socket FD whose integer was still cached in the
owning httpx worker's SSL BIO. The kernel then recycled the FD into the next
``open()`` (e.g. the kanban dispatcher's ``kanban.db``), and the worker's
delayed TLS flush wrote a 24-byte TLS application-data record on top of the
SQLite header.

The fix has two prongs:

1. ``force_close_tcp_sockets`` no longer calls ``sock.close()`` — only
   ``shutdown(SHUT_RDWR)``. Shutdown unblocks the worker's pending
   ``recv``/``send`` without releasing the FD.

2. ``_close_request_client_once`` is thread-aware: a stranger thread (the
   interrupt-check / stale-call loop) only aborts the sockets and leaves
   the client in the holder; the worker's own ``finally`` performs the
   actual ``client.close()`` from its own thread context.

Both prongs together close the FD-recycling window. The tests below pin
prong 1 and the agent's stranger-thread abort entry point (no network, no
real sockets).
"""
from __future__ import annotations

import logging
import socket as _socket
from types import SimpleNamespace
from unittest.mock import MagicMock



# ---------------------------------------------------------------------------
# Prong 1: force_close_tcp_sockets must NOT release file descriptors.
# ---------------------------------------------------------------------------


class _FakeSocket:
    """Records shutdown/close calls without touching real FDs."""

    def __init__(self):
        self.shutdown_calls = 0
        self.close_calls = 0

    def shutdown(self, _how):
        self.shutdown_calls += 1

    def close(self):
        self.close_calls += 1


def _build_fake_client(sock):
    """Mimic the httpcore-1 layout that ``_iter_pool_sockets`` walks."""
    stream = SimpleNamespace(_sock=sock)
    http11 = SimpleNamespace(_network_stream=stream)
    pool_entry = SimpleNamespace(_connection=http11)
    pool = SimpleNamespace(_connections=[pool_entry])
    transport = SimpleNamespace(_pool=pool)
    http_client = SimpleNamespace(_transport=transport)
    return SimpleNamespace(_client=http_client)


def test_force_close_tcp_sockets_shutdown_only_no_close():
    """The smoking-gun guarantee: shutdown is called, close is NOT.

    If a future refactor reintroduces ``sock.close()`` here, the
    FD-recycling race that corrupted ``kanban.db`` (issue #29507) will
    re-open. Pin the contract explicitly.
    """
    from agent.agent_runtime_helpers import force_close_tcp_sockets

    sock = _FakeSocket()
    client = _build_fake_client(sock)

    n = force_close_tcp_sockets(client)

    assert n == 1
    assert sock.shutdown_calls == 1, "shutdown() must run — it's how we unblock the worker"
    assert sock.close_calls == 0, (
        "close() must NOT run from this helper — releasing the FD here is the "
        "race that wrote TLS bytes into kanban.db (#29507)"
    )


def test_force_close_tcp_sockets_uses_shut_rdwr():
    """Both directions must be shut down so the SSL state machine fully unwinds.

    Half-close (e.g. SHUT_WR only) wouldn't unblock a worker blocked in
    ``recv``, defeating the whole point of the helper.
    """
    from agent.agent_runtime_helpers import force_close_tcp_sockets

    captured = []

    class _ProbingSocket:
        def shutdown(self, how):
            captured.append(how)

        def close(self):  # pragma: no cover — must not run, asserted below
            captured.append("CLOSE_CALLED")

    sock = _ProbingSocket()
    client = _build_fake_client(sock)

    force_close_tcp_sockets(client)

    assert captured == [_socket.SHUT_RDWR]



# ---------------------------------------------------------------------------
# End-to-end: the agent's ``_abort_request_openai_client`` shuts sockets and
# logs deferred_close=stranger_thread without ever calling client.close().
# ---------------------------------------------------------------------------


def test_agent_abort_request_openai_client_does_not_call_client_close(caplog):
    """``_abort_request_openai_client`` must shutdown sockets but NEVER close().

    This is the actual entry point used by the stranger-thread path. If a
    future refactor accidentally wires it back to ``_close_openai_client``
    the FD race is back. Pin both the shutdown side-effect AND the absence
    of any ``client.close()`` call.
    """
    from run_agent import AIAgent

    sock = _FakeSocket()
    client = _build_fake_client(sock)

    # ``client.close()`` would mutate the holder if invoked — give it a
    # MagicMock spy so we can assert no call.
    client.close = MagicMock()

    agent = AIAgent.__new__(AIAgent)
    agent._client_log_context = lambda: "provider=test"

    with caplog.at_level(logging.INFO, logger="run_agent"):
        agent._abort_request_openai_client(client, reason="interrupt_abort")

    # Sockets shut down (one in our fake pool).
    assert sock.shutdown_calls == 1
    assert sock.close_calls == 0
    # And critically: client.close() never ran here.
    client.close.assert_not_called()
