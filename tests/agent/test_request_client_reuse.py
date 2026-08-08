"""Per-request OpenAI wire client reuse across sequential LLM calls.

Building a fresh ``openai.OpenAI`` client per LLM call costs ~19-35ms (new
httpx pool, TCP+TLS handshake), so ``_create_request_openai_client`` caches
ONE reusable wire client on the agent, keyed by the effective client kwargs:

- identical kwargs → same client object handed back (the reuse win);
- kwargs change (credential rotation, provider failover) → evict + rebuild;
- cross-thread abort (#29507) poisons the slot → the owner-thread close does
  a real close and the next create rebuilds;
- non-reuse close reasons (error cleanups, stale/interrupt kills, retry
  cleanups) discard — only a request that produced a response reports a
  reuse reason (request_complete / stream_request_complete);
- vision-header copilot variant is a distinct kwargs key;
- teardown (release_clients / close) really closes the cached client, or
  detaches it to the in-flight worker's own close when checked out (#29507);
- MoA facade and Mock passthroughs never enter the cache.
"""

import socket
import ssl
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.agent_runtime_helpers import _idle_pool_socket_is_unusable
from run_agent import AIAgent


class _StubClient:
    """Minimal non-Mock client: _is_openai_client_closed reads ``is_closed``."""

    def __init__(self):
        self.is_closed = False

    def close(self):
        self.is_closed = True


class _PoolSocket:
    """Minimal socket probe target for the cached request-client pool."""

    def __init__(self, *, dead=False):
        self.dead = dead
        self.blocking = True
        self.recv_calls = 0

    def setblocking(self, value):
        self.blocking = value

    def recv(self, *_args):
        self.recv_calls += 1
        if self.dead:
            return b""
        raise BlockingIOError


def _client_for_pool_socket(pool_socket):
    client = _StubClient()
    stream = SimpleNamespace(_sock=pool_socket)
    connection = SimpleNamespace(_network_stream=stream)
    pool_entry = SimpleNamespace(_connection=connection)
    pool = SimpleNamespace(_connections=[pool_entry])
    setattr(
        client,
        "_client",
        SimpleNamespace(
            _transport=SimpleNamespace(_pool=pool),
            is_closed=False,
        ),
    )
    setattr(client, "_pool_socket", pool_socket)
    return client


def _client_with_pool_socket(*, dead=False):
    return _client_for_pool_socket(_PoolSocket(dead=dead))


def _set_cached_request_client(agent, client, *, in_use=False):
    request_kwargs = dict(agent._client_kwargs)
    request_kwargs["max_retries"] = 0
    agent._request_client_cache = {
        "client": client,
        "kwargs": request_kwargs,
        "poisoned": False,
        "in_use": in_use,
    }


def _make_agent(provider="openai", base_url="https://api.openai.com/v1", model="gpt-5.4"):
    with patch("run_agent.OpenAI") as mock_openai:
        mock_openai.return_value = MagicMock()
        agent = AIAgent(
            api_key="sk-test",
            base_url=base_url,
            provider=provider,
            model=model,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    # Non-Mock shared client so the Mock passthrough branch doesn't trigger.
    agent.client = _StubClient()
    return agent


class _Harness:
    """Patch the wire-client build/close/socket seams and record calls."""

    def __init__(self, agent):
        self.agent = agent
        self.built = []   # (kwargs_copy, reason)
        self.closed = []  # (client, reason)
        self._patchers = []

    def __enter__(self):
        def _fake_create(kwargs, *, reason, shared):
            # Only record per-request wire clients; teardown tests can also
            # trigger a shared-client rebuild via _ensure_primary_openai_client.
            if not shared:
                self.built.append((dict(kwargs), reason))
            return _StubClient()

        def _fake_close(client, *, reason, shared):
            self.closed.append((client, reason))

        self._patchers = [
            patch.object(self.agent, "_create_openai_client", side_effect=_fake_create),
            patch.object(self.agent, "_close_openai_client", side_effect=_fake_close),
            patch.object(self.agent, "_force_close_tcp_sockets", return_value=0),
        ]
        for p in self._patchers:
            p.start()
        return self

    def __exit__(self, *exc):
        for p in self._patchers:
            p.stop()

    def closed_clients(self):
        return [client for client, _reason in self.closed]


def test_reuse_on_identical_kwargs_same_object():
    agent = _make_agent()
    with _Harness(agent) as h:
        a = agent._create_request_openai_client(reason="chat_completion_request")
        # The agent's outer loop owns retries — the SDK loop must stay off.
        assert h.built[-1][0]["max_retries"] == 0
        agent._close_request_openai_client(a, reason="request_complete")
        assert h.closed == []  # kept for reuse, not really closed

        b = agent._create_request_openai_client(reason="chat_completion_request")
        assert b is a
        assert len(h.built) == 1




def test_rebuild_on_client_kwargs_change():
    agent = _make_agent()
    with _Harness(agent) as h:
        a = agent._create_request_openai_client(reason="r")
        agent._close_request_openai_client(a, reason="request_complete")

        # Credential rotation / provider failover mutate _client_kwargs.
        agent._client_kwargs["api_key"] = "sk-rotated"
        b = agent._create_request_openai_client(reason="r")
        assert b is not a
        # The stale cached client was really closed on eviction.
        assert a in h.closed_clients()
        assert h.built[-1][0]["api_key"] == "sk-rotated"

        # And the rotated client is itself reusable.
        agent._close_request_openai_client(b, reason="request_complete")
        c = agent._create_request_openai_client(reason="r")
        assert c is b


def test_pre_turn_cleanup_rebuilds_dead_primary_client():
    agent = _make_agent()
    agent.client = _client_with_pool_socket(dead=True)

    with patch.object(
        agent, "_replace_primary_openai_client", return_value=True
    ) as replace_primary:
        cleaned = agent._cleanup_dead_connections()

    assert cleaned is True
    replace_primary.assert_called_once_with(reason="dead_connection_cleanup")


def test_request_creation_evicts_dead_idle_cached_client_before_reuse():
    agent = _make_agent()
    cached = _client_with_pool_socket(dead=True)
    _set_cached_request_client(agent, cached)

    with _Harness(agent) as h:
        replacement = agent._create_request_openai_client(reason="next_request")

    assert replacement is not cached
    assert (cached, "reuse_evict:next_request") in h.closed
    assert len(h.built) == 1


def test_concurrent_request_creation_does_not_probe_in_flight_cached_client():
    agent = _make_agent()
    cached = _client_with_pool_socket(dead=True)
    _set_cached_request_client(agent, cached, in_use=True)

    with _Harness(agent) as h:
        concurrent = agent._create_request_openai_client(reason="concurrent_request")

    assert concurrent is not cached
    assert h.closed == []
    assert len(h.built) == 1
    assert agent._request_client_cache["client"] is cached
    assert agent._request_client_cache["in_use"] is True
    assert getattr(cached, "_pool_socket").recv_calls == 0


def test_openai_client_lock_is_stable_and_reentrant():
    agent = AIAgent.__new__(AIAgent)

    lock = agent._openai_client_lock()

    assert agent._openai_client_lock() is lock
    with lock:
        assert lock.acquire(blocking=False) is True
        lock.release()


def test_dead_socket_probe_handles_tls_wrapped_connections():
    raw_socket, peer_socket = socket.socketpair()
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    tls_socket = context.wrap_socket(
        raw_socket,
        server_hostname="localhost",
        do_handshake_on_connect=False,
    )
    client = _client_for_pool_socket(tls_socket)

    try:
        assert AIAgent._dead_request_client_socket_count(client) == 0
        peer_socket.close()
        assert AIAgent._dead_request_client_socket_count(client) == 1
    finally:
        peer_socket.close()
        tls_socket.close()


def test_idle_plain_socket_probe_distinguishes_buffered_data_from_eof():
    local_socket, peer_socket = socket.socketpair()

    try:
        peer_socket.sendall(b"x")
        assert _idle_pool_socket_is_unusable(local_socket) is False
        assert local_socket.recv(1, socket.MSG_PEEK) == b"x"

        local_socket.recv(1)
        peer_socket.close()
        assert _idle_pool_socket_is_unusable(local_socket) is True
    finally:
        peer_socket.close()
        local_socket.close()


def test_socket_probe_fallback_skips_incomplete_double_and_continues_pool_scan():
    incomplete_socket = SimpleNamespace()
    dead_socket = _PoolSocket(dead=True)

    with patch(
        "agent.agent_runtime_helpers._iter_pool_sockets",
        return_value=iter([incomplete_socket, dead_socket]),
    ):
        assert AIAgent._dead_request_client_socket_count(_StubClient()) == 1

    assert dead_socket.recv_calls == 1


def test_idle_socket_probe_uses_poll_for_high_posix_fds():
    sock = MagicMock()
    sock.fileno.return_value = 2048
    sock.getsockopt.return_value = 0
    poller = MagicMock()
    poller.poll.return_value = []

    with (
        patch("sys.platform", "linux"),
        patch("select.poll", return_value=poller, create=True),
        patch("select.POLLIN", 1, create=True),
        patch("select.select", side_effect=AssertionError("select must not run")),
    ):
        assert _idle_pool_socket_is_unusable(sock) is False

    poller.register.assert_called_once_with(2048, 1)


def test_idle_socket_probe_uses_select_on_windows():
    sock = MagicMock()
    sock.fileno.return_value = 2048
    sock.getsockopt.return_value = 0

    with (
        patch("sys.platform", "win32"),
        patch("select.poll", side_effect=AssertionError("poll must not run"), create=True),
        patch("select.select", return_value=([], [], [])) as select_mock,
    ):
        assert _idle_pool_socket_is_unusable(sock) is False

    select_mock.assert_called_once_with([sock], [], [sock], 0)


def test_pre_turn_cleanup_evicts_dead_idle_cached_request_client():
    agent = _make_agent()
    cached = _client_with_pool_socket(dead=True)
    _set_cached_request_client(agent, cached)

    with (
        _Harness(agent) as h,
        patch.object(agent, "_replace_primary_openai_client") as replace_primary,
    ):
        cleaned = agent._cleanup_dead_connections()
        replacement = agent._create_request_openai_client(reason="next_request")

    assert cleaned is True
    assert (cached, "dead_connection_cleanup") in h.closed
    assert replacement is not cached
    assert len(h.built) == 1
    replace_primary.assert_not_called()


def test_pre_turn_cleanup_keeps_healthy_idle_cached_request_client():
    agent = _make_agent()
    cached = _client_with_pool_socket(dead=False)
    _set_cached_request_client(agent, cached)

    with _Harness(agent) as h:
        cleaned = agent._cleanup_dead_connections()
        reused = agent._create_request_openai_client(reason="next_request")

    assert cleaned is False
    assert h.closed == []
    assert reused is cached
    assert h.built == []


def test_pre_turn_cleanup_does_not_close_in_flight_request_client():
    agent = _make_agent()
    cached = _client_with_pool_socket(dead=True)
    _set_cached_request_client(agent, cached, in_use=True)

    with _Harness(agent) as h:
        cleaned = agent._cleanup_dead_connections()

    assert cleaned is False
    assert h.closed == []
    assert agent._request_client_cache["client"] is cached
    assert getattr(cached, "_pool_socket").recv_calls == 0


def test_agent_close_closes_cached_request_client():
    agent = _make_agent()
    with _Harness(agent) as h:
        a = agent._create_request_openai_client(reason="r")
        agent._close_request_openai_client(a, reason="request_complete")

        agent.close()
        assert (a, "agent_close") in h.closed

        # Idempotent: a second teardown must not double-close.
        before = len(h.closed)
        agent._close_cached_request_openai_client(reason="agent_close")
        assert len(h.closed) == before






def test_moa_passthrough_unaffected():
    agent = _make_agent()
    agent.provider = "moa"
    facade = agent.client
    with _Harness(agent) as h:
        a = agent._create_request_openai_client(reason="r")
        assert a is facade
        assert h.built == []  # facade handed back, no wire client built

        # Close behaves exactly as before the cache existed.
        agent._close_request_openai_client(facade, reason="request_complete")
        assert facade in h.closed_clients()


