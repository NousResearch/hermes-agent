import errno
import selectors
import socket

import httpcore
import pytest

from agent import process_bootstrap


@pytest.fixture
def no_proxy_env(monkeypatch):
    for name in (
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "ALL_PROXY",
        "https_proxy",
        "http_proxy",
        "all_proxy",
        "NO_PROXY",
        "no_proxy",
    ):
        monkeypatch.delenv(name, raising=False)


def _client_backends(client):
    transports = [client._transport, *client._mounts.values()]
    return [
        transport._pool._network_backend
        for transport in transports
        if transport is not None and hasattr(transport, "_pool")
    ]


def test_codex_sync_client_uses_happy_eyeballs_backend(no_proxy_env):
    from run_agent import AIAgent

    client = AIAgent._build_keepalive_http_client(
        "https://chatgpt.com/backend-api/codex"
    )
    try:
        assert any(
            isinstance(backend, process_bootstrap._HappyEyeballsSyncBackend)
            for backend in _client_backends(client)
        )
    finally:
        client.close()


def test_other_sync_clients_keep_httpcore_default_backend(no_proxy_env):
    client = process_bootstrap.build_keepalive_http_client(
        "https://api.openai.com/v1"
    )
    try:
        assert all(
            isinstance(backend, httpcore.SyncBackend)
            for backend in _client_backends(client)
        )
    finally:
        client.close()


def test_connection_staggers_past_blackholed_ipv6(monkeypatch):
    clock = [0.0]
    sockets = []

    class FakeSocket:
        def __init__(self, family, socktype, proto):
            self.family = family
            self.closed = False
            self.timeout = None
            sockets.append(self)

        def setsockopt(self, *_args):
            pass

        def setblocking(self, _blocking):
            pass

        def settimeout(self, timeout):
            self.timeout = timeout

        def bind(self, _address):
            pass

        def connect_ex(self, _address):
            if self.family == socket.AF_INET6:
                return errno.EINPROGRESS
            return 0

        def close(self):
            self.closed = True

    class FakeSelector:
        def __init__(self):
            self.registered = set()

        def register(self, fileobj, _events):
            self.registered.add(fileobj)

        def unregister(self, fileobj):
            self.registered.discard(fileobj)

        def select(self, timeout):
            clock[0] += timeout or 0.0
            return []

        def close(self):
            pass

    monkeypatch.setattr(
        process_bootstrap.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (
                socket.AF_INET6,
                socket.SOCK_STREAM,
                socket.IPPROTO_TCP,
                "",
                ("2001:db8::1", 443, 0, 0),
            ),
            (
                socket.AF_INET,
                socket.SOCK_STREAM,
                socket.IPPROTO_TCP,
                "",
                ("192.0.2.1", 443),
            ),
        ],
    )
    monkeypatch.setattr(process_bootstrap.socket, "socket", FakeSocket)
    monkeypatch.setattr(
        process_bootstrap.selectors, "DefaultSelector", FakeSelector
    )
    monkeypatch.setattr(
        process_bootstrap.time, "monotonic", lambda: clock[0]
    )

    winner = process_bootstrap._happy_eyeballs_create_connection(
        ("chatgpt.com", 443),
        timeout=10.0,
    )

    assert winner.family == socket.AF_INET
    assert winner.timeout == 10.0
    assert clock[0] == process_bootstrap._HAPPY_EYEBALLS_DELAY_SECONDS
    assert sockets[0].closed is True
    assert sockets[1].closed is False
