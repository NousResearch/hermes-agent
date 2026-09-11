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


def test_async_codex_client_relies_on_native_anyio_racing(no_proxy_env):
    """The async transport needs no custom backend — anyio races natively.

    httpcore's ``AnyIOBackend.connect_tcp`` delegates to
    ``anyio.connect_tcp``, whose ``happy_eyeballs_delay`` default (0.25s)
    implements RFC 8305 staggered family racing. This pins the contract the
    ``async_mode`` branch of ``build_keepalive_http_client`` documents: if
    anyio ever drops the parameter (or the default stops racing), this fails
    and the async path needs an explicit backend like the sync one.
    """
    import inspect

    import anyio

    params = inspect.signature(anyio.connect_tcp).parameters
    assert "happy_eyeballs_delay" in params
    assert params["happy_eyeballs_delay"].default == pytest.approx(0.25)

    client = process_bootstrap.build_keepalive_http_client(
        "https://chatgpt.com/backend-api/codex", async_mode=True
    )
    try:
        assert all(
            not isinstance(backend, process_bootstrap._HappyEyeballsSyncBackend)
            for backend in _client_backends(client)
        )
    finally:
        import asyncio

        asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
            client.aclose()
        )


def test_async_connect_races_past_blackholed_ipv6(monkeypatch):
    """IPv4 completes ~250ms after a hanging IPv6 attempt on the async path.

    Mirrors ``test_connection_staggers_past_blackholed_ipv6`` for the async
    transport: resolve a fake host to a blackholed IPv6 address plus a live
    local IPv4 listener and assert httpcore's async backend connects fast
    instead of serially waiting out the IPv6 connect timeout.
    """
    import asyncio
    import threading
    import time as _time

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(5)
    port = server.getsockname()[1]

    def _accept_loop():
        while True:
            try:
                conn, _ = server.accept()
                conn.close()
            except OSError:
                return

    thread = threading.Thread(target=_accept_loop, daemon=True)
    thread.start()

    real_getaddrinfo = socket.getaddrinfo

    def fake_getaddrinfo(host, *args, **kwargs):
        name = host.decode() if isinstance(host, (bytes, bytearray)) else str(host)
        if name == "codex-he-async.test":
            return [
                (
                    socket.AF_INET6,
                    socket.SOCK_STREAM,
                    socket.IPPROTO_TCP,
                    "",
                    ("100::1", port, 0, 0),  # RFC 6666 discard prefix: blackhole
                ),
                (
                    socket.AF_INET,
                    socket.SOCK_STREAM,
                    socket.IPPROTO_TCP,
                    "",
                    ("127.0.0.1", port),
                ),
            ]
        return real_getaddrinfo(host, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)

    async def _connect():
        from httpcore._backends.auto import AutoBackend

        backend = AutoBackend()
        start = _time.monotonic()
        stream = await backend.connect_tcp(
            "codex-he-async.test", port, timeout=30.0
        )
        elapsed = _time.monotonic() - start
        await stream.aclose()
        return elapsed

    try:
        elapsed = asyncio.run(_connect())
    finally:
        server.close()

    # Native anyio racing: IPv6 is attempted first, IPv4 starts 0.25s later
    # and wins immediately. Serial behavior would block until the IPv6
    # connect timeout (tens of seconds). Generous bound for slow CI hosts.
    assert elapsed < 5.0


class _RecordingPool:
    def __init__(self):
        self._network_backend = "default"


class _RecordingTransport:
    def __init__(self):
        self._pool = _RecordingPool()


def test_enable_happy_eyeballs_on_client_covers_transport_and_mounts():
    class _Client:
        pass

    client = _Client()
    client._transport = _RecordingTransport()
    client._mounts = {"https://": _RecordingTransport(), "http://": None}

    process_bootstrap.enable_happy_eyeballs_on_client(client)

    assert isinstance(
        client._transport._pool._network_backend,
        process_bootstrap._HappyEyeballsSyncBackend,
    )
    assert isinstance(
        client._mounts["https://"]._pool._network_backend,
        process_bootstrap._HappyEyeballsSyncBackend,
    )


def test_enable_happy_eyeballs_on_client_skips_proxy_pools(no_proxy_env):
    import httpcore
    import httpx

    client = httpx.Client(proxy="http://127.0.0.1:3128")
    try:
        process_bootstrap.enable_happy_eyeballs_on_client(client)
        proxy_pools = [
            transport._pool
            for transport in client._mounts.values()
            if transport is not None
            and isinstance(getattr(transport, "_pool", None), httpcore.HTTPProxy)
        ]
        assert proxy_pools  # the all:// mount is proxy-backed
        assert all(
            not isinstance(
                pool._network_backend, process_bootstrap._HappyEyeballsSyncBackend
            )
            for pool in proxy_pools
        )
    finally:
        client.close()


def test_codex_auth_http_client_uses_happy_eyeballs_backend(no_proxy_env):
    from hermes_cli.auth import _codex_http_client

    client = _codex_http_client(timeout=5.0)
    try:
        assert any(
            isinstance(backend, process_bootstrap._HappyEyeballsSyncBackend)
            for backend in _client_backends(client)
        )
    finally:
        client.close()
