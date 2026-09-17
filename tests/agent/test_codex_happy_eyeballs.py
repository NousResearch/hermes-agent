import errno
import selectors
import socket

import httpcore
import pytest

from agent import process_bootstrap

# Captured at import time, before any test can construct an AIAgent (whose
# init_agent now installs the process-level racer) — the restore fixture below
# needs the pristine originals regardless of test ordering.
from urllib3.util import connection as _urllib3_connection_util

_STOCK_SOCKET_CREATE_CONNECTION = socket.create_connection
_STOCK_URLLIB3_CREATE_CONNECTION = _urllib3_connection_util.create_connection


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


@pytest.fixture
def restored_socket_connect():
    yield
    socket.create_connection = _STOCK_SOCKET_CREATE_CONNECTION
    _urllib3_connection_util.create_connection = _STOCK_URLLIB3_CREATE_CONNECTION
    process_bootstrap._SOCKET_CONNECT_RACER_INSTALLED = False


def test_install_happy_eyeballs_socket_connect_patches_both_stacks(restored_socket_connect):
    process_bootstrap.install_happy_eyeballs_socket_connect()

    assert socket.create_connection is not _STOCK_SOCKET_CREATE_CONNECTION
    assert _urllib3_connection_util.create_connection is not _STOCK_URLLIB3_CREATE_CONNECTION

    # Idempotent: a second install must not wrap the racer again.
    first_socket_racer = socket.create_connection
    first_urllib3_racer = _urllib3_connection_util.create_connection
    process_bootstrap.install_happy_eyeballs_socket_connect()
    assert socket.create_connection is first_socket_racer
    assert _urllib3_connection_util.create_connection is first_urllib3_racer


def test_installed_socket_connect_races_past_blackholed_ipv6(
        monkeypatch, restored_socket_connect):
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

    process_bootstrap.install_happy_eyeballs_socket_connect()

    # http.client passes the module timeout sentinel through positionally.
    winner = socket.create_connection(
        ("example.com", 443), socket._GLOBAL_DEFAULT_TIMEOUT, None
    )

    assert winner.family == socket.AF_INET
    assert winner.timeout is None  # sentinel resolves to the process default (None here), like stock
    assert clock[0] == process_bootstrap._HAPPY_EYEBALLS_DELAY_SECONDS
    assert sockets[0].closed is True
    assert sockets[1] is winner


def test_installed_racer_serves_http_client_and_urllib3_connects(restored_socket_connect):
    import http.client

    import urllib3

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(4)
    port = listener.getsockname()[1]

    process_bootstrap.install_happy_eyeballs_socket_connect()
    http_conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    urllib3_conn = urllib3.connection.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        http_conn.connect()
        urllib3_conn.connect()  # exercises the socket_options kwarg of the urllib3 racer
        assert http_conn.sock is not None
        assert urllib3_conn.sock is not None
    finally:
        http_conn.close()
        urllib3_conn.close()
        listener.close()


def test_installed_racer_honours_process_default_timeout_on_sentinel(restored_socket_connect):
    # Stock create_connection leaves the sentinel alone, so the socket keeps the
    # process default set by socket.setdefaulttimeout(); the racer re-applies the
    # timeout on the winner and must resolve the sentinel to that same default
    # instead of forcing a blocking socket.
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(4)
    port = listener.getsockname()[1]

    process_bootstrap.install_happy_eyeballs_socket_connect()
    socket.setdefaulttimeout(5.0)
    try:
        winner = socket.create_connection(("127.0.0.1", port), socket._GLOBAL_DEFAULT_TIMEOUT)
        try:
            assert winner.gettimeout() == 5.0
        finally:
            winner.close()
    finally:
        socket.setdefaulttimeout(None)
        listener.close()


def test_installed_racer_accepts_all_errors_keyword(restored_socket_connect):
    # socket.create_connection gained the keyword-only all_errors parameter in
    # Python 3.11 (the repo floor); forwarding it through the installed racer must
    # not fail with a TypeError before the connect is even attempted.
    process_bootstrap.install_happy_eyeballs_socket_connect()
    with pytest.raises(OSError):
        socket.create_connection(("127.0.0.1", 1), 1.0, None, all_errors=True)
