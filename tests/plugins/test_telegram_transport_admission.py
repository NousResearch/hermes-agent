"""Request-write admission tests with synthetic and loopback socket I/O.

The real HTTPX/httpcore request path runs above the fake asyncio stream.  This
keeps the admission/write boundary under test without contacting Telegram;
loopback TCP covers connection reuse and peer-visible cleanup.
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
import sys
from types import SimpleNamespace
import threading

import httpx
import pytest

from plugins.platforms.telegram import transport_admission as wire

# "any": the OS lanes select only platforms-marked files, and socket
# readiness timing differs per host.
pytestmark = pytest.mark.platforms("any")


class AdmissionSource:
    def __init__(self) -> None:
        self.active = True
        self.lock = threading.RLock()
        self.registration = SimpleNamespace(lock=threading.RLock())

    def admitted(self) -> bool:
        return self.active


class SocketWriter:
    def __init__(self, pause: str | None = None, *, respond: bool = True) -> None:
        self.reader = asyncio.StreamReader()
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.pause = pause
        self.respond_enabled = respond
        self.writes: list[bytes] = []
        self.closed = False
        self.connecting = True
        self.buffer = b""
        self.responded = False
        self.socket_options: list[tuple] = []
        self.transport = SimpleNamespace(abort=self.close)

    async def drain(self) -> None:
        if self.pause == "drain" and not self.writes:
            self.entered.set()
            await self.release.wait()

    def write(self, data: bytes) -> None:
        self.writes.append(bytes(data))
        self.buffer += data
        if self.respond_enabled and b"\r\n\r\n" in self.buffer and not self.responded:
            headers, body = self.buffer.split(b"\r\n\r\n", 1)
            length = next(
                (
                    int(line.split(b":", 1)[1])
                    for line in headers.split(b"\r\n")
                    if line.lower().startswith(b"content-length:")
                ),
                0,
            )
            if len(body) >= length:
                self.responded = True
                self.reader.feed_data(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")

    async def start_tls(self, context, *, server_hostname, ssl_handshake_timeout) -> None:
        self.connecting = False
        if self.pause == "tls":
            self.entered.set()
            await self.release.wait()

    def get_extra_info(self, name: str):
        if name == "socket":
            return SimpleNamespace(
                fileno=lambda: -1 if self.closed else 1,
                setsockopt=lambda *args: self.socket_options.append(args),
            )
        return None

    def is_closing(self) -> bool:
        return self.closed

    def close(self) -> None:
        self.closed = True

    async def wait_closed(self) -> None:
        return None


async def _wait_until(
    predicate, *, timeout: float = 2.0, interval: float = 0.01
) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate() and loop.time() < deadline:
        await asyncio.sleep(interval)
    assert predicate(), f"condition not met within {timeout}s"


@contextmanager
def admitted_operation(source: AdmissionSource):
    admission = wire.OperationAdmission(source)
    token = wire.operation_admission.set(admission)
    try:
        yield admission
    finally:
        wire.operation_admission.reset(token)


async def _request_through_writer(
    monkeypatch,
    writer: SocketWriter,
    *,
    url: str = "http://api.telegram.org/bot123:synthetic/getMe",
    content: bytes | None = None,
):
    async def connect(*args, **kwargs):
        if writer.pause == "connect":
            writer.entered.set()
            await writer.release.wait()
        return writer.reader, writer

    monkeypatch.setattr(wire.asyncio, "open_connection", connect)
    async with httpx.AsyncClient(
        transport=wire.AdmissionHTTPTransport(), timeout=2
    ) as client:
        if content is None:
            return await client.get(url)
        return await client.post(url, content=content)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("pause", "url"),
    [
        ("connect", "http://api.telegram.org/bot123:synthetic/getMe"),
        ("tls", "https://api.telegram.org/bot123:synthetic/getMe"),
        ("drain", "http://api.telegram.org/bot123:synthetic/getMe"),
    ],
)
async def test_revocation_before_request_write_is_definitely_unsent(
    monkeypatch, pause, url
):
    source = AdmissionSource()
    writer = SocketWriter(pause)
    with admitted_operation(source) as admission:
        task = asyncio.create_task(
            _request_through_writer(monkeypatch, writer, url=url)
        )
        await asyncio.wait_for(writer.entered.wait(), 2)
        with source.registration.lock, source.lock:
            source.active = False
        writer.release.set()
        with pytest.raises(wire.AdmissionRevoked):
            await task

    assert writer.writes == []
    assert admission.dispatched is False


def test_check_and_write_share_the_revocation_locks():
    held: list[str] = []

    class TrackingLock:
        def __init__(self, name: str) -> None:
            self.name = name

        def __enter__(self):
            held.append(self.name)

        def __exit__(self, *exc):
            assert held.pop() == self.name

    source = AdmissionSource()
    source.registration.lock = TrackingLock("registration")
    source.lock = TrackingLock("operation")
    admission = wire.OperationAdmission(source)
    writer = SimpleNamespace(
        write=lambda data: (
            held == ["registration", "operation"]
            or pytest.fail("request write escaped the admission locks")
        )
    )

    admission.write(writer, b"request")

    assert admission.dispatched is True
    assert held == []


def test_synchronous_write_error_is_conservatively_dispatched():
    source = AdmissionSource()
    admission = wire.OperationAdmission(source)

    def fail_write(data: bytes) -> None:
        raise OSError("synthetic enqueue failure")

    with pytest.raises(OSError, match="enqueue failure"):
        admission.write(SimpleNamespace(write=fail_write), b"request")

    assert admission.dispatched is True


@pytest.mark.asyncio
async def test_revocation_after_headers_preserves_uncertain_outcome(monkeypatch):
    source = AdmissionSource()
    writer = SocketWriter(respond=False)
    original_write = writer.write

    def revoke_after_headers(data: bytes) -> None:
        original_write(data)
        if data.startswith(b"POST "):
            with source.registration.lock, source.lock:
                source.active = False

    writer.write = revoke_after_headers
    with admitted_operation(source) as admission:
        with pytest.raises(wire.AdmissionRevoked):
            await _request_through_writer(monkeypatch, writer, content=b"payload")

    assert len(writer.writes) == 1
    assert admission.dispatched is True


@pytest.mark.asyncio
async def test_request_without_admission_keeps_ordinary_http_behavior(monkeypatch):
    writer = SocketWriter()

    response = await _request_through_writer(monkeypatch, writer)

    assert response.status_code == 200
    assert response.text == "ok"
    assert writer.writes[0].startswith(b"GET ")
    assert wire.operation_admission.get() is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "transport_factory",
    [
        pytest.param(
            lambda: httpx.AsyncHTTPTransport(),
            id="httpx",
            # The Windows proactor loop drains the idle 408 into the
            # StreamReader buffer, so httpcore's socket-readiness probe can
            # never see it and the stock pool reuses the connection there.
            marks=pytest.mark.skipif(
                sys.platform == "win32",
                reason="stock httpcore cannot observe a response already "
                "buffered in user space by the proactor loop",
            ),
        ),
        pytest.param(
            lambda: wire.AdmissionHTTPTransport(trust_env=False),
            id="admission",
        ),
    ],
)
async def test_idle_buffered_response_retires_connection(
    monkeypatch, transport_factory
):
    allow_idle_response = asyncio.Event()
    idle_response_sent = asyncio.Event()
    connection_count = 0
    writers: set[asyncio.StreamWriter] = set()
    admission_readers: list[asyncio.StreamReader] = []

    async def handle_connection(reader, writer):
        nonlocal connection_count
        connection_count += 1
        connection_number = connection_count
        writers.add(writer)
        try:
            await reader.readuntil(b"\r\n\r\n")
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")
            await writer.drain()
            if connection_number == 1:
                await allow_idle_response.wait()
                writer.write(
                    b"HTTP/1.1 408 Request Timeout\r\n"
                    b"Content-Length: 0\r\n\r\n"
                )
                await writer.drain()
                idle_response_sent.set()
                await reader.read()
        finally:
            writers.discard(writer)
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(handle_connection, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    original_open_connection = asyncio.open_connection

    async def capture_admission_reader(*args, **kwargs):
        reader, writer = await original_open_connection(*args, **kwargs)
        admission_readers.append(reader)
        return reader, writer

    monkeypatch.setattr(wire.asyncio, "open_connection", capture_admission_reader)
    transport = transport_factory()
    try:
        async with httpx.AsyncClient(transport=transport, timeout=2) as client:
            first = await client.get(f"http://127.0.0.1:{port}/first")
            allow_idle_response.set()
            await asyncio.wait_for(idle_response_sent.wait(), 2)
            if admission_readers:
                await _wait_until(lambda: admission_readers[0]._buffer)
            else:
                # On selector loops the idle response stays on the socket until
                # read, so the pool's own readiness probe observes it (the
                # admission transport also checks its reader buffer instead).
                [idle] = transport._pool.connections
                await _wait_until(idle.has_expired)
            second = await client.get(f"http://127.0.0.1:{port}/second")

        assert [first.status_code, second.status_code] == [200, 200]
        assert connection_count == 2
    finally:
        for writer in list(writers):
            writer.close()
        await asyncio.gather(
            *(writer.wait_closed() for writer in list(writers)),
            return_exceptions=True,
        )
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_revocation_before_tls_closes_connected_stream_without_writing(
    monkeypatch,
):
    peer_eof = asyncio.Event()
    peer_bytes = bytearray()
    writers: set[asyncio.StreamWriter] = set()
    client_writers: list[asyncio.StreamWriter] = []

    async def handle_connection(reader, writer):
        writers.add(writer)
        try:
            while data := await reader.read(4096):
                peer_bytes.extend(data)
        finally:
            peer_eof.set()
            writers.discard(writer)
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(handle_connection, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    source = AdmissionSource()
    original_open_connection = asyncio.open_connection

    async def capture_client_stream(*args, **kwargs):
        reader, writer = await original_open_connection(*args, **kwargs)
        client_writers.append(writer)
        return reader, writer

    monkeypatch.setattr(wire.asyncio, "open_connection", capture_client_stream)

    async def revoke_at_tls_start(name, info):
        if name == "connection.start_tls.started":
            with source.registration.lock, source.lock:
                source.active = False

    try:
        with admitted_operation(source) as admission:
            request = httpx.Request(
                "GET",
                f"https://127.0.0.1:{port}/prewrite",
                extensions={"trace": revoke_at_tls_start},
            )
            async with httpx.AsyncClient(
                transport=wire.AdmissionHTTPTransport(
                    verify=False, trust_env=False
                ),
                timeout=2,
            ) as client:
                with pytest.raises(wire.AdmissionRevoked):
                    await client.send(request)
            await asyncio.wait_for(peer_eof.wait(), 2)

        assert peer_bytes == b""
        assert admission.dispatched is False
    finally:
        for writer in client_writers:
            writer.close()
        await asyncio.gather(
            *(writer.wait_closed() for writer in client_writers),
            return_exceptions=True,
        )
        for writer in list(writers):
            writer.close()
        await asyncio.gather(
            *(writer.wait_closed() for writer in list(writers)),
            return_exceptions=True,
        )
        server.close()
        await server.wait_closed()
