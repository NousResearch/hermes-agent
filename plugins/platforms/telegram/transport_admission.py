"""Exact-operation admission at the asyncio request-write boundary.

HTTPX's request hook and httpcore's header trace precede pool and stream
awaits.  Use httpcore's public network-backend interface instead: drain first,
then check admission and synchronously enqueue bytes with ``StreamWriter.write``
under the fence.  No suspension is allowed between those last two operations.
Bytes enqueued before revocation are possibly dispatched, even if a later
drain or response fails.
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import select
from typing import Any

import httpcore
import httpx


class AdmissionRevoked(Exception):
    """No further writes are authorized (not a claim of remote non-delivery)."""


@dataclass
class OperationAdmission:
    """Admission source and conservative dispatch state for one operation."""

    source: Any
    dispatched: bool = False
    transport_entered: bool = False

    def check(self) -> None:
        if not self.source.admitted():
            raise AdmissionRevoked("operation admission revoked")

    def write(self, writer, data: bytes) -> None:
        source = self.source
        with source.registration.lock, source.lock:
            self.check()
            # Set before enqueue: even a synchronous write error is uncertain.
            self.dispatched = True
            writer.write(data)


operation_admission: ContextVar[OperationAdmission | None] = ContextVar(
    "telegram_operation_admission", default=None
)


def check_admission() -> OperationAdmission | None:
    admission = operation_admission.get()
    if admission is not None:
        admission.transport_entered = True
        admission.check()
    return admission


@contextmanager
def map_core_errors():
    try:
        yield
    except httpcore.NetworkError as exc:
        raise getattr(httpx, type(exc).__name__)(str(exc)) from exc
    except (
        httpcore.TimeoutException,
        httpcore.ProtocolError,
        httpcore.ProxyError,
        httpcore.UnsupportedProtocol,
    ) as exc:
        raise getattr(httpx, type(exc).__name__)(str(exc)) from exc


class AdmissionStream(httpcore.AsyncNetworkStream):
    def __init__(self, reader, writer):
        self.reader = reader
        self.writer = writer

    async def read(self, max_bytes, timeout=None):
        try:
            async with asyncio.timeout(timeout):
                return await self.reader.read(max_bytes)
        except TimeoutError as exc:
            raise httpcore.ReadTimeout() from exc
        except OSError as exc:
            raise httpcore.ReadError() from exc

    async def write(self, buffer, timeout=None):
        if not buffer:
            return
        try:
            async with asyncio.timeout(timeout):
                # The first drain covers flow control and connection-loss checks.
                await self.writer.drain()
                admission = operation_admission.get()
                if admission is None:
                    self.writer.write(buffer)
                else:
                    admission.write(self.writer, buffer)
                await self.writer.drain()
        except TimeoutError as exc:
            raise httpcore.WriteTimeout() from exc
        except OSError as exc:
            raise httpcore.WriteError() from exc

    async def aclose(self):
        self.writer.close()
        # A failed pool must not block indefinitely on TLS shutdown.
        try:
            async with asyncio.timeout(1):
                await self.writer.wait_closed()
        except (OSError, TimeoutError):
            self.writer.transport.abort()
        except asyncio.CancelledError:
            self.writer.transport.abort()
            raise

    async def start_tls(self, ssl_context, server_hostname=None, timeout=None):
        try:
            check_admission()
            async with asyncio.timeout(timeout):
                await self.writer.start_tls(
                    ssl_context,
                    server_hostname=server_hostname,
                    ssl_handshake_timeout=timeout,
                )
            check_admission()
        except TimeoutError as exc:
            await self.aclose()
            raise httpcore.ConnectTimeout() from exc
        except OSError as exc:
            await self.aclose()
            raise httpcore.ConnectError() from exc
        except BaseException:
            await self.aclose()
            raise
        return self

    def get_extra_info(self, info):
        if info == "is_readable":
            # Error-driven connection_lost sets an exception, not EOF.  Closed
            # idle streams and eagerly buffered peer data must expire instead
            # of poisoning the pool scan or becoming the next response.
            if (
                self.reader._buffer
                or self.reader.at_eof()
                or self.reader.exception() is not None
                or self.writer.is_closing()
            ):
                return True
            sock = self.writer.get_extra_info("socket")
            try:
                fd = None if sock is None else sock.fileno()
                if fd is None or fd < 0:
                    return True
                if getattr(select, "poll", None) is not None:
                    poll = select.poll()
                    poll.register(fd, select.POLLIN)
                    return bool(poll.poll(0))
                return bool(select.select([fd], [], [], 0)[0])
            except (OSError, ValueError):
                # The fd can close between inspection and readiness polling.
                return True
        name = {"client_addr": "sockname", "server_addr": "peername"}.get(
            info, info
        )
        return self.writer.get_extra_info(name)


class AdmissionBackend(httpcore.AsyncNetworkBackend):
    async def connect_tcp(
        self,
        host,
        port,
        timeout=None,
        local_address=None,
        socket_options=None,
    ):
        check_admission()
        try:
            async with asyncio.timeout(timeout):
                reader, writer = await asyncio.open_connection(
                    host,
                    port,
                    local_addr=(local_address, 0) if local_address else None,
                    happy_eyeballs_delay=0.25,
                    interleave=1,
                )
        except TimeoutError as exc:
            raise httpcore.ConnectTimeout() from exc
        except OSError as exc:
            raise httpcore.ConnectError() from exc
        stream = AdmissionStream(reader, writer)
        try:
            check_admission()
            sock = writer.get_extra_info("socket")
            for option in socket_options or ():
                sock.setsockopt(*option)
        except BaseException:
            await stream.aclose()
            raise
        return stream

    async def sleep(self, seconds):
        await asyncio.sleep(seconds)
        check_admission()


class ResponseStream(httpx.AsyncByteStream):
    def __init__(self, stream):
        self.stream = stream

    async def __aiter__(self):
        with map_core_errors():
            async for part in self.stream:
                yield part

    async def aclose(self):
        with map_core_errors():
            await self.stream.aclose()


class AdmissionHTTPTransport(httpx.AsyncBaseTransport):
    """HTTP/1 transport whose network backend owns the admission fence.

    Telegram's SDK defaults to HTTP/1.1.  Retaining that invariant avoids
    sharing a multiplexed writer across operation contexts.  Ordinary requests
    have no admission context and keep the same pool retries and fallback
    recovery behavior.
    """

    def __init__(
        self,
        *,
        limits=None,
        socket_options=None,
        proxy=None,
        retries=0,
        verify=True,
        cert=None,
        trust_env=True,
    ):
        limits = limits or httpx.Limits()
        self.limits = limits
        proxy = httpx.Proxy(proxy) if isinstance(proxy, (str, httpx.URL)) else proxy
        self.proxy = proxy
        core_proxy = None
        if proxy is not None:
            core_proxy = httpcore.Proxy(
                str(proxy.url),
                auth=proxy.raw_auth,
                headers=proxy.headers.raw,
                ssl_context=proxy.ssl_context,
            )
        self.pool = httpcore.AsyncConnectionPool(
            ssl_context=httpx.create_ssl_context(
                verify=verify, cert=cert, trust_env=trust_env
            ),
            proxy=core_proxy,
            max_connections=limits.max_connections,
            max_keepalive_connections=limits.max_keepalive_connections,
            keepalive_expiry=limits.keepalive_expiry,
            http1=True,
            http2=False,
            retries=retries,
            socket_options=socket_options,
            network_backend=AdmissionBackend(),
        )

    async def handle_async_request(self, request):
        check_admission()
        core_request = httpcore.Request(
            method=request.method,
            url=httpcore.URL(
                scheme=request.url.raw_scheme,
                host=request.url.raw_host,
                port=request.url.port,
                target=request.url.raw_path,
            ),
            headers=request.headers.raw,
            content=request.stream,
            extensions=request.extensions,
        )
        with map_core_errors():
            response = await self.pool.handle_async_request(core_request)
        return httpx.Response(
            response.status,
            headers=response.headers,
            stream=ResponseStream(response.stream),
            extensions=response.extensions,
        )

    async def aclose(self):
        await self.pool.aclose()
