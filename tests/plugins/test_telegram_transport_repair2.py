"""Retry, fallback, and connection-retirement controls for admission transport."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
import select
import socket
from types import SimpleNamespace
import threading

import httpx
import pytest

from plugins.platforms.telegram import telegram_network
from plugins.platforms.telegram import transport_admission as wire

# "any": the OS lanes select only platforms-marked files, and the readiness
# probe falls back to select() on Windows.
pytestmark = pytest.mark.platforms("any")


class AdmissionSource:
    def __init__(self) -> None:
        self.active = True
        self.lock = threading.RLock()
        self.registration = SimpleNamespace(lock=threading.RLock())

    def admitted(self) -> bool:
        return self.active


@contextmanager
def admitted_operation():
    admission = wire.OperationAdmission(AdmissionSource())
    token = wire.operation_admission.set(admission)
    try:
        yield admission
    finally:
        wire.operation_admission.reset(token)


class ScriptedAdmissionTransport(wire.AdmissionHTTPTransport):
    """Admission-aware test double for the outer fallback transport."""

    def __init__(self, action: str, calls: list[str]) -> None:
        self.action = action
        self.calls = calls
        self.closed = False

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request.url.host or "")
        if self.action == "prewrite-error":
            raise httpx.ConnectError("synthetic pre-write failure")
        if self.action == "postwrite-error":
            admission = wire.operation_admission.get()
            assert admission is not None
            admission.dispatched = True
            raise httpx.ConnectError("synthetic post-write failure")
        return httpx.Response(200, request=request, text="ok")

    async def aclose(self) -> None:
        self.closed = True


def scripted_factory(actions: list[str], calls: list[str]):
    instances: list[ScriptedAdmissionTransport] = []

    def factory(**kwargs):
        transport = ScriptedAdmissionTransport(actions.pop(0), calls)
        instances.append(transport)
        return transport

    factory.instances = instances
    return factory


def telegram_request() -> httpx.Request:
    return httpx.Request("POST", "https://api.telegram.org/bot123:synthetic/sendMessage")


@pytest.mark.asyncio
@pytest.mark.parametrize("with_admission", [False, True])
async def test_prewrite_connect_failure_keeps_fallback_retry(with_admission):
    calls: list[str] = []
    # Primary is constructed first but IPv4 fallbacks are attempted first.
    factory = scripted_factory(["ok", "prewrite-error", "ok"], calls)
    transport = telegram_network.TelegramFallbackTransport(
        ["149.154.166.110", "149.154.167.220"], transport_factory=factory
    )

    if with_admission:
        with admitted_operation() as admission:
            response = await transport.handle_async_request(telegram_request())
            assert admission.dispatched is False
    else:
        response = await transport.handle_async_request(telegram_request())

    assert response.status_code == 200
    assert calls == ["149.154.166.110", "149.154.167.220"]
    assert factory.instances[1].closed is True
    await transport.aclose()


@pytest.mark.asyncio
async def test_postwrite_failure_is_not_retried():
    calls: list[str] = []
    factory = scripted_factory(["ok", "postwrite-error", "ok"], calls)
    transport = telegram_network.TelegramFallbackTransport(
        ["149.154.166.110", "149.154.167.220"], transport_factory=factory
    )

    with admitted_operation() as admission:
        with pytest.raises(httpx.ConnectError, match="post-write"):
            await transport.handle_async_request(telegram_request())

    assert admission.dispatched is True
    assert calls == ["149.154.166.110"]
    await transport.aclose()


@pytest.mark.asyncio
async def test_uninstrumented_transport_is_never_reported_as_unsent():
    calls: list[str] = []

    class ForeignTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            calls.append(request.url.host or "")
            raise httpx.ConnectError("foreign transport outcome unknown")

        async def aclose(self):
            return None

    transport = telegram_network.TelegramFallbackTransport(
        ["149.154.166.110", "149.154.167.220"],
        transport_factory=lambda **kwargs: ForeignTransport(),
    )

    with admitted_operation() as admission:
        with pytest.raises(httpx.ConnectError, match="outcome unknown"):
            await transport.handle_async_request(telegram_request())

    assert admission.dispatched is True
    assert calls == ["149.154.166.110"]
    await transport.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "state",
    [
        "eof",
        "error",
        "closing",
        "no-socket",
        "negative-fd",
        "none-fd",
        "bad-fd",
        "ready",
        "idle",
    ],
)
async def test_closed_or_readable_streams_expire_from_the_pool(state):
    # A socket pair, not os.pipe(): Windows select() accepts only sockets.
    read_sock, write_sock = socket.socketpair()
    reader = asyncio.StreamReader()
    fd = read_sock.fileno()
    if state == "eof":
        reader.feed_eof()
    elif state == "error":
        reader.set_exception(ConnectionResetError("fixture"))
    elif state == "negative-fd":
        fd = -1
    elif state == "none-fd":
        fd = None
    elif state == "bad-fd":
        read_sock.close()
    elif state == "ready":
        write_sock.send(b"x")
        select.select([read_sock], [], [], 2)  # bounded wait for delivery
    writer = SimpleNamespace(
        is_closing=lambda: state == "closing",
        get_extra_info=lambda name: (
            None if state == "no-socket" else SimpleNamespace(fileno=lambda: fd)
        ),
    )
    try:
        assert wire.AdmissionStream(reader, writer).get_extra_info(
            "is_readable"
        ) is (state != "idle")
    finally:
        read_sock.close()
        write_sock.close()


@pytest.mark.asyncio
async def test_transport_close_aborts_hung_tls_shutdown():
    reader = asyncio.StreamReader()
    aborted: list[bool] = []
    writer = SimpleNamespace(
        close=lambda: None,
        wait_closed=lambda: asyncio.Event().wait(),
        transport=SimpleNamespace(abort=lambda: aborted.append(True)),
    )

    await asyncio.wait_for(wire.AdmissionStream(reader, writer).aclose(), 2)

    assert aborted == [True]
