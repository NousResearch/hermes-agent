"""A client that goes away before the gateway accepts it must not close the Windows listener.

Gateway sibling of #120164. CPython's proactor closes the listening socket on any accept
``OSError``: WinError 64 for a client reset in the listen queue (gh-93821) and WinError 10014
from ``getpeername()``. The port then refuses Desktop while ``gateway ensure`` still reports ready.
"""

import asyncio
from asyncio.proactor_events import BaseProactorEventLoop
import socket
import struct
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from gateway.windows_proactor_accept import GuardedAccept, install_proactor_accept_guard


def _winerror(cls, code):
    exc = cls(22, "accept failed")
    exc.winerror = code
    return exc


class _ScriptedAccept:
    """A proactor ``accept`` whose futures settle, one per call, with the scripted outcomes."""

    def __init__(self, loop, outcomes):
        self._loop = loop
        self._outcomes = list(outcomes)
        self.issued = []

    def __call__(self, listener):
        future = self._loop.create_future()
        self.issued.append(future)
        outcome = self._outcomes.pop(0)
        if outcome is not None:
            settle = future.set_exception if isinstance(outcome, BaseException) else future.set_result
            self._loop.call_soon(settle, outcome)
        return future


def _accept_through_guard(outcomes):
    async def scenario():
        loop = asyncio.get_running_loop()
        accept = _ScriptedAccept(loop, outcomes)
        try:
            return await GuardedAccept(loop, accept)(object()), accept.issued
        except OSError as exc:
            return exc, accept.issued

    return asyncio.run(scenario())


def test_clients_that_went_away_are_skipped_and_the_next_connection_arrives():
    connection = (object(), ("127.0.0.1", 50000))
    outcomes = [
        _winerror(OSError, 64),
        # gh-93821's CPython re-raises the queued reset as ConnectionResetError, still winerror 64.
        _winerror(ConnectionResetError, 64),
        _winerror(OSError, 10014),
        ConnectionAbortedError(53, "aborted"),
        connection,
    ]

    result, issued = _accept_through_guard(outcomes)

    assert result is connection
    assert len(issued) == 5


def test_an_accept_error_that_is_not_a_lost_client_reaches_the_server():
    failure = _winerror(OSError, 10055)

    result, issued = _accept_through_guard([failure])

    assert result is failure
    assert len(issued) == 1


def test_cancelling_the_accept_cancels_the_accept_in_flight():
    async def scenario():
        loop = asyncio.get_running_loop()
        accept = _ScriptedAccept(loop, [_winerror(OSError, 64), None])
        outer = GuardedAccept(loop, accept)(object())
        for _ in range(100):
            if len(accept.issued) == 2:
                break
            await asyncio.sleep(0)
        outer.cancel()
        await asyncio.sleep(0)
        return accept.issued

    issued = asyncio.run(scenario())

    assert len(issued) == 2
    assert issued[1].cancelled()


def test_inner_cancellation_cancels_the_waiting_accept():
    async def scenario():
        loop = asyncio.get_running_loop()
        inner = loop.create_future()
        waiting = GuardedAccept(loop, lambda _: inner)(object())
        inner.cancel()
        await asyncio.sleep(0)
        assert waiting.cancelled()

    asyncio.run(scenario())


def test_queued_completion_does_not_settle_a_cancelled_accept():
    async def scenario():
        loop = asyncio.get_running_loop()
        errors = []
        loop.set_exception_handler(lambda _, context: errors.append(context))
        inner = loop.create_future()
        waiting = GuardedAccept(loop, lambda _: inner)(object())
        inner.set_result((object(), ("127.0.0.1", 50000)))
        waiting.cancel()
        await asyncio.sleep(0)
        assert waiting.cancelled()
        assert not errors

    asyncio.run(scenario())


def test_synchronous_rearm_failure_reaches_the_waiting_accept():
    async def scenario():
        loop = asyncio.get_running_loop()
        inner = loop.create_future()
        failure = _winerror(OSError, 995)
        accept = Mock(side_effect=[inner, failure])
        waiting = GuardedAccept(loop, accept)(object())
        inner.set_exception(_winerror(OSError, 64))
        await asyncio.sleep(0)
        assert waiting.done()
        assert waiting.exception() is failure

    asyncio.run(scenario())


def test_install_wraps_only_the_selected_proactor_and_only_once():
    accept = Mock()
    first, second = (
        Mock(spec=BaseProactorEventLoop, _proactor=SimpleNamespace(accept=accept)) for _ in range(2)
    )
    install_proactor_accept_guard(first)
    installed = first._proactor.accept
    assert isinstance(installed, GuardedAccept)
    install_proactor_accept_guard(first)
    assert first._proactor.accept is installed
    assert second._proactor.accept is accept


def test_selector_install_leaves_a_callable_accept_alone(monkeypatch):
    with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop) as runner:
        loop = runner.get_loop()
        proactor = SimpleNamespace(accept=Mock())
        accept = proactor.accept
        monkeypatch.setattr(loop, "_proactor", proactor, raising=False)
        install_proactor_accept_guard(loop)
        assert proactor.accept is accept


@pytest.mark.platforms("windows")
def test_listener_keeps_serving_after_a_client_resets_in_the_listen_queue():
    async def scenario():
        install_proactor_accept_guard(asyncio.get_running_loop())

        async def handle(_reader, writer):
            writer.write(b"ok")
            await writer.drain()
            writer.close()

        server = await asyncio.start_server(handle, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        # Both connects land while this coroutine holds the loop: the first meets the pending
        # AcceptEx, the second waits in the listen queue and is reset there (linger 0).
        first = socket.create_connection(("127.0.0.1", port))
        second = socket.create_connection(("127.0.0.1", port))
        second.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("HH", 1, 0))
        second.close()
        try:
            reader, writer = await asyncio.wait_for(asyncio.open_connection("127.0.0.1", port), 10)
            reply = await asyncio.wait_for(reader.readexactly(2), 10)
            writer.close()
            return reply
        finally:
            first.close()
            server.close()

    with asyncio.Runner(loop_factory=asyncio.ProactorEventLoop) as runner:
        assert runner.run(scenario()) == b"ok"
