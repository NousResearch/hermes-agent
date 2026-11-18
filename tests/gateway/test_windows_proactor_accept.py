"""A dropped Windows AcceptEx must not close the gateway listening socket.

Regression for the gateway sibling of #120164. ``hermes serve`` runs on a
selector loop. The gateway stays on the proactor (subprocesses need it), and
CPython closes that listener when ``getpeername()`` raises WinError 10014.
Desktop then sees connection refused while the process is still up.
"""

import asyncio
import types

from gateway.windows_proactor_accept import (
    _guarded_start_serving,
    install_proactor_accept_guard,
    transient_proactor_accept_error,
)


class _Future:
    def __init__(self, result=None, error=None):
        self._result = result
        self._error = error
        self.callbacks = []

    def result(self):
        if self._error is not None:
            raise self._error
        return self._result

    def add_done_callback(self, callback):
        self.callbacks.append(callback)


class _Sock:
    def __init__(self):
        self.closed = False

    def fileno(self):
        return -1 if self.closed else 7

    def close(self):
        self.closed = True


class _Loop:
    def __init__(self, accepts):
        self._accepts = list(accepts)
        self._accept_futures = {}
        self._proactor = self
        self.scheduled = []
        self.transports = []
        self.handlers = []
        self.closed = False

    def accept(self, sock):
        del sock
        return self._accepts.pop(0)

    def call_soon(self, callback):
        self.scheduled.append(callback)

    def is_closed(self):
        return self.closed

    def call_exception_handler(self, context):
        self.handlers.append(context)

    def _make_socket_transport(self, conn, protocol, extra=None, server=None):
        self.transports.append((conn, protocol, extra, server))


def _winerror(code):
    exc = OSError(code, "accept failed")
    exc.winerror = code
    return exc


def _start(loop, sock):
    types.MethodType(_guarded_start_serving, loop)(lambda: object(), sock)
    loop.scheduled.pop(0)()


def test_transient_winerrors_are_dropped_peers():
    assert transient_proactor_accept_error(_winerror(10014))
    assert transient_proactor_accept_error(_winerror(10054))
    assert not transient_proactor_accept_error(OSError(98, "address in use"))


def test_winerror_10014_rearms_and_keeps_the_listener():
    sock = _Sock()
    failed = _Future(error=_winerror(10014))
    rearmed = _Future()
    loop = _Loop([failed, rearmed])

    _start(loop, sock)
    assert sock.closed is False
    assert failed.callbacks

    failed.callbacks.pop(0)(failed)

    assert sock.closed is False
    assert loop.handlers == []
    assert rearmed.callbacks
    assert loop._accept_futures[7] is rearmed


def test_other_oserror_still_closes_the_listener():
    sock = _Sock()
    failed = _Future(error=OSError(98, "address in use"))
    loop = _Loop([failed])

    _start(loop, sock)
    failed.callbacks.pop(0)(failed)

    assert sock.closed is True
    assert loop.handlers and loop.handlers[0]["message"] == "Accept failed on a socket"


def test_successful_accept_hands_off_and_arms_the_next():
    sock = _Sock()
    conn = object()
    ok = _Future(result=(conn, ("127.0.0.1", 9)))
    nxt = _Future()
    loop = _Loop([ok, nxt])

    _start(loop, sock)
    ok.callbacks.pop(0)(ok)

    assert loop.transports and loop.transports[0][0] is conn
    assert sock.closed is False
    assert nxt.callbacks


def test_cancelled_accept_closes_the_listener():
    sock = _Sock()
    failed = _Future(error=asyncio.CancelledError())
    loop = _Loop([failed])

    _start(loop, sock)
    failed.callbacks.pop(0)(failed)

    assert sock.closed is True


def test_guard_stays_off_when_disabled_or_not_a_proactor():
    loop = _Loop([])
    assert install_proactor_accept_guard(loop, enabled=False) is False
    assert install_proactor_accept_guard(loop, enabled=True) is False
