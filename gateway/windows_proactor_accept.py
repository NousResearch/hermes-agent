"""Keep a Windows gateway listener open when a client goes away before it is accepted.

CPython's proactor ``_start_serving`` closes the listening socket on any ``OSError`` from an
accept. A client that resets while it waits in the listen queue fails the next accept with
WinError 64 (``ERROR_NETNAME_DELETED``), and ``getpeername()`` in ``finish_accept`` can fail with
WinError 10014. Either one closes the gateway API port while the process keeps running and
``gateway ensure`` keeps reporting it ready, so Desktop's dials are refused until a restart.

``hermes serve`` runs on a selector loop (#120164); the gateway needs the proactor for
subprocesses. The guard wraps the proactor's ``accept`` rather than ``_start_serving``, so a
CPython that already handles the reset (gh-93821) keeps its own serving loop.
"""

from __future__ import annotations

import asyncio
from asyncio.proactor_events import BaseProactorEventLoop
import logging

logger = logging.getLogger(__name__)

# ERROR_NETNAME_DELETED (reset in the listen queue), WSAEFAULT (getpeername after AcceptEx).
_DROPPED_PEER_WINERRORS = frozenset({64, 10014})


def dropped_peer(exc: BaseException) -> bool:
    """True when an accept failed because its client went away, not because the listener did."""
    if isinstance(exc, (ConnectionResetError, ConnectionAbortedError)):
        return True
    return getattr(exc, "winerror", None) in _DROPPED_PEER_WINERRORS


class GuardedAccept:
    """A proactor ``accept(listener)`` whose future completes with a connection, accepting again
    past clients that went away; any other failure, and cancellation, pass through unchanged."""

    def __init__(self, loop: asyncio.AbstractEventLoop, accept):
        self._loop = loop
        self._accept = accept

    def __call__(self, listener):
        outer = self._loop.create_future()
        pending = self._accept(listener)

        def settle(inner):
            nonlocal pending
            if outer.done():
                return
            if inner.cancelled():
                outer.cancel()
                return
            exc = inner.exception()
            if exc is None:
                outer.set_result(inner.result())
                return
            if not dropped_peer(exc):
                outer.set_exception(exc)
                return
            logger.warning("A client went away before the gateway accepted it (%s); still listening", exc)
            try:
                pending = self._accept(listener)
            except OSError as rearm_exc:
                outer.set_exception(rearm_exc)
                return
            pending.add_done_callback(settle)

        def cancel_pending(done):
            if done.cancelled():
                pending.cancel()

        pending.add_done_callback(settle)
        outer.add_done_callback(cancel_pending)
        return outer


def install_proactor_accept_guard(loop: asyncio.AbstractEventLoop) -> None:
    """Wrap the loop's proactor accept once, if CPython exposes that hook."""
    if not isinstance(loop, BaseProactorEventLoop):
        return
    proactor = getattr(loop, "_proactor", None)
    if proactor is None:
        return
    accept = getattr(proactor, "accept", None)
    if not callable(accept) or isinstance(accept, GuardedAccept):
        return
    proactor.accept = GuardedAccept(loop, accept)
