"""Import-safe synchronous networking helpers for Hermes.

This module intentionally uses only the Python standard library.  It is shared by
``hermes_bootstrap`` (the launcher, whose import has process side effects) and
library consumers such as ``agent.process_bootstrap`` that may run in a foreign
interpreter.  Keep imports here free of Hermes packages and launch-time work.
"""

from __future__ import annotations

import errno
import importlib.abc
import importlib.util
import os
import selectors
import socket
import sys
import time

_HAPPY_EYEBALLS_DELAY_SECONDS = 0.25
_URLLIB3_CONNECTION_MODULE = "urllib3.util.connection"


def _interleave_addrinfos(addrinfos: list[tuple]) -> list[tuple]:
    """Round-robin the resolved address families (deduped), preserving resolver order within each."""
    queues: dict[int, list[tuple]] = {}
    seen: set[tuple] = set()
    for addrinfo in addrinfos:
        family, socktype, proto, _canonname, sockaddr = addrinfo
        if (family, socktype, proto, sockaddr) not in seen:
            seen.add((family, socktype, proto, sockaddr))
            queues.setdefault(family, []).append(addrinfo)
    interleaved: list[tuple] = []
    while any(queues.values()):
        interleaved.extend(queue.pop(0) for queue in queues.values() if queue)
    return interleaved


def _quiet_unregister(selector, sock) -> None:
    try:
        selector.unregister(sock)
    except Exception:
        pass


def _happy_eyeballs_create_connection(address: tuple[str, int], timeout: float | None,
                                      source_address: tuple[str, int] | None = None, socket_options=()):
    """RFC 8305-style connect: staggered non-blocking attempts across families.

    ``socket.create_connection`` tries addresses serially, so broken-but-
    advertised IPv6 can burn the whole timeout per AAAA record before IPv4.
    """
    host, port = address
    addrinfos = _interleave_addrinfos(socket.getaddrinfo(host, port, type=socket.SOCK_STREAM))
    if not addrinfos:
        raise OSError(f"getaddrinfo returned no addresses for {host}")

    selector = selectors.DefaultSelector()
    active: set[socket.socket] = set()
    winner = None
    last_error: OSError | None = None
    deadline = None if timeout is None else time.monotonic() + max(timeout, 0.0)
    next_launch = time.monotonic()
    pending = list(addrinfos)
    in_progress = {0, errno.EINPROGRESS, errno.EWOULDBLOCK, errno.EALREADY, errno.EINTR, getattr(errno, "WSAEWOULDBLOCK", 10035)}

    def start_attempt(addrinfo):
        family, socktype, proto, _canonname, sockaddr = addrinfo
        candidate = socket.socket(family, socktype, proto)
        try:
            if source_address is not None:
                local_infos = socket.getaddrinfo(source_address[0], source_address[1], family=family, type=socktype)
                if not local_infos:
                    raise OSError(f"getaddrinfo returned no local {family} address for {source_address[0]}")
                candidate.bind(local_infos[0][4])
            candidate.setblocking(False)
            result = candidate.connect_ex(sockaddr)
            if result in (0, errno.EISCONN):
                return candidate
            if result not in in_progress:
                raise OSError(result, os.strerror(result))
            selector.register(candidate, selectors.EVENT_WRITE)
            active.add(candidate)
            return None
        except Exception:
            candidate.close()
            raise

    try:
        while pending or active:
            now = time.monotonic()
            if deadline is not None and now >= deadline:
                raise socket.timeout("timed out")
            if pending and now >= next_launch:
                try:
                    winner = start_attempt(pending.pop(0))
                except OSError as exc:
                    last_error = exc
                    if not active:
                        next_launch = now
                    continue
                if winner is not None:
                    break
                next_launch = now + _HAPPY_EYEBALLS_DELAY_SECONDS
            wait_timeout = None if deadline is None else max(0.0, deadline - now)
            if pending:
                until_launch = max(0.0, next_launch - now)
                wait_timeout = until_launch if wait_timeout is None else min(wait_timeout, until_launch)
            for key, _mask in selector.select(wait_timeout):
                candidate = key.fileobj
                error_code = candidate.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR)
                selector.unregister(candidate)
                active.discard(candidate)
                if error_code == 0:
                    winner = candidate
                    break
                candidate.close()
                last_error = OSError(error_code, os.strerror(error_code))
            if winner is not None:
                break
            if not active and pending:
                next_launch = time.monotonic()

        if winner is None:
            raise last_error if last_error is not None else OSError(f"Could not connect to {host}:{port}")
        _quiet_unregister(selector, winner)
        active.discard(winner)
        winner.settimeout(timeout)
        for option in socket_options or ():
            winner.setsockopt(*option)
        winner.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        return winner
    finally:
        for candidate in active:
            _quiet_unregister(selector, candidate)
            candidate.close()
        selector.close()


def _patch_urllib3_create_connection(module) -> None:
    """Point ``urllib3.util.connection.create_connection`` (its own serial walker) at the racer."""
    if getattr(module.create_connection, "_hermes_happy_eyeballs", False):
        return
    urllib3_sentinel = module._DEFAULT_TIMEOUT

    def _urllib3_racer(address, timeout=urllib3_sentinel, source_address=None, socket_options=None):
        effective = socket.getdefaulttimeout() if timeout is urllib3_sentinel else timeout
        # OSError = every candidate failed (identical to the serial original); anything else is a
        # racer bug and must surface rather than silently fall back to the serial stall.
        return _happy_eyeballs_create_connection(
            address, effective, source_address=source_address, socket_options=tuple(socket_options or ()))

    _urllib3_racer._hermes_happy_eyeballs = True  # type: ignore[attr-defined]
    module.create_connection = _urllib3_racer


class _Urllib3ConnectionPatcher(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """One-shot import hook: patch urllib3's connect walker the moment the module loads.

    Importing urllib3 eagerly costs ~50 ms on every CLI start, and ``hermes`` / the TUI
    gateway never load it unless something actually calls ``requests``.
    """

    def find_spec(self, fullname, path, target=None):
        if fullname != _URLLIB3_CONNECTION_MODULE:
            return None
        if self in sys.meta_path:
            sys.meta_path.remove(self)
        spec = importlib.util.find_spec(fullname)
        if spec is None or spec.loader is None:
            return None
        self._inner = spec.loader
        spec.loader = self
        return spec

    def create_module(self, spec):
        return self._inner.create_module(spec)

    def exec_module(self, module):
        self._inner.exec_module(module)
        _patch_urllib3_create_connection(module)


def install_happy_eyeballs_socket_connect() -> None:
    """Race IPv6/IPv4 for every sync TCP connect in the process (RFC 8305, #114265).

    The startup path does not build its HTTP clients in one place: the model catalog
    fetch goes through ``requests``/``urllib3``, sync LLM and OAuth clients through
    httpcore, plugins through ``urllib``/``http.client``. All of them funnel their TCP
    connect into ``socket.create_connection`` (``http.client`` re-reads it per connection;
    httpcore looks it up at call time) or into urllib3's own serial copy in
    ``urllib3.util.connection``. The stock implementations walk the ``getaddrinfo``
    results serially — on a network whose advertised IPv6 route is blackholed, each AAAA
    record burns the full connect timeout before IPv4 answers. Idempotent, best-effort.
    """
    if getattr(socket.create_connection, "_hermes_happy_eyeballs", False):
        return

    def _socket_racer(address, timeout=socket._GLOBAL_DEFAULT_TIMEOUT, source_address=None, *, all_errors=False):
        # Stock create_connection leaves the sentinel alone, so the socket keeps the
        # process default from socket.setdefaulttimeout(); the racer re-applies the
        # timeout on the winner, so it must resolve the sentinel the same way.
        effective = socket.getdefaulttimeout() if timeout is socket._GLOBAL_DEFAULT_TIMEOUT else timeout
        # OSError = every candidate failed (identical to the serial original); anything else is a
        # racer bug and must surface rather than silently fall back to the serial stall.
        return _happy_eyeballs_create_connection(address, effective, source_address=source_address)

    _socket_racer._hermes_happy_eyeballs = True  # type: ignore[attr-defined]
    socket.create_connection = _socket_racer

    urllib3_connection = sys.modules.get(_URLLIB3_CONNECTION_MODULE)
    if urllib3_connection is not None:
        _patch_urllib3_create_connection(urllib3_connection)
    elif not any(isinstance(finder, _Urllib3ConnectionPatcher) for finder in sys.meta_path):
        sys.meta_path.insert(0, _Urllib3ConnectionPatcher())
