"""Multi-host bind helpers for the dashboard/serve HTTP backend.

Why this exists
---------------
``hermes_cli.web_server.start_server`` historically bound a *single* interface
(``host: str``).  Operators who want the dashboard reachable over both IPv4 and
IPv6 (e.g. ``--host 0.0.0.0 --host '::'``) had no way to do it: uvicorn's
``Config(host=...)`` takes one address, and binding two listeners on one port
requires pre-creating one socket per family with ``IPV6_V6ONLY`` set so they can
coexist.

This module owns the *pure*, app-free parts of that capability — socket
pre-binding and the loopback/public classification over a *set* of bound hosts —
so ``web_server.py`` stays thin and these helpers are independently testable
(no FastAPI / starlette / uvicorn import at module level).  The request-time
Host-header validation stays in ``web_server._is_accepted_host`` because it is
tightly coupled to the ``trusted_public_hosts`` snapshot and the shared
``_host_header_hostname`` parser there; only the multi-host *decision* logic is
extracted here and re-exported by ``web_server`` (seam identity preserved).
"""

from __future__ import annotations

import socket


# Loopback host spellings treated as "local-only" for the auth gate.  Kept
# local to avoid importing web_server (which would be circular); web_server
# passes its own authoritative set into the helpers below where relevant.
_LOOPBACK_HOST_VALUES = frozenset({"127.0.0.1", "localhost", "::1"})


def create_server_sockets(hosts: list[str], port: int) -> list[socket.socket]:
    """Pre-bind one TCP socket per host entry and return them as a list.

    Each socket is created with ``SO_REUSEADDR``.  IPv6 sockets get
    ``IPV6_V6ONLY=1`` so they coexist with IPv4 listeners on the same port.
    Intended for uvicorn's ``server.startup(sockets=…)`` API which accepts
    multiple pre-created sockets natively.

    When *port* is 0, the first socket is bound to an OS-assigned ephemeral
    port; subsequent sockets reuse that same port so all listeners share a
    single port number.

    Raises ``OSError`` if any host fails to bind.  On failure, already-created
    sockets are closed before the exception propagates so we never leak fds.
    """
    socks: list[socket.socket] = []
    try:
        for h in hosts:
            family = socket.AF_INET6 if ":" in h else socket.AF_INET
            sock = socket.socket(family, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if family == socket.AF_INET6:
                sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
            sock.bind((h, port))
            if port == 0 and not socks:
                # First bind with port 0 — read the OS-assigned port so we
                # reuse it for remaining listeners instead of getting N
                # different ephemeral ports.
                port = sock.getsockname()[1]
            sock.set_inheritable(True)
            socks.append(sock)
    except OSError:
        for s in socks:
            try:
                s.close()
            except OSError:
                pass
        raise
    return socks


def close_server_sockets(sockets: list[socket.socket]) -> None:
    """Best-effort close of every socket returned by :func:`create_server_sockets`.

    Failures are swallowed — this runs on teardown paths where an individual
    already-closed socket must not mask the real shutdown flow.
    """
    for s in sockets:
        try:
            s.close()
        except OSError:
            pass


def all_hosts_loopback(hosts, loopback_values=_LOOPBACK_HOST_VALUES) -> bool:
    """True iff EVERY bound host is a loopback spelling.

    A dual-stack loopback bind (``["127.0.0.1", "::1"]``) is still fully local,
    so the WS keepalive-ping-disabled / peer-loopback-gate behaviour applies.
    Any non-loopback member makes the whole bind public.
    """
    normalized = [str(h).strip().lower() for h in hosts]
    return all(h in loopback_values for h in normalized)


def any_host_requires_auth(
    hosts, allow_public: bool = False, should_require_auth=None
) -> bool:
    """True iff the auth gate engages for ANY of the bound hosts.

    Delegates the per-host decision to ``should_require_auth`` (injected so this
    module never imports ``web_server`` at load time — circular-import safe).
    ``allow_public`` is accepted for signature parity with the legacy escape
    hatch but, matching current policy, does NOT disable the gate.
    """
    if should_require_auth is None:
        # Lazy import keeps this module standalone-importable while giving a
        # correct default when called from within the running app.
        from hermes_cli.web_server import should_require_auth as _default

        should_require_auth = _default
    return any(should_require_auth(h, allow_public) for h in hosts)
