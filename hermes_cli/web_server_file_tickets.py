"""Path-bound tickets for browser file URLs (``/api/files/download|stream``).

A download link or an ``<audio>``/``<video>`` source cannot send the session
header, and the session token grants host shells and file access, so it must
never appear in a URL the browser records (download history, "Copy video
address"). An authenticated ``POST /api/files/ticket`` mints a ticket that lets
the loopback token gate admit GET/HEAD of exactly one route + query and nothing
else. A download is one request, so its ticket is single-use; a media element
issues many Range requests for one source, so a stream ticket is reusable until
it expires. In-memory only: tickets die with the process, like the token.
"""
from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Mapping

if TYPE_CHECKING:
    from starlette.requests import Request

DOWNLOAD_TTL_SECONDS = 60
#: A transcript video may be played long after it mounted; seeking past the
#: buffered range needs the ticket again.
STREAM_TTL_SECONDS = 60 * 60
_MAX_TICKETS = 1024

ROUTES: Mapping[str, str] = {"download": "/api/files/download", "stream": "/api/files/stream"}

_Scope = tuple[str, tuple[tuple[str, str], ...]]


@dataclass(frozen=True)
class _Ticket:
    scope: _Scope
    expires_at: float
    single_use: bool


_lock = threading.Lock()
_tickets: dict[str, _Ticket] = {}


def _scope(route_path: str, query: Iterable[tuple[str, str]]) -> _Scope:
    return route_path, tuple(sorted(query))


def mint(route: str, query: Mapping[str, str]) -> str:
    """Ticket for GET/HEAD of ``ROUTES[route]`` with exactly ``query`` (plus ``ticket``)."""
    route_path = ROUTES[route]
    single_use = route == "download"
    now = time.monotonic()
    ticket = secrets.token_urlsafe(32)
    entry = _Ticket(
        scope=_scope(route_path, query.items()),
        expires_at=now + (DOWNLOAD_TTL_SECONDS if single_use else STREAM_TTL_SECONDS),
        single_use=single_use,
    )
    with _lock:
        for stale in [key for key, value in _tickets.items() if value.expires_at <= now]:
            del _tickets[stale]
        while len(_tickets) >= _MAX_TICKETS:
            del _tickets[next(iter(_tickets))]
        _tickets[ticket] = entry
    return ticket


def redeem(request: Request) -> bool:
    """True when ``?ticket=`` authorizes this exact request; consumes a download ticket."""
    ticket = request.query_params.get("ticket", "")
    if not ticket or request.method not in ("GET", "HEAD"):
        return False
    scope = _scope(request.url.path, ((k, v) for k, v in request.query_params.multi_items() if k != "ticket"))
    with _lock:
        entry = _tickets.get(ticket)
        if entry is None or entry.scope != scope or entry.expires_at <= time.monotonic():
            return False
        if entry.single_use:
            del _tickets[ticket]
    return True
