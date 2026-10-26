"""Path-bound tickets for browser file URLs (``/api/fs/download``, ``/api/files/stream``).

A download link or an ``<audio>``/``<video>`` source cannot send the session
header, and the session token grants host shells and file access, so it must
never appear in a URL the browser records (download history, "Copy video
address"). An authenticated ``POST /api/files/ticket`` mints a ticket that lets
the loopback token gate admit GET/HEAD of exactly one route + query and nothing
else. A download is one request, so its ticket is single-use; a media element
issues many Range requests for one source, so a stream ticket is reusable until
it expires. Each ticket captures the resolved file (on the profile's SSH workspace
when it has one, as for native Desktop's download) and the named profile homes it
belongs to; the resource opener refuses it once one of those homes is missing or
deleted (``hermes_constants.assert_named_profile_home_live``). A ticket minted for
a request that presented the browser session token records that credential's
generation (``hermes_cli.browser_credential``) and redeems only while it is
current, like a socket ticket, so signing out ends it even when the sign-out lands
while the ticket is being minted; signing out also drops it from the store. A
ticket minted with any other credential outlives a sign-out. In-memory only:
tickets die with the process, like the token.
"""
from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import TYPE_CHECKING, Any, Iterable, Mapping

from hermes_cli import browser_credential

if TYPE_CHECKING:
    from starlette.requests import Request

DOWNLOAD_TTL_SECONDS = 60
#: A transcript video may be played long after it mounted; seeking past the
#: buffered range needs the ticket again.
STREAM_TTL_SECONDS = 60 * 60
_MAX_TICKETS = 1024

#: A download reads the file where the profile's workspace is, like native Desktop's download;
#: media plays through the managed stream route.
ROUTES: Mapping[str, str] = {"download": "/api/fs/download", "stream": "/api/files/stream"}

_Scope = tuple[str, tuple[tuple[str, str], ...]]


@dataclass(frozen=True)
class FileAuthority:
    """Private, resolved resource identity; none of this is supplied by a ticket holder."""

    target: PurePath
    homes: tuple[Path, ...]
    #: The SSH workspace reader the file is read through (``target`` is then a path on that
    #: machine), or None for a file on this host.
    workspace: Any = None

    def check_live(self) -> None:
        """``FileNotFoundError`` once a named profile home this file belongs to is missing or deleted."""
        from hermes_constants import assert_named_profile_home_live

        for home in self.homes:
            assert_named_profile_home_live(home)


@dataclass(frozen=True)
class _Ticket:
    scope: _Scope
    expires_at: float
    single_use: bool
    authority: FileAuthority
    #: Generation of the browser credential the minting request presented
    #: (``hermes_cli.browser_credential``; None for any other credential): the ticket redeems only
    #: while it is current, so a sign-out that lands between the request's admission and the mint
    #: still ends it.
    generation: str | None = None


_lock = threading.Lock()
_tickets: dict[str, _Ticket] = {}


def _scope(route_path: str, query: Iterable[tuple[str, str]]) -> _Scope:
    return route_path, tuple(sorted(query))


def session_token_of(request: Request) -> str | None:
    """The browser session token *request* was admitted with on the loopback gate, or None when it
    came with any other credential (a sign-in, a native client's grant, a service token)."""
    from hermes_cli.web_server import _SESSION_HEADER_NAME

    if (getattr(request.app.state, "auth_required", False)
            or getattr(request.state, "native_http_principal", None)
            or getattr(request.state, "token_authenticated", False)):
        return None
    return (request.headers.get(_SESSION_HEADER_NAME)
            or request.headers.get("authorization", "").removeprefix("Bearer ") or None)


def mint(route: str, query: Mapping[str, str], *, authority: FileAuthority,
         session_token: str | None = None) -> str:
    """Ticket for GET/HEAD of ``ROUTES[route]`` with exactly ``query`` (plus ``ticket``); minted for
    a request admitted with *session_token*, it records that browser credential's generation and
    redeems only while it is current."""
    route_path = ROUTES[route]
    single_use = route == "download"
    now = time.monotonic()
    ticket = secrets.token_urlsafe(32)
    entry = _Ticket(
        scope=_scope(route_path, query.items()),
        expires_at=now + (DOWNLOAD_TTL_SECONDS if single_use else STREAM_TTL_SECONDS),
        single_use=single_use,
        authority=authority,
        generation=None if session_token is None else browser_credential.generation_of(session_token),
    )
    with _lock:
        for stale in [key for key, value in _tickets.items() if value.expires_at <= now]:
            del _tickets[stale]
        while len(_tickets) >= _MAX_TICKETS:
            del _tickets[next(iter(_tickets))]
        _tickets[ticket] = entry
    return ticket


def revoke_replaced() -> int:
    """Drop every ticket minted under a browser credential that has since been replaced (sign-out);
    tickets minted with any other credential stay. Returns how many were dropped."""
    with _lock:
        replaced = [key for key, value in _tickets.items()
                    if value.generation is not None and not browser_credential.is_current(value.generation)]
        for key in replaced:
            del _tickets[key]
    return len(replaced)


def redeem(request: Request) -> bool:
    """True when ``?ticket=`` authorizes this exact request; consumes a download ticket. A ticket
    minted under a browser credential that has since been replaced is dropped instead."""
    ticket = request.query_params.get("ticket", "")
    if not ticket or request.method not in ("GET", "HEAD"):
        return False
    scope = _scope(request.url.path, ((k, v) for k, v in request.query_params.multi_items() if k != "ticket"))
    with _lock:
        entry = _tickets.get(ticket)
        if entry is None or entry.scope != scope or entry.expires_at <= time.monotonic():
            return False
        if entry.generation is not None and not browser_credential.is_current(entry.generation):
            del _tickets[ticket]
            return False
        if entry.single_use:
            del _tickets[ticket]
    # Carry the captured authority through middleware to the resource opener.
    # A middleware-only check would race deletion before FileResponse opens it.
    request.state.file_ticket_authority = entry.authority
    return True


browser_credential.on_rotate("file tickets", revoke_replaced)
