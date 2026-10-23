"""Browser sockets the dashboard session token opens, and how they end when it is replaced.

A browser on the loopback Webapp never puts the session token in a socket URL. An authenticated
request mints a one-use ticket for one socket instead: ``GATEWAY_SOCKET`` (``/api/ws``),
``HOST_TERMINAL_SOCKET`` (``/api/host-terminal``) or one of ``AUDIO_SOCKETS`` (the voice streams). A
ticket named for one socket opens no other, and the host terminal opens only with its own ticket. A
ticket minted with the session token records that token's digest and opens nothing once the token is
replaced.

Behind the sign-in gate the Webapp's browsers sign in instead, and each login is bound to the session
token (``hermes_cli.webapp_sign_in``): a ticket such a login mints records the token's digest as
``sign_in`` and likewise opens nothing once the token is replaced.

Every socket the session token admits, directly or through such a ticket, is tracked here until it
ends, so ``close_all`` (run when ``hermes webapp --sign-out`` replaces the token) closes them all at
once. Sockets admitted on any other credential (a native client's grant, a dashboard login off the
Webapp, a provider token presented directly) are never tracked.
"""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import hmac
import logging
import threading
import weakref
from typing import Any, Optional

from starlette.websockets import WebSocketDisconnect

_log = logging.getLogger(__name__)

GATEWAY_SOCKET = "gateway"
HOST_TERMINAL_SOCKET = "host-terminal"
#: The voice streams beside ``/api/ws``, each named for its route ``/api/audio/<name>``.
SPEAK_STREAM_SOCKET = "speak-stream"
TRANSCRIBE_STREAM_SOCKET = "transcribe-stream"
AUDIO_SOCKETS = frozenset({SPEAK_STREAM_SOCKET, TRANSCRIBE_STREAM_SOCKET})
#: Sockets that open only with a ticket minted for them, never with a standing credential.
OWN_TICKET_SOCKETS = frozenset({HOST_TERMINAL_SOCKET})
#: What a signed-out tab's sockets are closed with: the same code a refused credential gets.
SIGNED_OUT_CODE = 4401

_lock = threading.Lock()
_live: dict[int, tuple[weakref.ref, asyncio.AbstractEventLoop, str]] = {}


def token_digest(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def current_digest() -> str:
    """Digest of the browser session token in use now."""
    from hermes_cli.web_server import _SESSION_TOKEN
    return token_digest(_SESSION_TOKEN)


def ticket_refusal(info: dict[str, Any], socket: Optional[str], *, auth_required: bool) -> Optional[str]:
    """Why a redeemed ticket may not open *socket*, or None when it may.

    A ticket named for a socket opens only that socket; an unnamed ticket opens any socket except
    one that takes only its own ticket. A ticket minted with the session token opens nothing behind
    the sign-in gate, nor once that token is replaced; one minted by a Webapp login opens nothing
    once the token that login was bound to is replaced.
    """
    named = info.get("socket")
    if named != socket and (named is not None or socket in OWN_TICKET_SOCKETS):
        return f"ticket for {named or 'any socket'} presented to {socket or 'an unnamed socket'}"
    minted_with = info.get("session_token")
    if minted_with is not None and (auth_required or not hmac.compare_digest(minted_with, current_digest())):
        return "ticket minted with a replaced session token"
    signed_in = info.get("sign_in")
    if signed_in is not None and not hmac.compare_digest(signed_in, current_digest()):
        return "ticket minted by a login that has since been signed out"
    return None


def admitted_with(info: dict[str, Any]) -> Optional[str]:
    """The session token digest a redeemed ticket's socket is tracked under, or None when it was
    minted with another credential."""
    return info.get("session_token") or info.get("sign_in")


def track(ws, admitted_with: str) -> None:
    """Keep *ws*, admitted on the session token whose digest is *admitted_with*, until it ends.

    A socket admitted while that token was being replaced closes at once, so ``close_all`` never
    misses one.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:  # judged outside a server loop: no live socket to close later
        return
    with _lock:
        # Ended sockets are dropped here rather than from a weakref callback, which garbage
        # collection could run while this thread already holds the lock.
        for key in [key for key, (ref, *_rest) in _live.items() if ref() is None]:
            del _live[key]
        _live[id(ws)] = (weakref.ref(ws), loop, admitted_with)
    if not hmac.compare_digest(admitted_with, current_digest()):
        _close_on(ws, loop)


def admission_of(ws) -> Optional[str]:
    """The digest live socket *ws* was tracked under, or None when signing out does not close it.
    A socket it opens on the browser's behalf (a screen viewer) is tracked under the same one."""
    with _lock:
        entry = _live.get(id(ws))
    return entry[2] if entry is not None and entry[0]() is ws else None


async def _close(ws) -> None:
    # The socket may already be closing or gone; it ends either way.
    with contextlib.suppress(RuntimeError, OSError, WebSocketDisconnect):
        await ws.close(code=SIGNED_OUT_CODE, reason="Signed out")


def _close_on(ws, loop: asyncio.AbstractEventLoop) -> bool:
    if loop.is_closed():
        return False
    asyncio.run_coroutine_threadsafe(_close(ws), loop)
    return True


def close_all() -> int:
    """Close every live socket the session token admitted, directly, through a ticket or through a
    login bound to it, each on its own event loop. Safe from any thread; returns how many were asked
    to close."""
    with _lock:
        live = list(_live.values())
        _live.clear()
    closing = sum(_close_on(ws, loop) for ref, loop, _digest in live if (ws := ref()) is not None)
    _log.info("Closing %d browser sockets opened with the replaced session token", closing)
    return closing
