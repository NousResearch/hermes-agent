"""Browser sockets the dashboard session token opens, and how they end when it is replaced.

A browser on the loopback Webapp never puts the session token in a socket URL. An authenticated
request mints a one-use ticket for one socket instead: ``GATEWAY_SOCKET`` (``/api/ws``),
``HOST_TERMINAL_SOCKET`` (``/api/host-terminal``) or one of ``AUDIO_SOCKETS`` (the voice streams). A
ticket named for one socket opens no other, and the host terminal opens only with its own ticket. A
ticket minted with the session token records that browser credential's generation
(``hermes_cli.browser_credential``) as ``session_token`` and opens nothing once the credential is
replaced, nor behind the sign-in gate.

Behind the sign-in gate the Webapp's browsers sign in instead, and each login is bound to the browser
credential (``hermes_cli.webapp_sign_in``): a ticket such a login mints records the credential's
generation as ``sign_in`` and likewise opens nothing once it is replaced.

Every socket the browser credential admits, directly or through such a ticket, is tracked here under
that generation until it ends; ``close_all``, which ``browser_credential.rotate`` runs when ``hermes
webapp --sign-out`` replaces the credential, closes them all at once. Sockets admitted on any other
credential (a native client's grant or sign-in, a dashboard login off the Webapp) are never tracked.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
import weakref
from typing import Any, Optional

from starlette.websockets import WebSocketDisconnect

from hermes_cli import browser_credential

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


def ticket_refusal(info: dict[str, Any], socket: Optional[str], *, auth_required: bool) -> Optional[str]:
    """Why a redeemed ticket may not open *socket*, or None when it may.

    A ticket named for a socket opens only that socket; an unnamed ticket opens any socket except
    one that takes only its own ticket. A ticket minted under the browser credential (with the
    session token, or by a Webapp login bound to it) opens nothing once that credential is replaced,
    and one minted with the session token opens nothing behind the sign-in gate.
    """
    named = info.get("socket")
    if named != socket and (named is not None or socket in OWN_TICKET_SOCKETS):
        return f"ticket for {named or 'any socket'} presented to {socket or 'an unnamed socket'}"
    granted = admitted_with(info)
    if granted is not None and not browser_credential.is_current(granted):
        return "ticket minted under a browser credential that has since been replaced"
    if auth_required and info.get("session_token") is not None:
        return "ticket minted with the session token, which opens nothing behind the sign-in gate"
    return None


def admitted_with(info: dict[str, Any]) -> Optional[str]:
    """The browser credential generation a redeemed ticket was minted under, which its socket is
    tracked under; None when it was minted with another credential."""
    return info.get("session_token") or info.get("sign_in")


def track(ws, admitted_with: str) -> None:
    """Keep *ws*, admitted under the browser credential generation *admitted_with*, until it ends.

    A socket admitted while that credential was being replaced closes at once, so ``close_all``
    never misses one.
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
    if not browser_credential.is_current(admitted_with):
        _close_on(ws, loop)


def admission_of(ws) -> Optional[str]:
    """The browser credential generation live socket *ws* was tracked under, or None when signing
    out does not close it. A socket it opens on the browser's behalf (a screen viewer) records the
    same one."""
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
    """Close every live socket the browser credential admitted, directly, through a ticket or through
    a login bound to it, each on its own event loop. Safe from any thread; returns how many were
    asked to close."""
    with _lock:
        live = list(_live.values())
        _live.clear()
    closing = sum(_close_on(ws, loop) for ref, loop, _digest in live if (ws := ref()) is not None)
    _log.info("Closing %d browser sockets opened with the replaced browser credential", closing)
    return closing


browser_credential.on_rotate("sockets", close_all)
