"""``GET /api/session-attach`` — the owner half of cooperative session attach.

A local ``hermes --tui --resume <id>`` whose session this process owns (``hermes_cli/
shared_session_attach.py`` found our ``shared_runtime_url`` on the lease) asks here for the
WebSocket it should dial. We answer with this process's internal ``/api/ws`` URL — the same
credential the embedded dashboard TUI child receives through its environment — so the TUI
attaches to the LIVE session beside the Desktop/dashboard client instead of starting a second
runtime (``methods_session._resume_reuse_live``).

The boundary is the same-user loopback machine, like the lease registry file itself:

- the peer must be a loopback address and the request must not come from a browser (an
  ``Origin`` header means a page fetched us cross-origin; curl/httpx never send one);
- ``session_id``, ``lease_id`` and ``profile_home`` must name a lease THIS pid holds, and that
  lease must advertise our origin — a guess at a session id learns nothing without the random
  lease id from the registry file, and another process's lease is not ours to hand out.

Anything else is a 404 with no detail. Unauthenticated on purpose (``PUBLIC_API_PATHS``): the
caller is a fresh local process with no dashboard identity; loopback + lease identity is the gate.
"""
from __future__ import annotations

import asyncio
import ipaddress
import logging
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

_log = logging.getLogger("hermes_cli.web_server")
router = APIRouter()

PATH = "/api/session-attach"


def _loopback_peer(request: Request) -> bool:
    host = (request.client.host if request.client else "") or ""
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _owned_lease(session_id: str, lease_id: str, profile_home: str, origin: str) -> bool:
    """True when this process holds exactly that lease and advertises ``origin`` on it.
    Reads the registry file, so callers run it off the event loop."""
    from hermes_cli.active_sessions import active_session_registry_snapshot

    try:
        entries = active_session_registry_snapshot(Path(profile_home), strict=True)
    except OSError:
        # An unreadable or absent registry for that home is simply "not ours".
        _log.debug("session-attach: registry unreadable for %s", profile_home, exc_info=True)
        return False
    for entry in entries:
        if entry.get("session_id") != session_id or entry.get("lease_id") != lease_id:
            continue
        try:
            if int(entry.get("pid") or -1) != os.getpid():
                return False
        except (TypeError, ValueError):
            return False
        return (entry.get("metadata") or {}).get("shared_runtime_url") == origin
    return False


def _fenced_identity(session_id: str, lease_id: str, profile_home: str, origin: str) -> bool:
    """Everything the handshake checks that touches the filesystem (resolve + registry read)."""
    if str(Path(profile_home).resolve()) != profile_home:
        # The client sends its resolved home; anything else is not the shape we fenced.
        return False
    return _owned_lease(session_id, lease_id, profile_home, origin)


@router.get(PATH)
async def session_attach(request: Request, session_id: str = "", lease_id: str = "", profile_home: str = ""):
    from hermes_cli.web_server_chat import _build_gateway_ws_url
    from tui_gateway.shared_runtime import shared_runtime_origin

    origin = shared_runtime_origin()
    refused = HTTPException(status_code=404)
    if origin is None or not _loopback_peer(request) or request.headers.get("origin") is not None:
        raise refused
    if not session_id or not lease_id or not profile_home:
        raise refused
    if not await asyncio.to_thread(_fenced_identity, session_id, lease_id, profile_home, origin):
        raise refused
    websocket_url = _build_gateway_ws_url()
    if not websocket_url:
        raise refused
    return {"session_id": session_id, "lease_id": lease_id, "profile_home": profile_home,
            "websocket_url": websocket_url}
